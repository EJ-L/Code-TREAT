from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple, Iterable, Union, Set
import threading
import concurrent.futures
from collections import defaultdict
import json
import os
import time
import random
import traceback
from datetime import datetime
from .normalization_utils import normalize_categories, normalize_batch
from .data_manager import DatasetManager
from .record_manager import RecordManager
from .template_manager import TemplateManager
from models.model_list import MODELS
from dataclasses import dataclass

@dataclass
class TaskConfig:
    """Configuration for task execution."""
    models: List[str]
    dataset: str
    language: str          # not used by translation, but kept for uniformity
    template_categories: Optional[List[str]] = None
    test_samples: Optional[int] = None             # legacy deterministic slice
    n_requests: int = 1
    parallel_requests: Optional[int] = None
    save_dir: str = "results"
    reproduce: bool = True

    # --- Sampling / Replication controls ---
    sampling_mode: Optional[str] = None            # "random" | "manifest" | None
    sample_size: Optional[int] = None              # used when sampling_mode == "random"
    sampling_seed: Optional[int] = 42              # deterministic random sampling
    sampling_manifest_path: Optional[str] = None   # where to write/read sampled items json

class BaseTaskRunner(ABC):
    """Abstract base class for task runners using persisted ref_key and sampling manifests."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.dataset_manager = DatasetManager(
            self.get_task_name(),
            self.config.dataset,
            self.config.language,
            reproduce=self.config.reproduce,
        )
        # Normalize dataset name for record manager to unify similar datasets
        normalized_dataset = self._get_normalized_dataset_name(self.config.dataset)
        self.record_manager = RecordManager(self.get_task_name(), normalized_dataset, self.config.save_dir)
        self.dataset: List[Any] = []
        self._mem_lock = threading.Lock()

    # ---- Dataset normalization ----------------------------------------------
    
    def _get_normalized_dataset_name(self, dataset_name: str) -> str:
        """Normalize dataset names to unify related datasets under same output folder"""
        if self.get_task_name() == "vulnerability_detection":
            # Map both primevul variants to same base name
            if dataset_name in ["primevul", "primevul_pairs", "primevul_pair"]:
                return "primevul"
        return dataset_name

    # ---- Abstracts ----------------------------------------------------------

    @abstractmethod
    def get_task_name(self):
        raise NotImplementedError

    @abstractmethod
    def create_work_items(self, dataset: List[Any], templates: List[Any], models: List[Any]) -> List[Tuple]:
        raise NotImplementedError

    @abstractmethod
    def process_work_item(self, work_item: Tuple) -> Dict[str, Any]:
        raise NotImplementedError

    # ---- Shared Fetch -------------------------------------------------------

    def _fetch_once_with_backoff(self, func, max_retries: int = 3) -> Any:
        for i in range(max_retries + 1):
            try:
                print(f"API call attempt {i+1}...", flush=True)
                result = func()
                print(f"API call SUCCESS!", flush=True)
                return result
            except Exception as e:
                print(f"API call failed: {e}", flush=True)
                if i >= max_retries:
                    raise
                sleep_time = (2 ** i) + random.random()
                print(f"Retrying in {sleep_time:.1f}s...", flush=True)
                time.sleep(sleep_time)

    def fetch_responses_parallel(
        self,
        model: Any,
        message: Any,
        n_requests: int
    ) -> Tuple[List[str], List[str]]:
        if int(n_requests) <= 1:
            resp = self._fetch_once_with_backoff(lambda: model.chat(message, n=1))
            out = normalize_batch(resp)
            return out

        def fetch_single_response() -> List[str]:
            resp = self._fetch_once_with_backoff(lambda: model.chat(message, n=1))
            out = normalize_batch(resp)
            return out

        remaining = max(1, int(n_requests))
        response_list: List[str] = []
        attempts = 0

        while remaining > 0 and attempts < max(10, n_requests * 2):
            attempts += 1
            batch_size = min(5, remaining)
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                results = list(executor.map(lambda _: fetch_single_response(), range(batch_size)))
            for resp in results:
                response_list.extend(resp)

        return response_list

    # ---- Sampling manifest logic -------------------------------------------

    def _manifest_default_path(self) -> str:
        dataset_name = str(self.config.dataset) if self.config.dataset is not None else "dataset"
        dataset_safe = dataset_name.replace(os.sep, "_").replace("/", "_")
        fname = f"{dataset_safe}_sample_manifest.json"
        return os.path.join(self.config.save_dir, self.get_task_name(), fname)

    def _save_sampling_manifest(self, items: List[Any]):
        """Save the sampled dataset metadata so the experiment can be replayed."""
        path = self.config.sampling_manifest_path or self._manifest_default_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        existing_items: List[Dict[str, Any]] = []
        dataset_stats: Dict[str, Dict[str, Any]] = {}

        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    manifest_data = json.load(f)
                if manifest_data.get("task") == self.get_task_name():
                    existing_items = manifest_data.get("items", []) or []
                    dataset_stats = manifest_data.get("dataset_stats", {}) or {}
            except Exception:
                existing_items = []
                dataset_stats = {}

        current_dataset_names: Set[str] = set()
        current_entries: List[Dict[str, Any]] = []

        for d in items:
            dataset_name = getattr(d, "dataset_name", getattr(d, "dataset", None))
            if dataset_name is None:
                dataset_name = "unknown"
            elif not isinstance(dataset_name, str):
                dataset_name = str(dataset_name)
            current_dataset_names.add(dataset_name)
            current_entries.append({
                "dataset": dataset_name,
                "id": self._resolve_item_id(d),
            })

        # Drop stale entries for datasets included in this run
        filtered_existing = [
            entry for entry in existing_items
            if entry.get("dataset") not in current_dataset_names
        ]

        merged_items = filtered_existing + current_entries

        total_candidates_current = len(self.dataset)
        timestamp = datetime.utcnow().isoformat() + "Z"

        for dataset_name in current_dataset_names:
            dataset_stats[dataset_name] = {
                "total_candidates": total_candidates_current,
                "sample_size": sum(1 for entry in current_entries if entry.get("dataset") == dataset_name),
                "sampling_mode": "random",
                "sampling_seed": self.config.sampling_seed,
                "timestamp": timestamp,
            }

        combined_sample_size = len(merged_items)
        combined_total_candidates = sum(stat.get("total_candidates", 0) for stat in dataset_stats.values())

        manifest = {
            "task": self.get_task_name(),
            "timestamp": timestamp,
            "total_candidates": combined_total_candidates,
            "sample_size": combined_sample_size,
            "sampling_mode": "random",
            "sampling_seed": self.config.sampling_seed,
            "items": merged_items,
            "dataset_stats": dataset_stats,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[{self.get_task_name()}] Wrote sampling manifest to: {path}")

    @staticmethod
    def _resolve_item_id(item: Any) -> Any:
        """Attempt to extract a stable identifier from a dataset item."""
        for attr in ("idx", "key", "id", "composite_id", "prompt_id"):
            if hasattr(item, attr):
                value = getattr(item, attr, None)
                if value is not None:
                    return value
        return None

    def _load_sampling_manifest(self) -> Optional[List[Dict[str, Any]]]:
        """Load a previously saved manifest and return item descriptors."""
        path = self.config.sampling_manifest_path or self._manifest_default_path()
        if not os.path.exists(path):
            print(f"[{self.get_task_name()}] Manifest not found at: {path}")
            return None
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("task") != self.get_task_name():
            print(f"[{self.get_task_name()}] Manifest task mismatch; ignoring file: {path}")
            return None
        return manifest.get("items", [])

    def _apply_sampling(self):
        """
        Apply sampling/replay according to TaskConfig.
        Priority: manifest > random sample_size > test_samples (deterministic slice).
        """
        mode = (self.config.sampling_mode or "").lower().strip()
        if mode == "manifest":
            entries = self._load_sampling_manifest()
            if entries:
                # Filter dataset to only those (dataset, id) pairs
                wanted = {(e.get("dataset"), e.get("id")) for e in entries}
                before = len(self.dataset)
                # Enhanced filtering for different data types
                filtered_dataset = []
                for d in self.dataset:
                    dataset_name = getattr(d, "dataset_name", getattr(d, "dataset", None))
                    
                    # Try different ID matching strategies based on data type
                    data_id = None
                    
                    # Strategy 1: Use idx if available (standard approach)
                    if hasattr(d, "idx"):
                        data_id = getattr(d, "idx", None)

                    # Strategy 2: Prefer explicit key attribute for custom identifiers
                    if data_id is None and hasattr(d, "key"):
                        data_id = getattr(d, "key", None)
                    
                    # Strategy 3: For code review data, try composite_id matching
                    if data_id is None and hasattr(d, "composite_id"):
                        composite_id = getattr(d, "composite_id", None)
                        # Check if any manifest entry's ID matches this composite_id
                        for entry in entries:
                            if entry.get("id") == composite_id and entry.get("dataset") == dataset_name:
                                data_id = composite_id
                                break
                    
                    # Strategy 4: Fallback to other common ID attributes
                    if data_id is None:
                        for attr in ["key", "id", "idx", "prompt_id"]:
                            if hasattr(d, attr):
                                data_id = getattr(d, attr, None)
                                break
                    
                    # Check if this item should be included
                    if (dataset_name, data_id) in wanted:
                        filtered_dataset.append(d)
                
                self.dataset = filtered_dataset
                print(f"[{self.get_task_name()}] Manifest mode: kept {len(self.dataset)}/{before} items.")
            else:
                print(f"[{self.get_task_name()}] Manifest mode requested but no entries loaded; proceeding without filtering.")
            return

        if mode == "random" and isinstance(self.config.sample_size, int) and self.config.sample_size > 0:
            pop = len(self.dataset)
            k = min(self.config.sample_size, pop)
            rng = random.Random(self.config.sampling_seed)
            sampled = rng.sample(self.dataset, k) if k < pop else list(self.dataset)
            self._save_sampling_manifest(sampled)
            self.dataset = sampled
            print(f"[{self.get_task_name()}] Random sampling: kept {len(self.dataset)}/{pop} items (seed={self.config.sampling_seed}).")
            return

        # Fallback to legacy deterministic slice if specified
        if self.config.test_samples is not None:
            n = int(self.config.test_samples)
            before = len(self.dataset)
            self.dataset = self.dataset[:n]
            print(f"[{self.get_task_name()}] Deterministic slice: kept {len(self.dataset)}/{before} items.")

    # ---- Execution Orchestration -------------------------------------------

    def execute_task(self):
        # Load datasets
        self.dataset = self.dataset_manager.load_dataset()
        print(f"[{self.get_task_name()}] Loaded {len(self.dataset)} dataset items.")

        # Load models
        models = []
        for m in self.config.models:
            if m not in MODELS:
                print(f"[{self.get_task_name()}] Warning: model '{m}' not found in MODELS; skipping.")
                continue
            models.append(MODELS[m])

        legacy = self.record_manager.load_records(models)

        # Load templates for this task
        template_manager = TemplateManager(self.config.template_categories or ["direct"])
        templates = template_manager.load_templates(self.get_task_name(), language=self.config.language, dataset=self.config.dataset)
        print(f"[{self.get_task_name()}] Loaded {len(templates)} templates.")

        # Apply sampling / replay (AFTER full load, BEFORE work item creation)
        self._apply_sampling()

        # Create work items
        work_items = self.create_work_items(self.dataset, templates, models)
        print(f"[{self.get_task_name()}] Created {len(work_items)} work items.")

        # Execute outer parallelism
        results: List[Dict[str, Any]] = []
        if self.config.parallel_requests and self.config.parallel_requests > 1:
            max_workers = int(self.config.parallel_requests)
            print(f"[{self.get_task_name()}] Running with outer parallelism: {max_workers} workers.")
            print(f"[{self.get_task_name()}] Starting processing of {len(work_items)} work items...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                for i, res in enumerate(pool.map(self.process_work_item, work_items), 1):
                    # Show progress more frequently
                    if i % 10 == 0 or i == 1:
                        print(f"[{self.get_task_name()}] Progress: {i}/{len(work_items)} ({i/len(work_items)*100:.1f}%)")
                    results.append(res)
        else:
            print(f"[{self.get_task_name()}] Running serially.")
            print(f"[{self.get_task_name()}] Starting processing of {len(work_items)} work items...")
            for i, item in enumerate(work_items, 1):
                if i % 10 == 0 or i == 1:
                    print(f"[{self.get_task_name()}] Progress: {i}/{len(work_items)} ({i/len(work_items)*100:.1f}%)")
                results.append(self.process_work_item(item))

        # Summary
        completed = sum(1 for r in results if r.get("status") == "completed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        errors = sum(1 for r in results if r.get("status") == "error")
        print(f"[{self.get_task_name()}] Summary -> completed: {completed}, skipped: {skipped}, errors: {errors}")
