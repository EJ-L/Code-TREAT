# task_runners.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple, Iterable, Union
import threading
import concurrent.futures
from collections import defaultdict
import json
import os
import time
import random
import traceback
from datetime import datetime

class DatasetManager:
    """Manages loading of datasets from a provided registry."""
    def __init__(self, dataset_registry: Dict[str, Any]):
        self.registry = dataset_registry

    def load_datasets(self, dataset_specs: Dict[str, str]) -> List[Any]:
        dataset: List[Any] = []
        for name, domain in dataset_specs.items():
            if name not in self.registry:
                print(f"[DatasetManager] Warning: dataset '{name}' not in registry; skipping.")
                continue
            try:
                obj = self.registry[name]()
                dataset.extend(obj.load_dataset(domain))
            except Exception as e:
                print(f"[DatasetManager] Error loading dataset '{name}': {e}")
        return dataset


class RecordManager:
    """Manages loading/saving of records with a persisted composite ref_key + flexible builder.
       Files live under: save_dir/<task>/<dataset>/<model_name>_hr.jsonl
       (with backward-compatible reads from legacy flat files if found).
    """

    def __init__(self, task_name: str, save_dir: str = "save"):
        self.task_name = task_name
        self.save_dir = save_dir
        self._lock = threading.Lock()
        # Fast O(1) skip-check index: ref_key -> True
        self.ref_index: Dict[str, bool] = {}
        # Optional legacy map for introspection/debug
        self.records_by_data_id: Dict[Any, set] = defaultdict(set)

    # ---------- Path helpers ----------

    def _dir_for(self, dataset_name: str) -> str:
        # save_dir/<task>/<dataset>
        return os.path.join(self.save_dir, self.task_name, dataset_name)

    def _file_path(self, model_name: str, dataset_name: str) -> str:
        model_name_fs = model_name.replace("/", "_")
        return os.path.join(self._dir_for(dataset_name), f"{model_name_fs}_hr.jsonl")

    def _legacy_file_path(self, model_name: str) -> str:
        """Backward-compat (old flat layout): save_dir/<task>_<model>_hr.jsonl"""
        model_name_fs = model_name.replace("/", "_")
        return os.path.join(self.save_dir, f"{self.task_name}_{model_name_fs}_hr.jsonl")

    # ---------- Flexible ref key builder ----------

    @staticmethod
    def make_ref_key(*parts: Any, _sep: str = "_") -> str:
        """Flexible ref key: numbers->str(num), list/tuple->'__'.join(str(x)), strings->as-is, else str(x)."""
        norm: List[str] = []
        for p in parts:
            if p is None:
                norm.append("None")
            elif isinstance(p, (int, float)):
                norm.append(str(p))
            elif isinstance(p, (list, tuple)):
                norm.append("__".join(str(x) for x in p))
            else:
                norm.append(str(p))
        return _sep.join(norm)

    def compose_ref_key_from_template(self, dataset: str, lang: str, template: Any, model_name: str) -> Optional[str]:
        """Build ref_key using fields on the template (category + template_id)."""
        categories = _normalize_categories(getattr(template, "category", []))
        prompt_id = getattr(template, "template_id", None)
        if prompt_id is None:
            return None
        return self.make_ref_key(dataset, lang, categories, model_name, prompt_id)

    def compose_ref_key_from_record_fields(self, record: Dict[str, Any]) -> Optional[str]:
        """Fallback for old records that lack ref_key."""
        dataset = record.get("dataset")
        lang = record.get("lang") or record.get("target_lang")
        categories = _normalize_categories(record.get("prompt_category") or record.get("prompting_category") or [])
        model_name = record.get("model_name")
        prompt_id = record.get("prompt_id")
        if None in (dataset, lang, model_name, prompt_id):
            return None
        return self.make_ref_key(dataset, lang, categories, model_name, prompt_id)

    # ---------- Load / Save ----------

    def load_records(self, models: List[Any], dataset_names: Optional[List[str]] = None) -> Dict[Any, set]:
        """
        Populate ref_index from saved JSONL files under save_dir/<task>/<dataset>/.
        Also falls back to legacy flat files if present.
        """
        self.ref_index.clear()
        self.records_by_data_id.clear()

        # Ensure directory roots exist
        os.makedirs(self.save_dir, exist_ok=True)

        # If no dataset_names provided, try to discover by listing the task dir
        if not dataset_names:
            task_root = os.path.join(self.save_dir, self.task_name)
            if os.path.isdir(task_root):
                dataset_names = [d for d in os.listdir(task_root) if os.path.isdir(os.path.join(task_root, d))]
            else:
                dataset_names = []

        for model in models:
            model_name = getattr(model, "model_name", None)
            if not model_name:
                print("[RecordManager] Warning: model without 'model_name' found; skipping record load for it.")
                continue

            # 1) Load per-dataset files (new layout)
            for ds in dataset_names:
                fname = self._file_path(model_name, ds)
                if not os.path.exists(fname):
                    continue
                self._load_one_file_into_index(fname)

            # 2) Also attempt legacy file (flat) for backward compatibility
            legacy_path = self._legacy_file_path(model_name)
            if os.path.exists(legacy_path):
                self._load_one_file_into_index(legacy_path)

        return self.records_by_data_id

    def _load_one_file_into_index(self, filepath: str):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    # Primary: existing ref_key, else compute
                    key = record.get("ref_key") or self.compose_ref_key_from_record_fields(record)
                    if key:
                        self.ref_index[key] = True

                    # Legacy in-memory map (optional)
                    data_idx = record.get("id")
                    mname = record.get("model_name")
                    lang = record.get("lang") or record.get("target_lang")
                    template = record.get("prompt_template")
                    if all(x is not None for x in [data_idx, mname, lang, template]):
                        self.records_by_data_id[data_idx].add((mname, lang, template))
        except Exception as e:
            print(f"[RecordManager] Warning: error loading '{filepath}': {e}")

    def is_ref_processed(self, ref_key: str) -> bool:
        return ref_key in self.ref_index

    def mark_ref_processed(self, ref_key: str):
        self.ref_index[ref_key] = True

    def attach_ref_key(self, record: Dict[str, Any]) -> Optional[str]:
        """Ensure record['ref_key'] exists (compute if needed) and add it to index."""
        key = record.get("ref_key")
        if not key:
            key = self.compose_ref_key_from_record_fields(record)
            if key:
                record["ref_key"] = key
        if key:
            self.mark_ref_processed(key)
        return key

    def save_record(self, model_name: str, record: Dict[str, Any], fsync: bool = False):
        """
        Save under: save_dir/<task>/<dataset>/<model>_hr.jsonl
        Derives `dataset` from the record. Also writes ref_key and updates index.
        """
        dataset_name = record.get("dataset")
        if not dataset_name:
            raise ValueError("Record missing 'dataset' — required to determine per-dataset path.")

        dirpath = self._dir_for(dataset_name)
        os.makedirs(dirpath, exist_ok=True)
        file_path = self._file_path(model_name, dataset_name)

        if "_schema_version" not in record:
            record["_schema_version"] = 1

        # Ensure ref_key is present and indexed
        self.attach_ref_key(record)

        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                if fsync:
                    f.flush()
                    os.fsync(f.fileno())


# =========================
# Base runner (sampling + ref_key)
# =========================

class BaseTaskRunner(ABC):
    """Abstract base class for task runners using persisted ref_key and sampling manifests."""

    def __init__(self, config: TaskConfig):
        self.config = config
        self.dataset_manager = DatasetManager(self.get_dataset_registry())
        self.record_manager = RecordManager(self.get_task_name(), config.save_dir)
        self.dataset: List[Any] = []
        self._mem_lock = threading.Lock()

    # ---- Abstracts ----------------------------------------------------------

    @abstractmethod
    def get_task_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_registry(self) -> Dict[str, Any]:
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
                return func()
            except Exception:
                if i >= max_retries:
                    raise
                time.sleep((2 ** i) + random.random())

    def fetch_responses_parallel(
        self,
        model: Any,
        message: Any,
        language: str,
        n_requests: int
    ) -> Tuple[List[str], List[str]]:
        if int(n_requests) <= 1:
            resp = self._fetch_once_with_backoff(lambda: model.chat(message, n=1))
            out = _normalize_batch(resp)
            return out, out

        def fetch_single_response() -> Tuple[List[str], List[str]]:
            resp = self._fetch_once_with_backoff(lambda: model.chat(message, n=1))
            out = _normalize_batch(resp)
            return out, out

        remaining = max(1, int(n_requests))
        response_list: List[str] = []
        extracted_output_list: List[str] = []
        attempts = 0

        while remaining > 0 and attempts < max(10, n_requests * 2):
            attempts += 1
            batch_size = min(5, remaining)
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                results = list(executor.map(lambda _: fetch_single_response(), range(batch_size)))
            for resp, out in results:
                got = len([o for o in out if o])
                remaining = max(0, remaining - got)
                response_list.extend(resp)
                extracted_output_list.extend(out)

        return response_list, extracted_output_list

    # ---- Templates / Models -------------------------------------------------

    def _load_template_manager(self):
        try:
            from templates.Template import TemplateManager  # preferred
            return TemplateManager
        except Exception:
            try:
                from Template import TemplateManager       # fallback (legacy)
                return TemplateManager
            except Exception as e:
                raise ImportError(
                    "Cannot import TemplateManager from either 'templates.Template' or 'Template'."
                ) from e

    def _load_models_registry(self):
        try:
            from models.model_list import MODELS
            return MODELS
        except Exception as e:
            raise ImportError(
                "Cannot import MODELS from 'models.model_list'. "
                "Please ensure your model registry exists and is importable."
            ) from e

    # ---- Sampling manifest logic -------------------------------------------

    def _manifest_default_path(self) -> str:
        fname = f"{self.get_task_name()}_sample_manifest.json"
        return os.path.join(self.config.save_dir, fname)

    def _save_sampling_manifest(self, items: List[Any]):
        """Save the sampled dataset metadata so the experiment can be replayed."""
        path = self.config.sampling_manifest_path or self._manifest_default_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        manifest = {
            "task": self.get_task_name(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_candidates": len(self.dataset),
            "sample_size": len(items),
            "sampling_mode": "random",
            "sampling_seed": self.config.sampling_seed,
            "items": [
                {"dataset": getattr(d, "dataset", None), "id": getattr(d, "id", None)}
                for d in items
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(f"[{self.get_task_name()}] Wrote sampling manifest to: {path}")

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
                self.dataset = [d for d in self.dataset if (getattr(d, "dataset", None), getattr(d, "id", None)) in wanted]
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
        TemplateManager = self._load_template_manager()
        MODELS = self._load_models_registry()

        # Load datasets
        self.dataset = self.dataset_manager.load_datasets(self.config.datasets)
        print(f"[{self.get_task_name()}] Loaded {len(self.dataset)} dataset items.")

        # Load models
        models = []
        for m in self.config.models:
            if m not in MODELS:
                print(f"[{self.get_task_name()}] Warning: model '{m}' not found in MODELS; skipping.")
                continue
            models.append(MODELS[m])

        # Build ref_index from disk (per-dataset layout + legacy flat files)
        dataset_names = list(self.config.datasets.keys())
        legacy = self.record_manager.load_records(models, dataset_names=dataset_names)
        print(f"[{self.get_task_name()}] Loaded {len(self.record_manager.ref_index)} ref_keys; "
              f"{len(legacy)} legacy data_id entries.")

        # Load templates for this task
        template_manager = TemplateManager(self.config.template_categories or ["direct"])
        templates = template_manager.load_templates(self.get_task_name().replace("_", " "))
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
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                for i, res in enumerate(pool.map(self.process_work_item, work_items), 1):
                    if i % 100 == 0:
                        print(f"[{self.get_task_name()}] Progress: {i}/{len(work_items)}")
                    results.append(res)
        else:
            print(f"[{self.get_task_name()}] Running serially.")
            for i, item in enumerate(work_items, 1):
                if i % 100 == 0:
                    print(f"[{self.get_task_name()}] Progress: {i}/{len(work_items)}")
                results.append(self.process_work_item(item))

        # Summary
        completed = sum(1 for r in results if r.get("status") == "completed")
        skipped = sum(1 for r in results if r.get("status") == "skipped")
        errors = sum(1 for r in results if r.get("status") == "error")
        print(f"[{self.get_task_name()}] Summary -> completed: {completed}, skipped: {skipped}, errors: {errors}")


# =========================
# Code Generation Runner
# =========================

class CodeGenerationRunner(BaseTaskRunner):
    """Unified runner for code generation tasks."""

    def __init__(self, config: TaskConfig, problem_ids: Optional[Iterable[int]] = None):
        super().__init__(config)
        self.problem_ids = set(problem_ids) if problem_ids else None

    def get_task_name(self) -> str:
        return "code_generation"

    def get_dataset_registry(self) -> Dict[str, Any]:
        try:
            from tests.code_generation.Dataset import Geeksforgeeks, Hackerrank, LCB
            return {"Geeksforgeeks": Geeksforgeeks, "Hackerrank": Hackerrank, "LCB": LCB}
        except Exception as e:
            print(f"[CodeGenerationRunner] Warning: Could not import datasets: {e}")
            return {}

    def create_work_items(self, dataset: List[Any], templates: List[Any], models: List[Any]) -> List[Tuple]:
        work_items: List[Tuple] = []
        # Optional filtering by problem IDs (applied post-sampling too)
        if self.problem_ids is not None:
            dataset = [d for d in dataset if getattr(d, "id", None) in self.problem_ids]
        for data in dataset:
            for language in self.config.languages:
                for model in models:
                    for template in templates:
                        work_items.append((data, language, model, template))
        return work_items

    def process_work_item(self, work_item: Tuple) -> Dict[str, Any]:
        data, language, model, template = work_item
        model_name = getattr(model, "model_name", "unknown_model")

        try:
            # Build ref_key via helper (no runner glue)
            ref_key = self.record_manager.compose_ref_key_from_template(
                dataset=data.dataset, lang=language, template=template, model_name=model_name
            )
            if ref_key is None:
                return {"status": "skipped", "reason": "no_prompt_id"}

            if self.record_manager.is_ref_processed(ref_key):
                return {"status": "skipped", "reason": "already_processed"}

            # Require function signature for this language
            function_signatures = data.function_signatures.get(language, None)
            if not function_signatures:
                return {"status": "skipped", "reason": "no_signature"}

            # Build prompt
            message, wrapped_text = self._build_prompt(data, language, template, model)

            # Fetch responses
            response_list, extracted_output_list = self.fetch_responses_parallel(
                model, message, language, self.config.n_requests
            )

            categories = _normalize_categories(getattr(template, "category", []))
            prompt_id = getattr(template, "template_id", None)

            # Assemble record (include ref_key)
            result = {
                "task": "code generation",
                "ref_key": ref_key,
                "lang": language,
                "dataset": data.dataset,
                "domain": data.domain,
                "id": data.id,
                "prompt_category": categories,
                "prompt_id": prompt_id,
                "prompt_template": template.template_string,
                "wrapped_text": wrapped_text,
                "model_name": model_name,
                "output": response_list,
                "extracted_output": extracted_output_list,
                "driver_code": data.driver_code.get(language, ""),
                "script_name": getattr(data, "java_script_name", None) if language == "java" else None,
                "func_sign": function_signatures,
                "metrics": {},
            }

            self.record_manager.save_record(model_name, result)
            return {"status": "completed"}

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _build_prompt(self, data: Any, language: str, template: Any, model: Any) -> Tuple[Any, str]:
        starter_code = data.starter_code.get(language, "# YOUR CODE HERE")

        if starter_code and starter_code not in ("# YOUR CODE HERE", ""):
            starter_code_msg = (
                f"### Format: You will use the following starter code to write the solution "
                f"and enclose your code within delimiters.\n```{language}\n{starter_code}\n```\n\n"
            )
        else:
            starter_code_msg = (
                f"### Format: Read the inputs from stdin solve the problem and write the answer "
                f"to stdout. Enclose your code within delimiters as follows.\n```{language}\n"
                f"# YOUR CODE HERE\n```\n\n"
            )

        class_name = data.starter_code_class.get(language, None)
        if class_name and language.lower() == "java":
            class_msg = f"under the class `{class_name}` without public static void main(String[] args) in it."
        else:
            class_msg = ""

        function_signatures = data.function_signatures.get(language, [])
        function_signature = function_signatures[0] if len(function_signatures) == 1 else ", ".join(function_signatures)

        template_info = {
            "PL": language,
            "problem_description": data.problem_description,
            "function_signatures": function_signature,
            "class_msg": class_msg,
            "starter_code_msg": starter_code_msg,
        }

        return template.get_prompt(model=model, is_chat_mode=False, **template_info)


# =========================
# Code Translation Runner
# =========================

class CodeTranslationRunner(BaseTaskRunner):
    """Unified runner for code translation tasks."""

    def __init__(self, config: TaskConfig, source_languages: List[str], target_languages: List[str]):
        super().__init__(config)
        self.source_languages = [lang.lower() for lang in source_languages]
        self.target_languages = [lang.lower() for lang in target_languages]

    def get_task_name(self) -> str:
        return "code_translation"

    def get_dataset_registry(self) -> Dict[str, Any]:
        try:
            from tests.code_translation.Dataset import Hackerrank, PolyHumanEval, CodeTransOcean
            return {"Hackerrank": Hackerrank, "PolyHumanEval": PolyHumanEval, "CodeTransOcean": CodeTransOcean}
        except Exception as e:
            print(f"[CodeTranslationRunner] Warning: Could not import datasets: {e}")
            return {}

    def create_work_items(self, dataset: List[Any], templates: List[Any], models: List[Any]) -> List[Tuple]:
        work_items: List[Tuple] = []
        for data in dataset:
            for src, tgt in data.generate_language_combinations():
                if (src.lower() not in self.source_languages) or (tgt.lower() not in self.target_languages):
                    continue
                for model in models:
                    for template in templates:
                        work_items.append((data, src, tgt, model, template))
        return work_items

    def process_work_item(self, work_item: Tuple) -> Dict[str, Any]:
        data, source_language, target_language, model, template = work_item
        model_name = getattr(model, "model_name", "unknown_model")

        try:
            ref_key = self.record_manager.compose_ref_key_from_template(
                dataset=data.dataset, lang=target_language, template=template, model_name=model_name
            )
            if ref_key is None:
                return {"status": "skipped", "reason": "no_prompt_id"}

            if self.record_manager.is_ref_processed(ref_key):
                return {"status": "skipped", "reason": "already_processed"}

            # Build prompt
            message, wrapped_text = self._build_prompt(data, source_language, target_language, template, model)

            # Fetch responses
            response_list, extracted_output_list = self.fetch_responses_parallel(
                model, message, target_language, self.config.n_requests
            )

            categories = _normalize_categories(getattr(template, "category", []))
            prompt_id = getattr(template, "template_id", None)

            result = {
                "task": "code translation",
                "ref_key": ref_key,
                "source_lang": source_language,
                "target_lang": target_language,
                "dataset": data.dataset,
                "domain": data.domain,
                "id": data.id,
                "prompting_category": categories,
                "prompt_id": prompt_id,
                "prompt_template": template.template_string,
                "wrapped_text": wrapped_text,
                "model_name": model_name,
                "output": response_list,
                "extracted_output": extracted_output_list,
                "driver_code": data.language_driver_code_dict.get(target_language, None),
                "script_name": "Main" if target_language.lower() == "java" else "NOT JAVA",
                "metrics": {},
            }

            self.record_manager.save_record(model_name, result)
            return {"status": "completed"}

        except Exception as e:
            traceback.print_exc()
            return {"status": "error", "error": str(e)}

    def _build_prompt(
        self,
        data: Any,
        source_language: str,
        target_language: str,
        template: Any,
        model: Any
    ) -> Tuple[Any, str]:
        source_code = data.language_code_dict.get(source_language, "")
        func_signatures = data.language_func_sign_dict.get(target_language, None)
        func_info = f" with function signature: `{func_signatures[0]}`" if func_signatures else ""
        template_info = {"SL": source_language, "TL": target_language, "SC": source_code, "FUNC_INFO": func_info}
        return template.get_prompt(model=model, is_chat_mode=False, **template_info)


# =========================
# Convenience factories / runners
# =========================

def create_code_generation_config(
    models: List[str],
    datasets: Dict[str, str],
    languages: List[str] = None,
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        datasets=datasets,
        languages=languages or ["python", "java"],
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )


def create_code_translation_config(
    models: List[str],
    datasets: Dict[str, str],
    template_categories: List[str] = None,
    test_samples: int = None,
    n_requests: int = 1,
    parallel_requests: int = None,
    save_dir: str = "save",
    *,
    sampling_mode: Optional[str] = None,
    sample_size: Optional[int] = None,
    sampling_seed: Optional[int] = 42,
    sampling_manifest_path: Optional[str] = None,
) -> TaskConfig:
    return TaskConfig(
        models=models,
        datasets=datasets,
        languages=[],  # Not used for translation
        template_categories=template_categories or ["direct"],
        test_samples=test_samples,
        n_requests=n_requests,
        parallel_requests=parallel_requests,
        save_dir=save_dir,
        sampling_mode=sampling_mode,
        sample_size=sample_size,
        sampling_seed=sampling_seed,
        sampling_manifest_path=sampling_manifest_path,
    )


def run_code_generation_task(
    models: List[str],
    datasets: Dict[str, str],
    languages: List[str] = None,
    problem_ids: Optional[Iterable[int]] = None,
    **kwargs,
):
    config = create_code_generation_config(models, datasets, languages=languages or ["python", "java"], **kwargs)
    runner = CodeGenerationRunner(config, problem_ids=problem_ids)
    runner.execute_task()


def run_code_translation_task(
    models: List[str],
    datasets: Dict[str, str],
    source_languages: List[str],
    target_languages: List[str],
    **kwargs,
):
    config = create_code_translation_config(models, datasets, **kwargs)
    runner = CodeTranslationRunner(config, source_languages, target_languages)
    runner.execute_task()


# =========================
# Optional smoke test
# =========================
if __name__ == "__main__":
    # Example usage:

    # RANDOM SAMPLE + MANIFEST WRITE:
    # run_code_generation_task(
    #     models=["deepseek-v3"],
    #     datasets={"Geeksforgeeks": "array", "Hackerrank": "strings"},
    #     languages=["python", "java"],
    #     sampling_mode="random",
    #     sample_size=25,
    #     sampling_seed=123,
    #     sampling_manifest_path="save/cg_manifest.json",
    #     n_requests=1,
    #     parallel_requests=8,
    # )

    # REPLAY FROM MANIFEST:
    # run_code_generation_task(
    #     models=["deepseek-v3"],
    #     datasets={"Geeksforgeeks": "array", "Hackerrank": "strings"},
    #     languages=["python", "java"],
    #     sampling_mode="manifest",
    #     sampling_manifest_path="save/cg_manifest.json",
    #     n_requests=1,
    #     parallel_requests=8,
    # )

    print("task_runners.py ready (per-dataset logs + flexible ref_key + sampling manifests).")
