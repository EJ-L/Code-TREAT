#!/usr/bin/env python3
"""Unified evaluation driver for Code-TREAT outputs.

For every enabled task this script:
1. Loads raw predictions from ``results/<task>/<dataset>/predictions``
2. Runs the appropriate extractor to create/update ``parsed`` files
3. Invokes the task-specific evaluator to populate ``evaluations`` artifacts

The workflow mirrors ``scripts/run_experiment.py`` but operates on completed
generation logs.  Use ``--tasks`` to limit execution or ``--force-parse`` to
regenerate parsed files even when they already exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import MethodType
from typing import Dict, Iterable, List, Optional, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.model_list import MODELS

from extractors.llm_extraction_utils.code_generation_extractor import (
    CodeGenerationExtractor,
)
from extractors.llm_extraction_utils.code_review_extractor import CodeReviewExtractor
from extractors.llm_extraction_utils.code_reasoning_extractors import (
    CodeReasoningExtractor,
)
from extractors.llm_extraction_utils.code_summarization_extractor import (
    CodeSummarizationExtractor,
)
from extractors.llm_extraction_utils.code_translation_extractor import (
    CodeTranslationExtractor,
)
from extractors.llm_extraction_utils.symprompt_extractor import SymPromptExtractor

from evaluators.code_generation_evaluator import (
    process_pipeline as evaluate_code_generation_file,
)
from evaluators.code_reasoning_evaluators import CodeReasoningEvaluator
from evaluators.code_review_evaluator import evaluate_code_review_file
from evaluators.code_summarization_evaluator import evaluate_code_summarization_file
from evaluators.code_translation_evaluator import (
    process_pipeline as evaluate_code_translation_file,
)
from evaluators.symprompt_evaluator import evaluate_symprompt_file
from evaluators.vulnerability_detection_evaluator import (
    process_pipeline as evaluate_vulnerability_detection_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run evaluation over generated results."
    )
    parser.add_argument(
        "--config",
        default="configs/configs.yaml",
        help="Path to the experiment configuration used for generation (defaults to configs/configs.yaml).",
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        help="Subset of tasks to evaluate (default: tasks marked enabled in config).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Override worker count for extraction/evaluation where applicable.",
    )
    parser.add_argument(
        "--force-parse",
        action="store_true",
        help="Re-run extractors even if parsed files already exist.",
    )
    parser.add_argument(
        "--judge-model",
        default="gpt-4o-2024-11-20",
        help="LLM judge used for review/summarization evaluation (must exist in MODELS).",
    )
    parser.add_argument(
        "--parsing-model",
        default="gpt-4o-2024-11-20",
        help="LLM used for code reasoning parsing (must exist in MODELS).",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def normalize_name(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum()) if name else ""


class EvaluationRunner:
    def __init__(self, config: Dict, args: argparse.Namespace) -> None:
        self.config = config or {}
        self.args = args
        self.results_root = PROJECT_ROOT / "results"
        self.task_params = {
            entry.get("task"): entry.get("parameters", {})
            for entry in self.config.get("tasks", [])
            if isinstance(entry, dict)
        }
        self.model_lookup = self._build_model_lookup()
        self.judge_model_key = self._resolve_model_key(args.judge_model)
        self.parsing_model_key = self._resolve_model_key(args.parsing_model)
        if args.judge_model and not self.judge_model_key:
            raise ValueError(f"Judge model '{args.judge_model}' not found in MODELS")
        if args.parsing_model and not self.parsing_model_key:
            raise ValueError(
                f"Parsing model '{args.parsing_model}' not found in MODELS"
            )
        self.summary: Dict[str, Dict[str, List[Dict[str, Optional[str]]]]] = {}

    def run(self, requested_tasks: Optional[Sequence[str]]) -> None:
        enabled_from_config = [
            entry.get("task")
            for entry in self.config.get("tasks", [])
            if entry.get("enabled")
        ]

        if requested_tasks:
            tasks = (
                [t for t in requested_tasks if t != "all"]
                if "all" not in requested_tasks
                else [entry.get("task") for entry in self.config.get("tasks", [])]
            )
        else:
            tasks = enabled_from_config

        tasks = [t for t in tasks if t]
        if not tasks:
            print("No tasks selected for evaluation.")
            return

        for task in tasks:
            handler = getattr(self, f"_run_{task}", None)
            if not handler:
                print(f"\n[skip] Unsupported task '{task}'")
                continue
            params = self.task_params.get(task, {})
            datasets = self._extract_dataset_names(params.get("datasets", []), task)
            print(f"\n=== {task} ===")
            if not datasets:
                print("  No datasets discovered; skipping task.")
                continue
            handler(params, datasets)

        self._print_summary()

    # ------------------------------------------------------------------
    # Task handlers

    def _run_code_translation(self, params: Dict, datasets: Sequence[str]) -> None:
        max_responses = self._max_responses(params)
        k_list = self._build_k_list(max_responses)
        num_workers = self._worker_override(params)

        for dataset in datasets:
            dataset_dir = self.results_root / "code_translation" / dataset
            self._run_llm_extraction_and_eval(
                task="code_translation",
                dataset=dataset,
                dataset_dir=dataset_dir,
                extractor_cls=CodeTranslationExtractor,
                max_responses=max_responses,
                extractor_workers=num_workers,
                evaluation_fn=lambda parsed_path: evaluate_code_translation_file(
                    str(parsed_path),
                    k_list=k_list,
                    evaluate_datasets=[dataset],
                    num_workers=num_workers,
                ),
            )

    def _run_code_generation(self, params: Dict, datasets: Sequence[str]) -> None:
        max_responses = self._max_responses(params)
        k_list = self._build_k_list(max_responses)
        num_workers = self._worker_override(params)

        for dataset in datasets:
            dataset_dir = self.results_root / "code_generation" / dataset
            self._run_llm_extraction_and_eval(
                task="code_generation",
                dataset=dataset,
                dataset_dir=dataset_dir,
                extractor_cls=CodeGenerationExtractor,
                max_responses=max_responses,
                extractor_workers=num_workers,
                evaluation_fn=lambda parsed_path: evaluate_code_generation_file(
                    str(parsed_path),
                    k_list=k_list,
                    evaluate_datasets=[dataset],
                    num_workers=num_workers,
                ),
            )

    def _run_unit_test_generation(self, params: Dict, datasets: Sequence[str]) -> None:
        max_responses = self._max_responses(params)
        num_workers = self._worker_override(params)

        for dataset in datasets:
            dataset_dir = self.results_root / "unit_test_generation" / dataset
            self._run_llm_extraction_and_eval(
                task="unit_test_generation",
                dataset=dataset,
                dataset_dir=dataset_dir,
                extractor_cls=SymPromptExtractor,
                max_responses=max_responses,
                extractor_workers=num_workers,
                evaluation_fn=lambda parsed_path: evaluate_symprompt_file(
                    str(parsed_path),
                    num_workers=num_workers,
                ),
            )

    def _run_code_review_generation(
        self, params: Dict, datasets: Sequence[str]
    ) -> None:
        max_responses = self._max_responses(params)
        num_workers = self._worker_override(params)

        for dataset in datasets:
            dataset_dir = self.results_root / "code_review_generation" / dataset
            self._run_simple_extraction_and_eval(
                task="code_review_generation",
                dataset=dataset,
                dataset_dir=dataset_dir,
                extractor_factory=CodeReviewExtractor,
                evaluation_fn=lambda parsed_path: evaluate_code_review_file(
                    str(parsed_path),
                    max_k=max_responses,
                    judge_model=self.judge_model_key,
                    auto_parse=False,
                    num_workers=num_workers,
                ),
            )

    def _run_code_summarization(self, params: Dict, datasets: Sequence[str]) -> None:
        max_responses = self._max_responses(params)
        num_workers = self._worker_override(params)

        for dataset in datasets:
            dataset_dir = self.results_root / "code_summarization" / dataset
            self._run_simple_extraction_and_eval(
                task="code_summarization",
                dataset=dataset,
                dataset_dir=dataset_dir,
                extractor_factory=CodeSummarizationExtractor,
                evaluation_fn=lambda parsed_path: evaluate_code_summarization_file(
                    str(parsed_path),
                    max_k=max_responses,
                    judge_model=self.judge_model_key,
                    auto_parse=False,
                    num_workers=num_workers,
                ),
            )

    def _run_vulnerability_detection(
        self, params: Dict, datasets: Sequence[str]
    ) -> None:
        num_response = self._max_responses(params)
        parsed_subdir = params.get("parsed_subdir", "parsed")

        for dataset in datasets:
            dataset_dir = self.results_root / "vulnerability_detection" / dataset
            predictions = self._list_prediction_files(dataset_dir)
            if not predictions:
                print(f"  - {dataset}: no prediction files found")
                continue

            print(f"  - {dataset}: evaluating {len(predictions)} file(s)")
            evaluate_vulnerability_detection_files(
                [str(p) for p in predictions],
                parsed_subdir_name=parsed_subdir,
                num_response=num_response,
            )
            evaluation_path = dataset_dir / "evaluations" / "vul_detect_score.json"
            for pred in predictions:
                self._record(
                    task="vulnerability_detection",
                    dataset=dataset,
                    model=pred.stem,
                    parsed_path=None,
                    evaluation_path=evaluation_path,
                )

    def _run_input_prediction(self, params: Dict, datasets: Sequence[str]) -> None:
        self._run_code_reasoning_task("input_prediction", params, datasets)

    def _run_output_prediction(self, params: Dict, datasets: Sequence[str]) -> None:
        self._run_code_reasoning_task("output_prediction", params, datasets)

    def _run_code_reasoning_task(
        self, task: str, params: Dict, datasets: Sequence[str]
    ) -> None:
        max_responses = self._max_responses(params)
        num_workers = self._worker_override(params)
        k_list = self._build_k_list(max_responses)

        for dataset in datasets:
            dataset_dir = self.results_root / task / dataset
            predictions = self._list_prediction_files(dataset_dir)
            if not predictions:
                print(f"  - {dataset}: no prediction files found")
                continue

            for prediction_path in predictions:
                print(f"  - {dataset}: {prediction_path.name}")
                extractor = CodeReasoningExtractor(
                    parsing_model=MODELS[self.parsing_model_key],
                    max_responses_to_parse=max_responses,
                    max_workers=self.args.max_workers or num_workers or 4,
                )
                parsed_path = Path(extractor.extract_from_file(str(prediction_path)))
                evaluator = CodeReasoningEvaluator(
                    max_workers=self.args.max_workers or num_workers or 4,
                    k_values=k_list,
                    max_responses_to_evaluate=max_responses,
                )
                evaluator.evaluate_file(str(parsed_path))
                evaluation_path = (
                    parsed_path.parent.parent
                    / "evaluations"
                    / (parsed_path.stem + "_evaluation.jsonl")
                )
                self._record(
                    task=task,
                    dataset=dataset,
                    model=prediction_path.stem,
                    parsed_path=parsed_path,
                    evaluation_path=evaluation_path,
                )

    # ------------------------------------------------------------------
    # Helpers

    def _run_llm_extraction_and_eval(
        self,
        task: str,
        dataset: str,
        dataset_dir: Path,
        extractor_cls,
        max_responses: int,
        extractor_workers: Optional[int],
        evaluation_fn,
    ) -> None:
        predictions = self._list_prediction_files(dataset_dir)
        if not predictions:
            print(f"  - {dataset}: no prediction files found")
            return

        for prediction_path in predictions:
            model_name = self._read_model_name(prediction_path) or prediction_path.stem
            model_key = self._resolve_model_key(model_name)
            print(f"  - {dataset}: {prediction_path.name}")
            workers = self.args.max_workers or extractor_workers or 8
            extractor = extractor_cls(model_key, max_workers=workers)
            if model_key is None:
                self._disable_llm_hooks(extractor)

            parsed_path = self._ensure_parsed_with_llm(
                extractor,
                prediction_path,
                max_responses,
            )
            evaluation_fn(parsed_path)
            evaluation_path = (
                parsed_path.parent.parent / "evaluations" / parsed_path.name
            )
            self._record(
                task=task,
                dataset=dataset,
                model=prediction_path.stem,
                parsed_path=parsed_path,
                evaluation_path=evaluation_path,
            )

    def _run_simple_extraction_and_eval(
        self,
        task: str,
        dataset: str,
        dataset_dir: Path,
        extractor_factory,
        evaluation_fn,
    ) -> None:
        predictions = self._list_prediction_files(dataset_dir)
        if not predictions:
            print(f"  - {dataset}: no prediction files found")
            return

        for prediction_path in predictions:
            print(f"  - {dataset}: {prediction_path.name}")
            extractor = extractor_factory()
            parsed_path = self._ensure_parsed_simple(extractor, prediction_path)
            evaluation_fn(parsed_path)
            evaluation_path = (
                parsed_path.parent.parent / "evaluations" / parsed_path.name
            )
            self._record(
                task=task,
                dataset=dataset,
                model=prediction_path.stem,
                parsed_path=parsed_path,
                evaluation_path=evaluation_path,
            )

    def _ensure_parsed_with_llm(
        self,
        extractor,
        prediction_path: Path,
        max_responses: int,
    ) -> Path:
        parsed_dir, _ = extractor._paths_for(str(prediction_path))
        parsed_path = Path(parsed_dir) / prediction_path.name
        if parsed_path.exists() and not self.args.force_parse:
            print(f"    parsed exists → {parsed_path.relative_to(PROJECT_ROOT)}")
            return parsed_path

        extractor.parse_file(
            str(prediction_path), max_responses, resume=not self.args.force_parse
        )
        print(f"    parsed → {parsed_path.relative_to(PROJECT_ROOT)}")
        return parsed_path

    def _ensure_parsed_simple(self, extractor, prediction_path: Path) -> Path:
        parsed_path = prediction_path.parent.parent / "parsed" / prediction_path.name
        if parsed_path.exists() and not self.args.force_parse:
            print(f"    parsed exists → {parsed_path.relative_to(PROJECT_ROOT)}")
            return parsed_path

        extractor.extract(str(prediction_path))
        print(f"    parsed → {parsed_path.relative_to(PROJECT_ROOT)}")
        return parsed_path

    def _record(
        self,
        task: str,
        dataset: str,
        model: str,
        parsed_path: Optional[Path],
        evaluation_path: Optional[Path],
    ) -> None:
        task_bucket = self.summary.setdefault(task, {})
        dataset_bucket = task_bucket.setdefault(dataset, [])
        dataset_bucket.append(
            {
                "model": model,
                "parsed": self._rel(parsed_path),
                "evaluation": self._rel(evaluation_path),
            }
        )

    def _print_summary(self) -> None:
        if not self.summary:
            return
        print("\n=== Summary ===")
        for task, datasets in self.summary.items():
            print(f"{task}:")
            for dataset, entries in datasets.items():
                print(f"  {dataset} ({len(entries)} file(s))")
                for entry in entries:
                    parts = [entry["model"]]
                    if entry["parsed"]:
                        parts.append(f"parsed={entry['parsed']}")
                    if entry["evaluation"]:
                        parts.append(f"eval={entry['evaluation']}")
                    print("    - " + ", ".join(parts))

    # ------------------------------------------------------------------
    # Utility helpers

    def _build_model_lookup(self) -> Dict[str, str]:
        lookup: Dict[str, str] = {}
        for key in MODELS.keys():
            normalized = normalize_name(key)
            lookup[normalized] = key
        return lookup

    def _resolve_model_key(self, raw_name: Optional[str]) -> Optional[str]:
        if not raw_name:
            return None
        normalized = normalize_name(raw_name)
        if normalized in self.model_lookup:
            return self.model_lookup[normalized]

        # Fallback: choose the longest key that overlaps with the request
        best_key = None
        best_len = 0
        for norm_key, key in self.model_lookup.items():
            if norm_key in normalized or normalized in norm_key:
                if len(norm_key) > best_len:
                    best_key = key
                    best_len = len(norm_key)
        return best_key

    def _extract_dataset_names(self, dataset_config: Iterable, task: str) -> List[str]:
        names = set()
        for entry in dataset_config or []:
            if isinstance(entry, dict):
                names.update(entry.keys())
            elif isinstance(entry, str):
                names.add(entry)
        if names:
            return sorted(names)

        task_dir = self.results_root / task
        if not task_dir.exists():
            return []
        return sorted(d.name for d in task_dir.iterdir() if d.is_dir())

    def _read_model_name(self, prediction_path: Path) -> Optional[str]:
        try:
            with open(prediction_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    return data.get("model_name")
        except Exception:
            return None
        return None

    def _list_prediction_files(self, dataset_dir: Path) -> List[Path]:
        predictions_dir = dataset_dir / "predictions"
        if not predictions_dir.exists():
            return []
        return sorted(p for p in predictions_dir.glob("*.jsonl") if p.is_file())

    def _disable_llm_hooks(self, extractor) -> None:
        def _pick_first(self, code_blocks, _target_lang):
            return code_blocks[0] if code_blocks else ""

        def _empty_fallback(self, _response_text, _target_lang):
            return ""

        extractor.select_best_code_block = MethodType(_pick_first, extractor)
        extractor.fallback_extract = MethodType(_empty_fallback, extractor)

    def _max_responses(self, params: Dict) -> int:
        value = params.get("n_requests") or params.get("max_responses") or 1
        try:
            return max(1, int(value))
        except Exception:
            return 1

    def _worker_override(self, params: Dict) -> Optional[int]:
        candidate = (
            self.args.max_workers
            or params.get("parallel_requests")
            or params.get("num_workers")
        )
        if not candidate:
            return None
        try:
            return max(1, int(candidate))
        except Exception:
            return None

    def _build_k_list(self, max_responses: int) -> List[int]:
        base = [1, 5, 10]
        k_list = [k for k in base if k <= max_responses]
        return k_list or [1]

    def _rel(self, path: Optional[Path]) -> Optional[str]:
        if not path:
            return None
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)


def main() -> None:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = load_config(config_path)
    runner = EvaluationRunner(config, args)
    runner.run(args.tasks)


if __name__ == "__main__":
    main()
