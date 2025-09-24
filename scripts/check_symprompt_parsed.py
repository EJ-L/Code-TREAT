#!/usr/bin/env python3
"""
Check Symprompt Parsed JSONL

Validates that all entries in a Symprompt parsed JSONL are structurally
sound and runnable in the local environment. By default performs a fast
sanity check without executing LLM tests. Optionally runs full evaluation
via SympromptEvaluator for deeper verification.

Checks performed (sanity mode):
- dataset == "symprompt"
- ref_key parsable to focal method using evaluator's dataset
- project venv availability
- test-apps repo path existence
- target module importable in the project's venv
- parsed_response exists and is non-empty

Usage:
  python scripts/check_symprompt_parsed.py \
    --input results/unit_test_generation/symprompt/parsed/GPT-5.jsonl \
    [--run-evaluation] [--limit N] [--verbose]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

# Ensure repository root is on sys.path for local imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluators.symprompt_evaluator import SympromptEvaluator, RA_ARISE_TEST_APPS


def non_empty_test_code(parsed_response: Any) -> bool:
    if parsed_response is None:
        return False
    if isinstance(parsed_response, list) and parsed_response:
        first = parsed_response[0]
        return isinstance(first, str) and first.strip() != ""
    if isinstance(parsed_response, str):
        return parsed_response.strip() != ""
    return False


def sanity_check_entry(evaluator: SympromptEvaluator, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Perform sanity validations on a single JSONL entry."""
    result: Dict[str, Any] = {
        "ok": False,
        "errors": [],
        "project": None,
        "module": None,
    }

    if data.get("dataset") != "symprompt":
        result["ok"] = False
        result["errors"].append("not_symprompt_dataset")
        return False, result

    ref_key = data.get("ref_key", data.get("key", ""))
    if not ref_key:
        result["errors"].append("missing_ref_key")
        return False, result

    # Parse focal method info
    parsed = evaluator._parse_ref_key_for_focal_method(ref_key)
    if not parsed:
        result["errors"].append("ref_key_parse_failed")
        return False, result
    project, module_name, class_name, method_name, method_lines = parsed
    result.update({
        "project": project,
        "module": module_name,
        "class": class_name,
        "method": method_name,
        "lines": method_lines,
    })

    # Venv availability
    if not evaluator.venv_manager.is_environment_available(project):
        result["errors"].append("venv_missing")
        # still continue other checks to collect as many issues as possible

    # Repo path existence
    repo_path = RA_ARISE_TEST_APPS / project
    if not repo_path.exists():
        result["errors"].append("repo_missing")

    # Module import validation in project's Python
    py_exe = evaluator.venv_manager.get_python_executable(project)
    if py_exe is None:
        result["errors"].append("python_exe_missing")
    else:
        ok, msg = evaluator.module_resolver.validate_module_import(module_name, py_exe)
        if not ok:
            result["errors"].append("module_import_failed")
            result["import_error"] = msg[-500:]

    # Parsed response must contain test code
    if not non_empty_test_code(data.get("parsed_response")):
        result["errors"].append("empty_parsed_response")

    ok = len(result["errors"]) == 0
    result["ok"] = ok
    return ok, result


def run_full_evaluation(evaluator: SympromptEvaluator, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Run evaluator to fully execute the test and collect status (slow)."""
    res = evaluator._evaluate_test_case(data)
    details = {
        "test_passed": res.test_passed,
        "compilation_successful": res.compilation_successful,
        "test_executable": res.test_executable,
        "error_category": res.error_category,
        "execution_time": res.execution_time,
    }
    if res.coverage_metrics:
        details.update({
            "line_coverage": res.coverage_metrics.line_coverage_percent,
            "branch_coverage": res.coverage_metrics.branch_coverage_percent,
        })
    return bool(res.compilation_successful), details


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity check a Symprompt parsed JSONL file")
    parser.add_argument("--input", required=True, help="Path to parsed JSONL file")
    parser.add_argument("--run-evaluation", action="store_true", help="Run full evaluation for each entry (slow)")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of entries to process (0 = all)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    evaluator = SympromptEvaluator()

    total = 0
    sanity_ok = 0
    eval_ok = 0
    errors: Dict[str, int] = {}

    for idx, line in enumerate(input_path.open("r"), 1):
        if args.limit and total >= args.limit:
            break
        try:
            data = json.loads(line.strip())
        except Exception as e:
            errors["json_decode_error"] = errors.get("json_decode_error", 0) + 1
            if args.verbose:
                print(f"Line {idx}: JSON decode error: {e}")
            continue

        if data.get("dataset") != "symprompt":
            continue

        total += 1
        ok, info = sanity_check_entry(evaluator, data)
        if ok:
            sanity_ok += 1
        else:
            for err in info["errors"]:
                errors[err] = errors.get(err, 0) + 1
            if args.verbose:
                print(f"Line {idx}: sanity errors: {info['errors']}")

        if args.run_evaluation and ok:
            success, details = run_full_evaluation(evaluator, data)
            if success:
                eval_ok += 1
            else:
                key = f"eval_{details.get('error_category','unknown')}"
                errors[key] = errors.get(key, 0) + 1
                if args.verbose:
                    print(f"Line {idx}: evaluation error: {details}")

    print("\nSummary")
    print(f"  Total symprompt entries: {total}")
    print(f"  Sanity OK: {sanity_ok}/{total}")
    if args.run_evaluation:
        print(f"  Eval OK (compilation_successful): {eval_ok}/{sanity_ok}")
    if errors:
        print("  Errors:")
        for k, v in sorted(errors.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"    {k}: {v}")

    return 0 if (sanity_ok == total and (not args.run_evaluation or eval_ok == sanity_ok)) else 2


if __name__ == "__main__":
    sys.exit(main())
