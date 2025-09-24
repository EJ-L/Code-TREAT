#!/usr/bin/env python3
"""
Materialize one LLM test per project and run via project venv pytest.

This script:
- Reads a Symprompt parsed JSONL and selects the first entry per project.
- Writes the generated test into test-apps/<project>/tests/generated/test_<slug>.py.
- Runs pytest from the project's venv with isolated settings and exact-file coverage.
- Summarizes pass/fail and key errors per project.

Usage:
  python scripts/materialize_and_run_per_project.py \
    --input results/unit_test_generation/symprompt/parsed/GPT-5.jsonl \
    [--keep] [--timeout 60] [--verbose]

Notes:
- Requires test-apps root path configured by SympromptEvaluator (RA_ARISE_TEST_APPS).
- Assumes per-project venvs under virtual_environments/symprompt/<project>_env.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Tuple

# Ensure repo root on path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluators.symprompt_evaluator import SympromptEvaluator, RA_ARISE_TEST_APPS
from module_path_resolver import ModulePathResolver


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "test"


def enhance_test_with_imports(test_code: str, module_name: str) -> str:
    lines = test_code.split("\n")
    module_base = module_name.split(".")[0]
    # If test already imports from the module, return as-is
    for line in lines:
        s = line.strip()
        if s.startswith("from ") and module_base in s:
            return test_code
        if s.startswith("import ") and module_base in s:
            return test_code
    # Inject 'from module import *' after initial imports/comments/blank
    insert_pos = 0
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith(("import ", "from ")) or s.startswith("#") or s == "":
            insert_pos = i + 1
        else:
            break
    lines.insert(insert_pos, f"from {module_name} import *")
    return "\n".join(lines)


def build_env_for_project(python_exe: Path, coveragerc: Path, repo_path: Path) -> Dict[str, str]:
    env = os.environ.copy()
    venv_bin = python_exe.parent
    venv_dir = venv_bin.parent
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PATH"] = os.pathsep.join([str(venv_bin), env.get("PATH", "")])
    # Isolate and speed up pytest
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    # Coverage
    env["COVERAGE_PROCESS_START"] = str(coveragerc)
    env["COVERAGE_CORE"] = "sysmon"
    # Imports should resolve to test-apps source first
    python_paths = [str(repo_path), str(repo_path.parent), env.get("PYTHONPATH", "")]
    env["PYTHONPATH"] = os.pathsep.join([p for p in python_paths if p])
    # Constrain network
    env["NO_PROXY"] = "*"
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    return env


def create_coveragerc(repo_path: Path, module_file: Path) -> str:
    # Minimal, exact-source coveragerc
    lines = [
        "[run]",
        "branch = True",
        "source =",
        f"    {module_file.parent}",
        f"    {repo_path}",
        f"    {repo_path.resolve()}",
        "",
        "[report]",
        "omit =",
        "    */site-packages/*",
        "    */tests/*",
        "    */__pycache__/*",
        "",
    ]
    return "\n".join(lines)


def run_one(project: str, data: Dict[str, Any], evaluator: SympromptEvaluator, resolver: ModulePathResolver, timeout: int, keep: bool, verbose: bool) -> Tuple[bool, str]:
    ref_key = data.get("ref_key", data.get("key", ""))
    parsed = evaluator._parse_ref_key_for_focal_method(ref_key)
    if not parsed:
        return False, "ref_key_parse_failed"
    project_name, module_name, _cls, _meth, method_lines = parsed
    if project_name != project:
        return False, f"project_mismatch:{project_name}"

    repo_path = RA_ARISE_TEST_APPS / project_name
    if not repo_path.exists():
        return False, "repo_missing"

    python_exe = evaluator.venv_manager.get_python_executable(project_name)
    if not python_exe:
        return False, "venv_missing"

    # Validate module import in project venv
    ok, msg = resolver.validate_module_import(module_name, python_exe)
    if not ok:
        return False, f"module_import_failed: {msg.strip()[-200:]}"

    # Resolve module file for precise coverage targeting
    module_file = resolver.find_module_file(module_name)
    if not module_file or not module_file.exists():
        return False, f"module_file_not_found:{module_name}"

    # Prepare test content
    parsed_responses = data.get("parsed_response", [])
    if not parsed_responses:
        return False, "empty_parsed_response"
    test_code = parsed_responses[0] if isinstance(parsed_responses[0], str) else str(parsed_responses[0])
    if not test_code.strip():
        return False, "empty_parsed_response"
    test_code = enhance_test_with_imports(test_code, module_name)

    # Materialize into tests/generated
    gen_dir = repo_path / "tests" / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    test_name = f"test_{slugify(ref_key)}.py"
    test_path = gen_dir / test_name
    test_path.write_text(test_code, encoding="utf-8")

    # Temporary coveragerc in tempdir
    with tempfile.TemporaryDirectory(prefix="symp_cov_") as td:
        cov_path = Path(td) / ".coveragerc"
        cov_path.write_text(create_coveragerc(repo_path, module_file), encoding="utf-8")

        env = build_env_for_project(python_exe, cov_path, repo_path)

        # Build pytest cmd: exact file and pytest-cov plugin
        cmd = [
            str(python_exe), "-m", "pytest",
            str(test_path),
            "--disable-warnings",
            "--continue-on-collection-errors",
            "--tb=short",
            "-q",
            "-p", "pytest_cov",
            f"--cov={str(module_file)}",
            "--cov-branch",
            "--cov-report=term-missing",
        ]
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
            check=False,
        )

    if not keep:
        try:
            test_path.unlink(missing_ok=True)
        except Exception:
            pass

    if verbose:
        print(f"[{project}] exit={proc.returncode}")
        if proc.stderr:
            print(proc.stderr[-800:])
        if proc.stdout:
            print(proc.stdout[-800:])

    return proc.returncode == 0, ("ok" if proc.returncode == 0 else "pytest_failed")


def main() -> int:
    ap = argparse.ArgumentParser(description="Materialize and run one LLM test per project")
    ap.add_argument("--input", required=True, help="Path to parsed JSONL file")
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--keep", action="store_true", help="Keep generated tests")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    evaluator = SympromptEvaluator()
    resolver = ModulePathResolver(RA_ARISE_TEST_APPS)

    # Select first entry per project
    seen: Dict[str, Dict[str, Any]] = {}
    with input_path.open("r") as f:
        for line in f:
            try:
                data = json.loads(line.strip())
            except Exception:
                continue
            if data.get("dataset") != "symprompt":
                continue
            ref_key = data.get("ref_key", data.get("key", ""))
            parsed = evaluator._parse_ref_key_for_focal_method(ref_key)
            if not parsed:
                continue
            project, _module, _cls, _meth, _lines = parsed
            if project not in seen:
                seen[project] = data

    if not seen:
        print("No symprompt entries found.")
        return 2

    total = len(seen)
    ok = 0
    for project, data in seen.items():
        success, note = run_one(project, data, evaluator, resolver, args.timeout, args.keep, args.verbose)
        print(f"Project {project}: {'PASS' if success else 'FAIL'} ({note})")
        ok += 1 if success else 0

    print(f"\nSummary: {ok}/{total} projects passed")
    return 0 if ok == total else 2


if __name__ == "__main__":
    sys.exit(main())

