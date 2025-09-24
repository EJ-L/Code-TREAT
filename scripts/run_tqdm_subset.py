#!/usr/bin/env python3
"""
Run all tqdm LLM tests (from a JSONL subset) against a specific project venv.

This script:
- Reads a JSONL file (e.g., GPT-5_tqdm_top100.jsonl) containing Symprompt LLM responses.
- Materializes each test as-is (no modification) into _symprompt_tmp_tests/ under the tqdm repo.
- Invokes pytest from the provided project .venv with branch coverage and a neutral cov config (--cov-config=/dev/null).
- Summarizes pass/fail and writes per-case results to stdout.

Usage:
  python scripts/run_tqdm_subset.py \
    --jsonl results/unit_test_generation/symprompt/parsed/GPT-5_tqdm_top100.jsonl \
    --repo /Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/Symprompt/test-apps/tqdm \
    --venv /Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/Symprompt/test-apps/tqdm/.venv/bin/python \
    [--keep]
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run tqdm LLM tests against a specific venv")
    ap.add_argument("--jsonl", required=True, help="Path to tqdm-only JSONL file")
    ap.add_argument("--repo", required=True, help="Path to tqdm repo root")
    ap.add_argument("--venv", required=True, help="Path to python in project .venv")
    ap.add_argument("--keep", action="store_true", help="Keep generated tests")
    args = ap.parse_args()

    jsonl = Path(args.jsonl)
    repo = Path(args.repo)
    py = Path(args.venv)
    if not jsonl.exists():
        print(f"JSONL not found: {jsonl}")
        return 2
    if not repo.exists():
        print(f"Repo not found: {repo}")
        return 2
    if not py.exists():
        print(f"Venv python not found: {py}")
        return 2

    tmp_dir = repo / "_symprompt_tmp_tests"
    tmp_dir.mkdir(exist_ok=True)

    # Collect cases
    cases = []
    with jsonl.open() as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("dataset") != "symprompt":
                continue
            ref = d.get("ref_key", d.get("key", ""))
            parsed = d.get("parsed_response") or []
            if not parsed:
                continue
            code = parsed[0] if isinstance(parsed[0], str) else str(parsed[0])
            if not code.strip():
                continue
            cases.append((ref, code))

    print(f"Found {len(cases)} tqdm cases in {jsonl}")

    ok = 0
    for idx, (ref, code) in enumerate(cases, 1):
        test_path = tmp_dir / f"test_tqdm_llm_{idx:03d}.py"
        test_path.write_text(code, encoding="utf-8")
        print(f"\n[{idx}/{len(cases)}] {ref}")

        # pytest with neutral coverage config
        cmd = [
            str(py), "-m", "pytest", "-q",
            "--cov=tqdm", "--cov-branch",
            "--cov-report=term-missing:skip-covered",
            "--cov-report=xml",
            "--cov-config=/dev/null",
            str(test_path),
        ]
        try:
            proc = subprocess.run(cmd, cwd=repo, text=True, capture_output=True)
            print(proc.stdout)
            if proc.returncode == 0:
                ok += 1
            else:
                # Show last lines of stderr for quick context
                print(proc.stderr.splitlines()[-10:])
        except Exception as e:
            print(f"Runner error: {e}")

        if not args.keep:
            try:
                test_path.unlink(missing_ok=True)
            except Exception:
                pass

    print(f"\nSummary: {ok}/{len(cases)} passed")
    return 0 if ok == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())

