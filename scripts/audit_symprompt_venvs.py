"""Audit Symprompt virtual environments for pytest baseline tooling."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMPROMPT_VENVS = ROOT / "virtual_environments" / "symprompt"


def _run(command: str) -> tuple[str, str, int]:
    proc = subprocess.run(
        ["bash", "-lc", command],
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip(), proc.stderr.strip(), proc.returncode


def main() -> None:
    for venv in sorted(SYMPROMPT_VENVS.glob("*_env")):
        repo = venv.name[:-4]
        python = venv / "bin" / "python"
        if not python.exists():
            continue

        print(f"=== {repo} ===")

        proc_version = subprocess.run(
            [
                str(python),
                "-c",
                "import pytest, sys; sys.stdout.write(pytest.__version__)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        version_info = proc_version.stdout.strip() or proc_version.stderr.strip() or f"exit {proc_version.returncode}"
        print(f"pytest version: {version_info}")

        proc = subprocess.run(
            [
                str(python),
                "-c",
                "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pytest_jsonreport.plugin') else 1)",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        has_json = proc.returncode == 0
        print(f"pytest-json-report importable: {'yes' if has_json else 'no'}")

        if not has_json:
            cmd = f'source "{venv}/bin/activate" && pip list --format=columns | head'
            out, err, rc = _run(cmd)
            print("pip list (head):")
            print(out or err)

        print()


if __name__ == "__main__":
    main()
