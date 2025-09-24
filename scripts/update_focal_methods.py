#!/usr/bin/env python3
"""
Fully automatic Symprompt focal-method branch annotator.

For each project:
  1. activate its virtualenv
  2. run coverage (`coverage run --branch -m <module>`) for every module mentioned
     in focal_methods.jsonl (skips ones already processed or explicitly ignored)
  3. parse the resulting coverage XMLs
  4. update focal_methods.jsonl records with:
        - adjusted focal_method_lines (based on any line numbers seen in coverage)
        - has_branch flag
        - branches_total (if coverage reports branch arcs)

Usage:
    python scripts/build_focal_branch_metadata.py \
        --input data/symprompt/data/focal_methods.jsonl \
        --output data/symprompt/data/focal_methods_with_branches.jsonl \
        --repos /Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/Symprompt/test-apps \
        --venvs virtual_environments/symprompt

You may set PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 when running this script if you truly
don’t want pytest plugins to load, but note that many repos (cookiecutter, pytutils, …)
*require* pytest-cov to be available, so leaving auto-load enabled is recommended.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm  # 顶部 import



@dataclass(frozen=True)
class CoverageLine:
    number: int           # 1-based line number
    hits: int
    branches_total: int   # 0 when not a branch line
    branches_covered: int


def load_focal_methods(path: Path) -> List[Dict]:
    records: List[Dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def safe_module_name(module: str) -> str:
    return module.replace(".", "_")


def run_coverage_for_module(
    project: str,
    module: str,
    repo_root: Path,
    venv_root: Path,
    extra_env: Optional[Dict[str, str]] = None,
    xml_dirname: str = "_symprompt_autogen",
) -> Optional[Path]:
    """
    Execute coverage run --branch -m <module> inside project's repo root, writing
    coverage XML into xml_dirname/coverage_<module>.xml.

    Returns path to generated XML or None on failure.
    """
    venv_path = venv_root / f"{project}_env"
    if not venv_path.exists():
        print(f"[warn] venv for project {project} not found at {venv_path}", file=sys.stderr)
        return None

    repo_path = repo_root / project
    if not repo_path.exists():
        print(f"[warn] repo for project {project} not found at {repo_path}", file=sys.stderr)
        return None

    xml_dir = repo_path / xml_dirname
    xml_dir.mkdir(parents=True, exist_ok=True)
    xml_path = xml_dir / f"coverage_{safe_module_name(module)}.xml"
    cov_data_path = xml_dir / f".coverage_{safe_module_name(module)}"

    base_exports = [
        f'export PYTHONPATH="$(pwd)"',
        f'export COVERAGE_FILE="{cov_data_path}"',
    ]
    if extra_env:
        for key, value in extra_env.items():
            base_exports.append(f'export {key}="{value}"')

    module_path = module.replace(".", "/")
    temp_rc_path: Optional[Path] = None
    try:
        temp_rc = tempfile.NamedTemporaryFile("w", delete=False, suffix=".rc")
        with temp_rc as fh:
            fh.write(
                "[run]\nbranch = True\n"
                f"include = */{module_path}.py\n"
                f"    {module_path}.py\n"
            )
        temp_rc_path = Path(temp_rc.name)
    except OSError as exc:
        print(
            f"[warn] failed to create coverage rcfile for {project}:{module}: {exc}",
            file=sys.stderr,
        )

    rc_export = (
        f'export COVERAGE_RCFILE="{temp_rc_path}"' if temp_rc_path is not None else None
    )

    def run_coverage_cmd(inner_cmd: str, *, include_rc: bool = True) -> subprocess.CompletedProcess[str]:
        exports = base_exports.copy()
        if include_rc and rc_export:
            exports.append(rc_export)
        exports_str = " && ".join(exports)
        if exports_str:
            exports_str += " && "
        return subprocess.run(
            f'source "{venv_path}/bin/activate" && '
            f'cd "{repo_path}" && '
            f'{exports_str}{inner_cmd}',
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    run_coverage_cmd("coverage erase")

    run_proc = run_coverage_cmd(f"coverage run --branch -m {module}")
    if run_proc.returncode != 0:
        print(f"[warn] coverage run failed for {project}:{module}", file=sys.stderr)
        print("command:", file=sys.stderr)
        print(run_proc.args if isinstance(run_proc.args, str) else "(internal)", file=sys.stderr)
        print(f"exit code: {run_proc.returncode}", file=sys.stderr)
        print("stdout:", file=sys.stderr)
        print(run_proc.stdout or "(empty)", file=sys.stderr)
        print("stderr:", file=sys.stderr)
        print(run_proc.stderr or "(empty)", file=sys.stderr)
        if xml_path.exists():
            xml_path.unlink()
        if temp_rc_path and temp_rc_path.exists():
            temp_rc_path.unlink()
        return None

    xml_cmd = f'coverage xml -o "{xml_path}"'
    xml_proc = run_coverage_cmd(xml_cmd)
    if xml_proc.returncode != 0 and "No data to report" in (xml_proc.stdout + xml_proc.stderr):
        fallback_proc = run_coverage_cmd(xml_cmd, include_rc=False)
        if fallback_proc.returncode == 0:
            xml_proc = fallback_proc
        else:
            print(
                f"[warn] coverage xml fallback also failed for {project}:{module}",
                file=sys.stderr,
            )
            xml_proc = fallback_proc

    if xml_proc.returncode != 0:
        print(f"[warn] coverage xml failed for {project}:{module}", file=sys.stderr)
        print("command:", file=sys.stderr)
        print(xml_proc.args if isinstance(xml_proc.args, str) else "(internal)", file=sys.stderr)
        print(f"exit code: {xml_proc.returncode}", file=sys.stderr)
        print("stdout:", file=sys.stderr)
        print(xml_proc.stdout or "(empty)", file=sys.stderr)
        print("stderr:", file=sys.stderr)
        print(xml_proc.stderr or "(empty)", file=sys.stderr)
        if xml_path.exists():
            xml_path.unlink()
        if temp_rc_path and temp_rc_path.exists():
            temp_rc_path.unlink()
        return None

    if temp_rc_path and temp_rc_path.exists():
        temp_rc_path.unlink()
    return xml_path


def parse_condition_coverage(value: Optional[str]) -> Tuple[int, int]:
    if not value or "(" not in value or "/" not in value:
        return 0, 0
    try:
        fraction = value.split("(", 1)[1].split(")", 1)[0]
        covered_str, total_str = fraction.split("/")
        covered = int(covered_str.strip())
        total = int(total_str.strip())
    except (ValueError, IndexError):
        return 0, 0
    return covered, total


def load_coverage_xml(xml_path: Path) -> Dict[str, List[CoverageLine]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    data: Dict[str, List[CoverageLine]] = {}
    for class_node in root.iter("class"):
        filename = class_node.get("filename")
        if not filename:
            continue
        lines: List[CoverageLine] = []
        for line_node in class_node.iter("line"):
            try:
                number = int(line_node.get("number", "0"))
                hits = int(line_node.get("hits", "0"))
            except ValueError:
                continue
            branch_attr = line_node.get("branch") == "true"
            branches_covered = branches_total = 0
            if branch_attr:
                branches_covered, branches_total = parse_condition_coverage(
                    line_node.get("condition-coverage")
                )
            lines.append(
                CoverageLine(
                    number=number,
                    hits=hits,
                    branches_total=branches_total,
                    branches_covered=branches_covered,
                )
            )
        if lines:
            data[filename] = lines
    return data


def candidate_filenames(module: str) -> Iterable[str]:
    dotted = module.split(".")
    rel = "/".join(dotted) + ".py"
    yield rel
    if len(dotted) > 1:
        yield "/".join(dotted[-2:]) + ".py"
    yield dotted[-1] + ".py"


def resolve_filename(module: str, coverage_files: Iterable[str]) -> Optional[str]:
    files = list(coverage_files)
    for candidate in candidate_filenames(module):
        matches = [f for f in files if f.endswith(candidate)]
        if len(matches) == 1:
            return matches[0]
        if matches:
            return min(matches, key=len)
    return None


def update_record(
    record: Dict,
    file_lines: List[CoverageLine],
) -> Tuple[Dict, bool]:
    start0, end0 = record["focal_method_lines"]
    start1, end1 = start0 + 1, end0 + 1  # convert to 1-based inclusive bounds

    relevant_lines = [
        line for line in file_lines if start1 <= line.number <= end1
    ]

    has_branch = any(line.branches_total > 0 for line in relevant_lines)
    max_branches = max((line.branches_total for line in relevant_lines), default=0)

    if relevant_lines:
        new_start = min(line.number for line in relevant_lines) - 1
        new_end = max(line.number for line in relevant_lines) - 1
    else:
        new_start, new_end = start0, end0

    record["focal_method_lines"] = [new_start, new_end]
    record["has_branch"] = has_branch
    if max_branches:
        record["branches_total"] = max_branches
    else:
        record.pop("branches_total", None)

    return record, bool(relevant_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automatically regenerate coverage XML per project/module and update Symprompt focal method metadata."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to focal_methods.jsonl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the updated JSONL.",
    )
    parser.add_argument(
        "--repos",
        type=Path,
        required=True,
        help="Root directory containing Symprompt test-app repositories.",
    )
    parser.add_argument(
        "--venvs",
        type=Path,
        required=True,
        help="Root directory containing symprompt virtual environments.",
    )
    parser.add_argument(
        "--xml-dirname",
        default="_symprompt_autogen",
        help="Subdirectory inside each repo to place generated coverage XML files (default: _symprompt_autogen).",
    )
    parser.add_argument(
        "--extra-env",
        default="PYTEST_DISABLE_PLUGIN_AUTOLOAD=0",
        help="Comma-separated KEY=VALUE pairs exported before coverage run (default keeps pytest plugins active).",
    )
    parser.add_argument(
        "--skip-modules",
        nargs="*",
        default=(),
        help="Optional modules to skip (full dotted module names).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, generates coverage XML only when absent and warns instead of failing.",
    )

    args = parser.parse_args()

    extra_env = {}
    if args.extra_env:
        for pair in args.extra_env.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                extra_env[k.strip()] = v.strip()

    records = load_focal_methods(args.input)
    grouped: Dict[str, Dict[str, List[Dict]]] = {}
    for rec in records:
        modules = grouped.setdefault(rec["project"], {})
        modules.setdefault(rec["module"], []).append(rec)

    module_xml_paths: Dict[Tuple[str, str], Optional[Path]] = {}
    for project, modules in tqdm(grouped.items(), desc="Projects"):
        for module in tqdm(modules.keys(), desc=f"{project} modules", leave=False):
            if module in args.skip_modules:
                print(f"[info] skip module {project}:{module}", file=sys.stderr)
                module_xml_paths[(project, module)] = None
                continue

            xml_path = module_xml_paths.get((project, module))
            if xml_path is None:
                xml_dir = args.repos / project / args.xml_dirname
                xml_dir.mkdir(parents=True, exist_ok=True)
                candidate = xml_dir / f"coverage_{safe_module_name(module)}.xml"
                if candidate.exists():
                    xml_path = candidate
                else:
                    xml_path = None

            if xml_path is None or not xml_path.exists():
                xml_path = run_coverage_for_module(
                    project=project,
                    module=module,
                    repo_root=args.repos,
                    venv_root=args.venvs,
                    extra_env=extra_env,
                    xml_dirname=args.xml_dirname,
                )
            module_xml_paths[(project, module)] = xml_path

    retry_attempted: Dict[Tuple[str, str], bool] = {}
    updated_lines: List[str] = []
    for rec in records:
        xml_path = module_xml_paths.get((rec["project"], rec["module"]))
        if not xml_path or not xml_path.exists():
            updated_lines.append(json.dumps(rec, ensure_ascii=True))
            continue

        reran = False
        coverage_data = load_coverage_xml(xml_path)
        filename = resolve_filename(rec["module"], coverage_data.keys())

        # If the existing XML doesn't reference the module, force a single regeneration.
        if not filename and not retry_attempted.get((rec["project"], rec["module"])):
            retry_attempted[(rec["project"], rec["module"])] = True
            fresh_xml_path = run_coverage_for_module(
                project=rec["project"],
                module=rec["module"],
                repo_root=args.repos,
                venv_root=args.venvs,
                extra_env=extra_env,
                xml_dirname=args.xml_dirname,
            )
            if fresh_xml_path and fresh_xml_path.exists():
                module_xml_paths[(rec["project"], rec["module"])] = fresh_xml_path
                xml_path = fresh_xml_path
                coverage_data = load_coverage_xml(xml_path)
                filename = resolve_filename(rec["module"], coverage_data.keys())
                reran = True

        if not filename:
            msg = (
                f"[warn] {rec['project']} coverage does not mention module {rec['module']}"
            )
            if reran:
                msg += " (after regeneration)"
            print(msg, file=sys.stderr)
            updated_lines.append(json.dumps(rec, ensure_ascii=True))
            continue

        updated, _ = update_record(rec, coverage_data[filename])
        updated_lines.append(json.dumps(updated, ensure_ascii=True))

    args.output.write_text("\n".join(updated_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(updated_lines)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
