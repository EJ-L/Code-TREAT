#!/usr/bin/env python3
"""
Update Symprompt focal_methods.jsonl with branch information harvested from coverage XML files.

Steps:
1. 为需要的项目先生成 coverage.xml，例如：
       source virtual_environments/symprompt/pymonet_env/bin/activate
       cd /Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/Symprompt/test-apps/pymonet
       PYTHONPATH=$(pwd) coverage run --branch -m pytest -q   # 任选合适命令
       coverage xml -o coverage.xml
2. 运行本脚本：
       python scripts/update_focal_methods_from_coverage.py \
         --input data/symprompt/data/focal_methods.jsonl \
         --output data/symprompt/data/focal_methods_with_branches.jsonl \
         --repos /Users/ericjohnli/Downloads/RA_ARISE/TREAT/tests/unit_test_generation/Symprompt/test-apps
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class CoverageLine:
    number: int            # 1-based
    hits: int
    branches_total: int    # 0 if none reported
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

    has_branches = any(line.branches_total > 0 for line in relevant_lines)
    max_branches = max((line.branches_total for line in relevant_lines), default=0)

    if relevant_lines:
        new_start = min(line.number for line in relevant_lines) - 1
        new_end = max(line.number for line in relevant_lines) - 1
    else:
        new_start, new_end = start0, end0

    record["focal_method_lines"] = [new_start, new_end]
    record["has_branch"] = has_branches
    if max_branches:
        record["branches_total"] = max_branches
    else:
        record.pop("branches_total", None)

    return record, bool(relevant_lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Annotate focal_methods.jsonl with branch information extracted from coverage XML files."
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
        help="Where to write the updated jsonl.",
    )
    parser.add_argument(
        "--repos",
        type=Path,
        required=True,
        help="Root directory containing Symprompt test-app repositories.",
    )
    parser.add_argument(
        "--coverage-name",
        default="coverage.xml",
        help="Name of coverage XML file inside each repo (default: coverage.xml).",
    )

    args = parser.parse_args()

    records = load_focal_methods(args.input)
    grouped: Dict[str, List[Dict]] = {}
    for rec in records:
        grouped.setdefault(rec["project"], []).append(rec)

    updated_records: List[str] = []
    missing_modules: List[Tuple[str, str]] = []
    missing_coverage: List[str] = []

    for project, items in grouped.items():
        repo_path = args.repos / project
        coverage_path = repo_path / args.coverage_name
        if not coverage_path.is_file():
            missing_coverage.append(project)
            continue

        coverage_data = load_coverage_xml(coverage_path)
        for rec in items:
            filename = resolve_filename(rec["module"], coverage_data.keys())
            if not filename:
                missing_modules.append((project, rec["module"]))
                continue
            updated, _ = update_record(rec, coverage_data[filename])
            updated_records.append(json.dumps(updated, ensure_ascii=True))

    if missing_coverage:
        print("Coverage XML missing for projects:", ", ".join(sorted(set(missing_coverage))), file=sys.stderr)
    if missing_modules:
        for proj, mod in sorted(set(missing_modules)):
            print(f"[warning] coverage file for project {proj} does not contain module {mod}", file=sys.stderr)

    args.output.write_text("\n".join(updated_records) + "\n", encoding="utf-8")
    print(f"Wrote {len(updated_records)} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
