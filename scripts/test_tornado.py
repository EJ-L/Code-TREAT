#!/usr/bin/env python3
"""Evaluate only the Tornado subset of a Symprompt predictions file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluators.symprompt_evaluator import evaluate_symprompt_file, compute_symprompt_metrics


def build_subset(source: Path, output: Path) -> int:
    """Write the subset of `source` entries whose ref_key targets Tornado."""
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)

    with source.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for raw_line in src:
            if not raw_line.strip():
                continue
            data = json.loads(raw_line)
            ref_key = data.get("ref_key", "")
            if "_tqdm_" in ref_key:
                dst.write(raw_line)
                count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Path to the full Symprompt predictions JSONL (e.g. GPT-5.jsonl).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the filtered Tornado JSONL. Defaults to <source>_tornado.jsonl",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Optional worker override passed to evaluate_symprompt_file.",
    )
    args = parser.parse_args()

    source = args.source.resolve()
    if not source.exists():
        raise SystemExit(f"Source file not found: {source}")

    output = args.output.resolve() if args.output else source.with_name(source.stem + "_tornado.jsonl")

    print(f"Building Tornado subset from {source}")
    tornado_count = build_subset(source, output)
    if tornado_count == 0:
        raise SystemExit("No Tornado entries found in the source file.")

    print(f"✓ Wrote {tornado_count} Tornado items to {output}")

    evaluation_path = evaluate_symprompt_file(str(output), num_workers=args.workers)
    metrics = compute_symprompt_metrics(evaluation_path)

    print("\nTornado-only metrics:")
    print(f"  Items evaluated      : {metrics['total_items']}")
    print(f"  Successful items     : {metrics['successful_items']}")
    print(f"  Success rate         : {metrics['compilation_success_rate']:.2f}%")
    print(f"  Avg line coverage    : {metrics['average_line_coverage']:.2f}%")
    print(f"  Avg branch coverage  : {metrics['average_branch_coverage']:.2f}%")


if __name__ == "__main__":
    main()
