"""Aggregate Symprompt evaluation failures for debugging environments.

Reads an evaluated Symprompt JSONL file and groups failure cases by
repository, error message, and stderr signatures to quickly surface
missing dependencies / import issues.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def bucket_errors(evaluated_path: Path) -> None:
    repo_breakdown: dict[str, Counter[str]] = defaultdict(Counter)
    stderr_topline = Counter()

    with evaluated_path.open() as fh:
        for line in fh:
            data = json.loads(line)
            ref_key = data.get("ref_key") or data.get("key", "")
            parts = ref_key.split("_")
            repo = parts[2] if len(parts) >= 3 else "unknown"

            for evaluation in data.get("evaluation", []):
                error = evaluation.get("error")
                stderr = evaluation.get("stderr") or ""
                if not error:
                    continue

                headline = stderr.splitlines()[0] if stderr else ""
                stderr_topline[headline] += 1

                bucket_key = error
                if headline:
                    bucket_key = f"{error} | {headline}"

                repo_breakdown[repo][bucket_key] += 1

    print("Top stderr headlines (all repos):")
    for msg, count in stderr_topline.most_common(20):
        headline = msg or "<empty stderr>"
        print(f"{count:>4}  {headline}")

    print("\nTop issues per repository:")
    for repo, counters in sorted(
        repo_breakdown.items(), key=lambda item: sum(item[1].values()), reverse=True
    ):
        total_errors = sum(counters.values())
        print(f"\n{repo} — {total_errors} failures")
        for bucket, count in counters.most_common(8):
            print(f"  {count:>4}  {bucket}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluated_jsonl",
        type=Path,
        help="Path to the evaluated Symprompt predictions JSONL",
    )
    args = parser.parse_args()

    bucket_errors(args.evaluated_jsonl)


if __name__ == "__main__":
    main()
