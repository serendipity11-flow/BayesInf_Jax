"""Select reproducible Kitchen BC checkpoints and enforce a PPO launch gate."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def candidate(path: str) -> str:
    match = re.search(r"/step_(\d+)/checkpoint$", path)
    if match:
        return f"step_{int(match.group(1)):06d}"
    if "/best_checkpoint" in path:
        return "best_validation"
    return "final"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--expected-seeds", type=int, default=3)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.checkpoints.open()))
    grouped = defaultdict(list)
    for row in rows:
        if row["method"] not in ("CSI-analytic-SDE", "DDPM"):
            continue
        grouped[(row["method"], candidate(row["checkpoint"]))].append(row)

    report = {"threshold": args.threshold, "methods": {}}
    failed = []
    for method in ("CSI-analytic-SDE", "DDPM"):
        options = []
        for (row_method, name), values in grouped.items():
            if row_method != method:
                continue
            seeds = {int(value["seed"]) for value in values}
            if len(seeds) != args.expected_seeds:
                continue
            success = sum(float(value["success_rate_mean"]) for value in values) / len(values)
            return_mean = sum(float(value["return_mean"]) for value in values) / len(values)
            options.append({
                "candidate": name,
                "success_rate": success,
                "return_mean": return_mean,
                "checkpoints": {
                    value["seed"]: value["checkpoint"] for value in values
                },
            })
        if not options:
            raise SystemExit(f"no complete {method} checkpoint candidates")
        selected = max(options, key=lambda value: (value["success_rate"], value["return_mean"]))
        selected["passed"] = selected["success_rate"] >= args.threshold
        report["methods"][method] = {"selected": selected, "candidates": options}
        if not selected["passed"]:
            failed.append(method)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failed:
        raise SystemExit(
            f"BC gate failed at success >= {args.threshold:.0%}: {', '.join(failed)}"
        )


if __name__ == "__main__":
    main()
