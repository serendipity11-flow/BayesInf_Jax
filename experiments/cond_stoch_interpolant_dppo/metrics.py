"""Append-only benchmark logging and aggregation."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev
import time
from typing import Any

import torch


FIELDS = (
    "timestamp", "environment", "method", "seed", "phase", "env_steps",
    "gradient_steps", "episode_return", "success", "subtask", "wall_seconds",
    "update_seconds", "env_step_seconds", "nfe", "latency_ms", "peak_gpu_mb",
)


class MetricLogger:
    def __init__(self, path: str | Path, fixed: dict[str, Any]):
        self.path, self.fixed = Path(path), fixed
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def log(self, **values: Any) -> None:
        row = {k: "" for k in FIELDS}
        row.update(self.fixed)
        row.update(values)
        row["timestamp"] = time.time()
        with self.path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)


def peak_gpu_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    return torch.cuda.max_memory_allocated(device) / 2**20


@torch.inference_mode()
def benchmark_latency(policy, cond, warmup: int = 20, repeats: int = 100) -> tuple[float, float]:
    device = cond["state"].device
    for _ in range(warmup):
        policy(cond, deterministic=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        policy(cond, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples.append(1e3 * (time.perf_counter() - start))
    return mean(samples), stdev(samples)


def aggregate(path: Path, output: Path) -> None:
    import pandas as pd

    df = pd.read_csv(path)
    keys = ["environment", "method", "phase", "env_steps", "subtask"]
    numeric = ["episode_return", "success", "wall_seconds", "update_seconds",
               "env_step_seconds", "nfe", "latency_ms", "peak_gpu_mb"]
    table = df.groupby(keys, dropna=False)[numeric].agg(["mean", "std", "count"])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    aggregate(a.metrics, a.output)

