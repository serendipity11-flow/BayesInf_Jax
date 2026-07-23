"""Plot seed-aggregated learning curves with 95% bootstrap confidence bands."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def bootstrap(values: np.ndarray, rng: np.random.Generator, n: int = 2000):
    means = values[rng.integers(len(values), size=(n, len(values)))].mean(1)
    return np.quantile(means, [0.025, 0.975])


def plot(metrics: Path, output: Path) -> None:
    df = pd.read_csv(metrics)
    df = df[(df.phase == "eval") & df.subtask.isna()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    rng = np.random.default_rng(0)
    for method, md in df.groupby("method"):
        for ax, col, label in zip(axes, ["episode_return", "success"],
                                  ["Episode return", "Success rate"]):
            xs, ys, los, his = [], [], [], []
            for step, sd in md.groupby("env_steps"):
                per_seed = sd.groupby("seed")[col].mean().dropna().to_numpy()
                if not len(per_seed):
                    continue
                lo, hi = bootstrap(per_seed, rng) if len(per_seed) > 1 else (np.nan, np.nan)
                xs.append(step); ys.append(per_seed.mean()); los.append(lo); his.append(hi)
            order = np.argsort(xs)
            x, y = np.asarray(xs)[order], np.asarray(ys)[order]
            ax.plot(x, y, label=method)
            ax.fill_between(x, np.asarray(los)[order], np.asarray(his)[order], alpha=.2)
            ax.set(xlabel="Environment steps", ylabel=label)
    axes[0].legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("metrics", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    plot(a.metrics, a.output)

