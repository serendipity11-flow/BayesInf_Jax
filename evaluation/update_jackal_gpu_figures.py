"""Regenerate the Jackal manuscript PDFs from a completed GPU benchmark."""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from evaluation.plot_jackal_cem_comparison import run as plot_in_domain
from experiments.jackal_obstacle_avoidance.evaluate_ood import plot, plot_open_loop


def main(in_domain: Path, ood: Path, cpu_in_domain: Path,
         cpu_ood: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plot_in_domain(
        in_domain, output / "jackal_cem_comparison_dashboard.pdf",
        cpu_in_domain,
    )

    gpu_summary = json.loads((ood / "jackal_ood_cem_comparison.json").read_text())
    cpu_summary = json.loads(
        (cpu_ood / "jackal_ood_cem_comparison.json").read_text()
    )
    cpu_summary["gpu_reference"] = gpu_summary
    trajectories_npz = np.load(cpu_ood / "jackal_ood_cem_trajectories.npz")
    trajectories = {name: trajectories_npz[name] for name in trajectories_npz.files}
    obstacles = jnp.asarray([[2.0, 0.0, 0.42], [3.0, 0.7, 0.28]])
    goal = jnp.asarray([4.0, 0.0])
    plot(cpu_summary, trajectories, output / "jackal_ood_cem_comparison.pdf", obstacles, goal)
    gpu_summary["cpu_reference"] = cpu_summary
    plot_open_loop(
        gpu_summary, output / "jackal_ood_open_loop_reconstruction.pdf"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-domain", type=Path, required=True)
    parser.add_argument("--ood", type=Path, required=True)
    parser.add_argument("--cpu-in-domain", type=Path, required=True)
    parser.add_argument("--cpu-ood", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation"))
    args = parser.parse_args()
    main(args.in_domain, args.ood, args.cpu_in_domain, args.cpu_ood, args.output)
