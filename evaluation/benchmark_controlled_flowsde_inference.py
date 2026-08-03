#!/usr/bin/env python3
"""Benchmark JIT-warmed controlled FlowSDE open-loop particle rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bayesinf.applications.optimal_control.train_aloha_flow_sde import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/open_loop_flowsde_inference_times.csv"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--particles", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    model = build_model(
        SimpleNamespace(**config), config["state_dim"], config["control_dim"]
    )
    model = eqx.nn.inference_mode(
        eqx.tree_deserialise_leaves(Path(config["checkpoint"]), model)
    )
    x0 = jnp.zeros((args.batch_size, config["state_dim"]))
    controls = jnp.zeros((args.batch_size, args.horizon, config["control_dim"]))

    @eqx.filter_jit
    def run(key):
        keys = jr.split(key, args.batch_size)
        return jax.vmap(
            lambda x, u, k: model.sample_rollout(
                x, u, k, args.particles, solver_name=config["solver"]
            )
        )(x0, controls, keys)

    for index in range(args.warmups):
        jax.block_until_ready(run(jr.PRNGKey(10_000 + index)))
    timings = []
    for index in range(args.repeats):
        started = time.perf_counter()
        jax.block_until_ready(run(jr.PRNGKey(20_000 + index)))
        timings.append(1e3 * (time.perf_counter() - started))
    median_batch = statistics.median(timings)
    row = {
        "method": "controlled_flow_sde",
        "device": str(jax.devices()[0]),
        "batch_size": args.batch_size,
        "particles_per_case": args.particles,
        "horizon": args.horizon,
        "median_batch_ms": median_batch,
        "median_ms_per_case": median_batch / args.batch_size,
        "median_ms_per_particle_trajectory": (
            median_batch / (args.batch_size * args.particles)
        ),
        "particle_trajectories_per_second": (
            1000 * args.batch_size * args.particles / median_batch
        ),
        "repeats": args.repeats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
