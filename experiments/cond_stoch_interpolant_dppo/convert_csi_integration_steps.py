"""Repackage a CSI checkpoint on a smaller fixed integration grid.

BC training does not depend on ``integration_steps`` when path loss is off.
This utility preserves all learned modules exactly and rebuilds only the
analytical SDE grid, allowing compute-matched DPPO fine-tuning.
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import equinox as eqx
import jax
import jax.random as jr

from experiments.cond_stoch_interpolant_dppo.train_jax import (
    create_policy,
    load_policy,
    save_policy,
)


def main(source: Path, output: Path, integration_steps: int) -> None:
    config = json.loads((source / "config.json").read_text())
    if config.get("path_loss_weight", 0.0) != 0.0:
        raise ValueError("conversion is valid only when path_loss_weight is zero")
    source_args = Namespace(**config)
    source_policy = load_policy(source, source_args, jr.PRNGKey(0))

    config["integration_steps"] = integration_steps
    target_args = Namespace(**config)
    target_policy = create_policy(target_args, jr.PRNGKey(0))
    target_policy = eqx.tree_at(
        lambda policy: (policy.encoder, policy.vel_net, policy.score_net),
        target_policy,
        (source_policy.encoder, source_policy.vel_net, source_policy.score_net),
    )

    for name in ("encoder", "vel_net", "score_net"):
        source_leaves = jax.tree_util.tree_leaves(
            eqx.filter(getattr(source_policy, name), eqx.is_array)
        )
        target_leaves = jax.tree_util.tree_leaves(
            eqx.filter(getattr(target_policy, name), eqx.is_array)
        )
        if len(source_leaves) != len(target_leaves) or any(
            not eqx.tree_equal(left, right)
            for left, right in zip(source_leaves, target_leaves)
        ):
            raise RuntimeError(f"learned {name} weights changed during conversion")

    save_policy(output, target_policy, target_args)
    (output / "conversion.json").write_text(json.dumps({
        "source": str(source),
        "source_integration_steps": source_policy.integration_steps,
        "integration_steps": target_policy.integration_steps,
        "learned_weights_preserved": True,
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--integration-steps", type=int, required=True)
    arguments = parser.parse_args()
    main(arguments.source, arguments.output, arguments.integration_steps)
