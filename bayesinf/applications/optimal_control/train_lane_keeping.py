"""Train AmortizedSeqFlow for kinematic-bicycle lane keeping.

Example: python -m bayesinf.applications.optimal_control.train_lane_keeping --limit
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from .amortized_seq_flow import AmortizedSeqFlow, control_aware_loss
from .bicycle_dataset import BicycleConfig, generate_dataset, split_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", type=int, default=12000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first-action-weight", type=float, default=10.0)
    p.add_argument("--rollout-weight", type=float, default=1.0)
    p.add_argument("--saturation-factor", type=float, default=1.0,
                   help="Internal tanh bound / physical steering bound")
    p.add_argument("--limit", action="store_true", help="Tiny end-to-end smoke run")
    p.add_argument("--output", type=Path, default=Path("results/optimal_control/lane_flow.eqx"))
    args = p.parse_args()
    if args.limit:
        args.samples, args.epochs = 256, 3

    cfg = BicycleConfig()
    kd, ks, km, kt = jr.split(jr.PRNGKey(args.seed), 4)
    train, val, _ = split_dataset(generate_dataset(kd, args.samples, cfg), ks)
    model = AmortizedSeqFlow(
        cfg.horizon, cfg.delta_max, km,
        saturation_factor=args.saturation_factor,
    )
    optim = optax.adamw(args.learning_rate, weight_decay=1e-5)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(model, state, batch):
        def objective(m):
            return control_aware_loss(
                m,
                batch["actions"], batch["observations"], batch["reference"],
                batch["past_controls"], batch["previous_steering"], batch["errors"],
                dt=cfg.dt, velocity=cfg.velocity, wheelbase=cfg.wheelbase,
                first_action_weight=args.first_action_weight,
                rollout_weight=args.rollout_weight,
            )

        (loss, components), grads = eqx.filter_value_and_grad(
            objective, has_aux=True
        )(model)
        updates, state = optim.update(grads, state, model)
        return eqx.apply_updates(model, updates), state, loss, components

    n = train["actions"].shape[0]
    for epoch in range(args.epochs):
        kt, kp = jr.split(kt)
        perm = jr.permutation(kp, n)
        losses = []
        component_losses = []
        for start in range(0, n, args.batch_size):
            ids = perm[start:start + args.batch_size]
            model, opt_state, loss, components = step(
                model, opt_state, {k: v[ids] for k, v in train.items()}
            )
            losses.append(loss)
            component_losses.append(jnp.stack(components))
        val_loss, val_components = control_aware_loss(
            model,
            val["actions"], val["observations"], val["reference"],
            val["past_controls"], val["previous_steering"], val["errors"],
            dt=cfg.dt, velocity=cfg.velocity, wheelbase=cfg.wheelbase,
            first_action_weight=args.first_action_weight,
            rollout_weight=args.rollout_weight,
        )
        train_components = jnp.mean(jnp.stack(component_losses), axis=0)
        print(
            f"epoch={epoch + 1:04d} train_total={float(jnp.mean(jnp.stack(losses))):.4f} "
            f"val_total={float(val_loss):.4f} "
            f"train_nll={float(train_components[0]):.4f} "
            f"train_first={float(train_components[1]):.6f} "
            f"train_rollout={float(train_components[2]):.4f} "
            f"val_nll={float(val_components[0]):.4f} "
            f"val_first={float(val_components[1]):.6f} "
            f"val_rollout={float(val_components[2]):.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as f:
        eqx.tree_serialise_leaves(f, model)
    metadata = args.output.with_suffix(".txt")
    metadata.write_text("\n".join(f"{k}={v}" for k, v in asdict(cfg).items()) + "\n")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
