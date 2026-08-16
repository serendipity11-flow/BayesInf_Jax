"""Train control-conditioned LatentSSMFlow dynamics for Jackal.

Input NPZ keys:
  states:  (episodes, T+1, 6)
  actions: (episodes, T, 2)

The row ``actions[e,t]`` must be the command applied between
``states[e,t]`` and ``states[e,t+1]``. With no ``--data``, a synthetic
lagged-unicycle dataset is generated as a runnable smoke test.
"""

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from bayesinf.applications.optimal_control.gru_latent_ssm_flow import GRULatentSSMFlow
from .model import (
    generate_control_task_identification_data,
    generate_high_fidelity_identification_data,
    generate_identification_data, transition_windows,
)


def split_episodes(states, actions, seed):
    count = len(states)
    if count < 3:
        raise ValueError("at least three episodes are required for train/val/test")
    order = np.asarray(jr.permutation(jr.PRNGKey(seed), count))
    n_train = max(1, int(0.8 * count))
    n_val = max(1, int(0.1 * count))
    if n_train + n_val >= count:
        n_train = count - 2
    return tuple(
        (states[index], actions[index])
        for index in (order[:n_train], order[n_train:n_train + n_val],
                      order[n_train + n_val:])
    )


def run(args):
    if args.data:
        bundle = np.load(args.data)
        states = np.asarray(bundle["states"], dtype=np.float32)
        actions = np.asarray(bundle["actions"], dtype=np.float32)
    else:
        if args.plant == "control_tasks":
            states, actions = generate_control_task_identification_data(
                jr.PRNGKey(args.seed), args.synthetic_episodes,
                args.synthetic_steps,
            )
        elif args.plant in ("high_fidelity", "robust"):
            states, actions = generate_high_fidelity_identification_data(
                jr.PRNGKey(args.seed), args.synthetic_episodes,
                args.synthetic_steps, robust=args.plant == "robust",
            )
        else:
            states, actions = generate_identification_data(
                jr.PRNGKey(args.seed), args.synthetic_episodes,
                args.synthetic_steps,
            )
    if states.ndim != 3 or states.shape[-1] != 6:
        raise ValueError("states must have shape (episodes,T+1,6)")
    if actions.ndim != 3 or actions.shape[-1] != 2:
        raise ValueError("actions must have shape (episodes,T,2)")
    if states.shape[:2] != (actions.shape[0], actions.shape[1] + 1):
        raise ValueError("each action u[t] needs both x[t] and x[t+1]")

    (train_s, train_a), (val_s, val_a), (test_s, test_a) = split_episodes(
        states, actions, args.seed
    )
    train_x, train_u = transition_windows(
        train_s, train_a, args.history, args.rollout_steps
    )
    val_x, val_u = transition_windows(
        val_s, val_a, args.history, args.rollout_steps
    )
    test_x, test_u = transition_windows(
        test_s, test_a, args.history, args.rollout_steps
    )

    model = GRULatentSSMFlow(
        6, 2, args.history, args.hidden_dim, jr.PRNGKey(args.seed + 1),
        predict_delta=True, n_coupling=args.blocks, hidden_size=args.width,
        n_hidden=args.hidden_layers, max_rho=args.max_rho,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0), optax.adamw(args.lr, weight_decay=1e-6)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    # Position, heading embedding, and velocity live on different natural
    # scales.  Scaling the squared rollout residual keeps the recursive loss
    # from being dominated by whichever channel has the largest variance.
    rollout_scale = jnp.asarray(args.rollout_scale, dtype=jnp.float32)

    def rollout_mse(candidate, x, u):
        def example(states, controls):
            predicted = candidate.rollout_deterministic_history(
                states[:args.history], controls[:args.history - 1],
                controls[args.history - 1:],
            )
            target = states[args.history:]
            residual = (predicted - target) / rollout_scale
            # Later predictions are the difficult part of a recursive model;
            # discount <= 1 can be used when early MPC accuracy is preferred.
            weights = args.rollout_discount ** jnp.arange(residual.shape[0])
            per_step = jnp.mean(jnp.square(residual), axis=-1)
            return jnp.sum(weights * per_step) / jnp.sum(weights)
        return jnp.mean(jax.vmap(example)(x, u))

    def objective(candidate, x, u):
        nll = candidate.nll_loss(x, u)
        recursive = rollout_mse(candidate, x, u)
        return nll + args.rollout_loss_weight * recursive, (nll, recursive)

    @eqx.filter_jit
    def update(current_model, current_state, x, u):
        (loss, components), gradients = eqx.filter_value_and_grad(
            objective, has_aux=True
        )(current_model, x, u)
        updates, next_state = optimizer.update(
            gradients, current_state,
            eqx.filter(current_model, eqx.is_inexact_array),
        )
        return (eqx.apply_updates(current_model, updates), next_state, loss,
                components)

    best, best_validation = model, float("inf")
    best_validation_nll = float("inf")
    for epoch in range(args.epochs):
        order = np.asarray(jr.permutation(
            jr.PRNGKey(args.seed + epoch + 10), len(train_x)
        ))
        losses, nlls, recursive_losses = [], [], []
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            model, opt_state, loss, components = update(
                model, opt_state, train_x[batch], train_u[batch]
            )
            losses.append(float(loss))
            nlls.append(float(components[0]))
            recursive_losses.append(float(components[1]))
        validation_nll = float(model.nll_loss(val_x, val_u))
        validation = float(rollout_mse(model, val_x, val_u))
        if validation < best_validation:
            best, best_validation = model, validation
            best_validation_nll = validation_nll
        print(
            f"epoch={epoch + 1:03d} loss={np.mean(losses):.5f} "
            f"train_nll={np.mean(nlls):.5f} "
            f"train_rollout_mse={np.mean(recursive_losses):.5f} "
            f"validation_nll={validation_nll:.5f} "
            f"validation_rollout_mse={validation:.5f}", flush=True
        )

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "jackal_latent_ssm_flow.eqx"
    eqx.tree_serialise_leaves(model_path, best)
    metrics = {
        "history": args.history,
        "rollout_steps": args.rollout_steps,
        "train_windows": len(train_x),
        "validation_nll": best_validation_nll,
        "validation_rollout_mse": best_validation,
        "test_nll": float(best.nll_loss(test_x, test_u)),
        "test_rollout_mse": float(rollout_mse(best, test_x, test_u)),
        "rollout_loss_weight": args.rollout_loss_weight,
        "rollout_discount": args.rollout_discount,
        "rollout_scale": args.rollout_scale,
        "state_order": ["x", "y", "sin_yaw", "cos_yaw", "v", "omega"],
        "action_order": ["v_command", "omega_command"],
        "training_plant": "recorded" if args.data else args.plant,
    }
    (args.output / "jackal_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--data", type=Path)
    result.add_argument("--output", type=Path,
                        default=Path("results/results_optimal_control/jackal"))
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--history", type=int, default=5)
    result.add_argument("--rollout-steps", type=int, default=5)
    result.add_argument("--epochs", type=int, default=30)
    result.add_argument("--batch-size", type=int, default=128)
    result.add_argument("--lr", type=float, default=3e-4)
    result.add_argument("--hidden-dim", type=int, default=64)
    result.add_argument("--width", type=int, default=64)
    result.add_argument("--hidden-layers", type=int, default=2)
    result.add_argument("--blocks", type=int, default=4)
    result.add_argument("--max-rho", type=float, default=0.9)
    result.add_argument("--rollout-loss-weight", type=float, default=10.0)
    result.add_argument("--rollout-discount", type=float, default=0.98)
    result.add_argument(
        "--rollout-scale", type=float, nargs=6,
        default=[1.0, 1.0, 0.5, 0.5, 0.5, 1.0],
        metavar=("X", "Y", "SIN_YAW", "COS_YAW", "V", "OMEGA"),
    )
    result.add_argument("--synthetic-episodes", type=int, default=128)
    result.add_argument("--synthetic-steps", type=int, default=100)
    result.add_argument(
        "--plant",
        choices=("high_fidelity", "robust", "control_tasks", "nominal"),
        default="high_fidelity",
        help="Synthetic data source when --data is not supplied.",
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
