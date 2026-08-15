"""Two-stage training: drift-only regression, then joint FlowSDE likelihood."""
import argparse, json, time
from pathlib import Path
import equinox as eqx
import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE


def angle_delta(next_state, state, angle_index=2):
    delta = next_state - state
    if angle_index is not None:
        delta[:, angle_index] = np.arctan2(np.sin(delta[:, angle_index]), np.cos(delta[:, angle_index]))
    return delta


def run(args):
    with h5py.File(args.data) as f:
        x, u, xn, split = (f[k][:] for k in ("state", "action", "next_state", "split"))
        dt = float(f.attrs["dt"])
    delta = angle_delta(xn, x)
    train = split == 0; valid = split == 1
    stats = {"state_mean": x[train].mean(0), "state_std": x[train].std(0) + 1e-6,
             "action_mean": u[train].mean(0), "action_std": u[train].std(0) + 1e-6,
             "delta_mean": delta[train].mean(0), "delta_std": delta[train].std(0) + 1e-6}
    xs = (x - stats["state_mean"]) / stats["state_std"]
    us = (u - stats["action_mean"]) / stats["action_std"]
    ds = (delta - stats["delta_mean"]) / stats["delta_std"]
    model = FlowSDE(5, 2, 5, jr.PRNGKey(args.seed), dt=dt, width=args.width,
                    depth=args.depth, condition_dim=args.condition_dim, flow_layers=args.flow_layers)
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            config={**{key: str(value) if isinstance(value, Path) else value
                       for key, value in vars(args).items()}, "dt": dt,
                    "state_dim": 5, "action_dim": 2},
        )
        wandb_run.define_metric("global_epoch")
        wandb_run.define_metric("*", step_metric="global_epoch")

    def drift_loss(candidate, bx, bu, bd):
        prediction = jax.vmap(candidate.mean_delta)(bx, bu)
        return jnp.mean((prediction - bd) ** 2)

    def joint_loss(candidate, bx, bu, bd):
        mse = drift_loss(candidate, bx, bu, bd)
        nll = -jnp.mean(jax.vmap(candidate.log_prob)(bd, bx, bu))
        return nll + args.joint_drift_weight * mse, (mse, nll)

    def make_step(loss, optimizer):
        @eqx.filter_jit
        def step(candidate, opt_state, bx, bu, bd):
            (value, aux), grads = eqx.filter_value_and_grad(loss, has_aux=True)(candidate, bx, bu, bd)
            updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(candidate, eqx.is_inexact_array))
            return eqx.apply_updates(candidate, updates), opt_state, value, aux
        return step

    rng = np.random.default_rng(args.seed); history = []; args.output.mkdir(parents=True, exist_ok=True)
    train_rows, valid_rows = np.flatnonzero(train), np.flatnonzero(valid)
    started = time.perf_counter()
    stages = (("drift", args.drift_epochs, args.drift_lr), ("joint", args.joint_epochs, args.joint_lr))
    for stage, epochs, learning_rate in stages:
        loss = ((lambda m, bx, bu, bd: (drift_loss(m, bx, bu, bd),
                 (drift_loss(m, bx, bu, bd), jnp.asarray(jnp.nan)))) if stage == "drift" else joint_loss)
        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate, weight_decay=1e-5))
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array)); update = make_step(loss, optimizer)
        for epoch in range(1, epochs + 1):
            chosen = rng.permutation(train_rows)[:min(args.samples_per_epoch, len(train_rows))]
            values = []
            for offset in range(0, len(chosen), args.batch_size):
                rows = chosen[offset:offset + args.batch_size]
                model, opt_state, value, aux = update(model, opt_state, jnp.asarray(xs[rows]), jnp.asarray(us[rows]), jnp.asarray(ds[rows]))
                values.append((float(value), float(aux[0]), float(aux[1])))
            vr = valid_rows[:args.validation_samples]
            validation, vaux = loss(model, jnp.asarray(xs[vr]), jnp.asarray(us[vr]), jnp.asarray(ds[vr]))
            batch_metrics = np.asarray(values)
            mean = np.asarray((batch_metrics[:, 0].mean(), batch_metrics[:, 1].mean(),
                               batch_metrics[:, 2].mean() if stage == "joint" else np.nan))
            row = {"stage": stage, "stage_epoch": epoch, "train_loss": float(mean[0]),
                   "train_drift_mse": float(mean[1]), "train_nll": float(mean[2]),
                   "validation_loss": float(validation), "validation_drift_mse": float(vaux[0]),
                   "validation_nll": float(vaux[1])}
            history.append(row)
            if wandb_run is not None:
                metrics = {"global_epoch": len(history), "stage_epoch": epoch,
                           "stage": stage, "optimizer/learning_rate": learning_rate,
                           "loss/train_total": row["train_loss"],
                           "loss/train_drift_mse": row["train_drift_mse"],
                           "loss/validation_total": row["validation_loss"],
                           "loss/validation_drift_mse": row["validation_drift_mse"]}
                if stage == "joint":
                    metrics.update({"loss/train_nll": row["train_nll"],
                                    "loss/validation_nll": row["validation_nll"]})
                wandb_run.log(metrics)
            if epoch == 1 or epoch % args.log_every == 0: print(json.dumps(row), flush=True)
        stage_dir = args.output / f"{stage}_stage"; stage_dir.mkdir(exist_ok=True)
        eqx.tree_serialise_leaves(stage_dir / "flowsde.eqx", model)
        np.savez(stage_dir / "normalization.npz", **stats)
        (stage_dir / "config.json").write_text(json.dumps({**vars(args), "data": str(args.data),
            "output": str(stage_dir), "dt": dt, "elapsed_seconds": time.perf_counter() - started}, default=str, indent=2) + "\n")
        (stage_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    if wandb_run is not None:
        wandb_run.summary["elapsed_seconds"] = time.perf_counter() - started
        wandb_run.summary["final_validation_loss"] = history[-1]["validation_loss"]
        wandb_run.finish()
    return model, history


def parser():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, default=Path("data/differential_drive/data.h5"))
    p.add_argument("--output", type=Path, default=Path("results/results_optimal_control/differential_drive/seed0"))
    p.add_argument("--drift-epochs", type=int, default=100); p.add_argument("--joint-epochs", type=int, default=100)
    p.add_argument("--drift-lr", type=float, default=5e-4); p.add_argument("--joint-lr", type=float, default=1e-4)
    p.add_argument("--joint-drift-weight", type=float, default=5.0); p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--samples-per-epoch", type=int, default=50000); p.add_argument("--validation-samples", type=int, default=10000)
    p.add_argument("--width", type=int, default=128); p.add_argument("--depth", type=int, default=3)
    p.add_argument("--condition-dim", type=int, default=64); p.add_argument("--flow-layers", type=int, default=6)
    p.add_argument("--log-every", type=int, default=10); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb-project", default="DifferentialDrive-FlowSDE")
    p.add_argument("--wandb-entity", default=""); p.add_argument("--wandb-run-name", default="")
    return p


if __name__ == "__main__": run(parser().parse_args())
