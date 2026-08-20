"""Train and evaluate NSF reconstruction of controlled stochastic pendulum dynamics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from bayesinf.applications.state_estimation import (
    BenchmarkTrainingConfig,
    ControlledStochasticPendulum,
    NSFTransitionAdapter,
    NeuralStochasticFlow,
    count_trainable_parameters,
    ensemble_transition_metrics,
    fit_normalization,
    make_transition_pairs,
    sample_controlled_pendulum_data,
    train_transition_kernel_nll,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--trajectories", type=int, default=320)
    parser.add_argument("--trajectory-steps", type=int, default=60)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--integration-substeps", type=int, default=8)
    parser.add_argument("--train-steps", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--evaluation-trajectories", type=int, default=12)
    parser.add_argument("--ensemble-samples", type=int, default=128)
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--grid-samples", type=int, default=256)
    return parser.parse_args()


def pairs_for(data, indices):
    states, controls, times = (
        data["states"][indices],
        data["controls"][indices],
        data["times"][indices],
    )
    result = make_transition_pairs(states, times, controls)
    result["t_current"] = times[:, :-1].reshape(-1)
    return result


def reference_one_step(process, key, current, control, delta_t, samples):
    return jax.vmap(
        lambda sample_key: process.sample_step(
            sample_key, current, control, delta_t
        )
    )(jax.random.split(key, samples))


def reference_rollouts(process, key, initial, controls, delta_t, samples):
    def one_sample(sample_key):
        keys = jax.random.split(sample_key, initial.shape[0])
        return jax.vmap(process.sample_rollout, in_axes=(0, 0, 0, None))(
            keys, initial, controls, delta_t
        )

    return jax.vmap(one_sample)(jax.random.split(key, samples))


def model_rollouts(kernel, key, initial, controls, delta_t, samples):
    keys = jax.random.split(key, initial.shape[0])
    paths = jax.vmap(
        lambda state, commands, path_key: kernel.sample_rollout(
            path_key,
            state,
            delta_t,
            commands,
            num_samples=samples,
        )
    )(initial, controls, keys)
    return jnp.swapaxes(paths, 0, 1)


def finite_time_drift_grid(kernel, process, key, delta_t, grid_size, samples):
    angles = jnp.linspace(-jnp.pi, jnp.pi, grid_size)
    velocities = jnp.linspace(-4.0, 4.0, grid_size)
    angle_grid, velocity_grid = jnp.meshgrid(angles, velocities, indexing="xy")
    states = jnp.stack((angle_grid.ravel(), velocity_grid.ravel()), axis=-1)
    controls = jnp.zeros((states.shape[0], 1), dtype=states.dtype)
    draws = kernel.sample(
        key, states, delta_t, controls, num_samples=samples
    )
    learned = (jnp.mean(draws, axis=0) - states) / delta_t
    truth = process.drift(states, controls)
    return {
        "angles": angles,
        "velocities": velocities,
        "states": states,
        "learned": learned,
        "truth": truth,
        "rmse": jnp.sqrt(jnp.mean((learned - truth) ** 2)),
        "angle_drift_rmse": jnp.sqrt(jnp.mean((learned[:, 0] - truth[:, 0]) ** 2)),
        "velocity_drift_rmse": jnp.sqrt(jnp.mean((learned[:, 1] - truth[:, 1]) ** 2)),
    }


def plot_trajectory_reconstruction(output, times, truth, controls, paths):
    mean = np.asarray(jnp.mean(paths, axis=0))
    lower = np.asarray(jnp.quantile(paths, 0.05, axis=0))
    upper = np.asarray(jnp.quantile(paths, 0.95, axis=0))
    truth, controls, times = map(np.asarray, (truth, controls, times))
    count = min(3, truth.shape[0])
    figure, axes = plt.subplots(count, 3, figsize=(13, 3.2 * count), squeeze=False)
    labels = ("unwrapped angle [rad]", "angular velocity [rad/s]")
    for trajectory in range(count):
        for state_index in range(2):
            axis = axes[trajectory, state_index]
            axis.plot(times[1:], truth[trajectory, 1:, state_index], "k-", label="truth")
            axis.plot(times[1:], mean[trajectory, :, state_index], color="tab:blue", label="NSF mean")
            axis.fill_between(
                times[1:],
                lower[trajectory, :, state_index],
                upper[trajectory, :, state_index],
                color="tab:blue",
                alpha=0.2,
                label="90% interval",
            )
            axis.set_ylabel(labels[state_index])
            axis.grid(alpha=0.25)
        control_axis = axes[trajectory, 2]
        control_axis.step(times[:-1], controls[trajectory, :, 0], where="post")
        control_axis.set_ylabel("torque [N m]")
        control_axis.grid(alpha=0.25)
    for axis in axes[-1]:
        axis.set_xlabel("time [s]")
    axes[0, 0].legend(loc="best")
    figure.tight_layout()
    figure.savefig(output / "trajectory_reconstruction.png", dpi=180)
    plt.close(figure)


def plot_drift_grid(output, grid):
    size = len(grid["angles"])
    states = np.asarray(grid["states"]).reshape(size, size, 2)
    truth = np.asarray(grid["truth"]).reshape(size, size, 2)
    learned = np.asarray(grid["learned"]).reshape(size, size, 2)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, field, title in zip(
        axes, (truth, learned), ("Reference drift", "NSF finite-time mean drift")
    ):
        axis.quiver(
            states[..., 0], states[..., 1], field[..., 0], field[..., 1],
            angles="xy", scale_units="xy", scale=12.0, width=0.003,
        )
        axis.set_title(title)
        axis.set_xlabel("angle [rad]")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("angular velocity [rad/s]")
    figure.tight_layout()
    figure.savefig(output / "drift_field_comparison.png", dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.trajectories < 10 or args.trajectory_steps < 2:
        raise ValueError("need at least ten trajectories and two transitions")
    args.output.mkdir(parents=True, exist_ok=True)
    process = ControlledStochasticPendulum(
        integration_substeps=args.integration_substeps
    )
    data_key, model_key, evaluation_key = jax.random.split(
        jax.random.key(args.seed), 3
    )
    data = sample_controlled_pendulum_data(
        data_key,
        process,
        args.trajectories,
        args.trajectory_steps,
        delta_t=args.delta_t,
    )
    order = np.asarray(jax.random.permutation(
        jax.random.key(args.seed + 1), args.trajectories
    ))
    train_end, validation_end = int(0.7 * args.trajectories), int(0.85 * args.trajectories)
    train_indices, validation_indices, test_indices = (
        order[:train_end], order[train_end:validation_end], order[validation_end:]
    )
    train, validation, test_pairs = (
        pairs_for(data, train_indices),
        pairs_for(data, validation_indices),
        pairs_for(data, test_indices),
    )
    state_mean, state_std, control_mean, control_std = fit_normalization(
        data["states"][train_indices], data["controls"][train_indices]
    )
    model = NeuralStochasticFlow(
        2,
        1,
        hidden_size=args.hidden_size,
        hidden_depth=args.hidden_depth,
        num_coupling_layers=args.flow_layers,
        state_mean=state_mean,
        state_std=state_std,
        control_mean=control_mean,
        control_std=control_std,
        key=model_key,
    )
    kernel = NSFTransitionAdapter(model)
    training_config = BenchmarkTrainingConfig(
        steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 2,
    )
    started = time.perf_counter()
    kernel, losses = train_transition_kernel_nll(kernel, train, training_config)
    jax.block_until_ready(losses)
    training_seconds = time.perf_counter() - started

    one_key, one_reference_key, path_key, path_reference_key, grid_key, timing_key = (
        jax.random.split(evaluation_key, 6)
    )
    condition_count = min(512, test_pairs["x_current"].shape[0])
    current = test_pairs["x_current"][:condition_count]
    target = test_pairs["x_next"][:condition_count]
    control = test_pairs["control"][:condition_count]
    current_time = test_pairs["t_current"][:condition_count]
    one_samples = kernel.sample(
        one_key,
        current,
        args.delta_t,
        control,
        num_samples=args.ensemble_samples,
        t_current=current_time,
    )
    one_reference = reference_one_step(
        process,
        one_reference_key,
        current,
        control,
        args.delta_t,
        args.ensemble_samples,
    )
    one_metrics = ensemble_transition_metrics(
        one_samples, target, reference_samples=one_reference
    )
    one_metrics["nll"] = -jnp.mean(kernel.log_prob(
        target, current, args.delta_t, control, current_time
    ))

    evaluation_count = min(args.evaluation_trajectories, len(test_indices))
    truth = data["states"][test_indices[:evaluation_count]]
    commands = data["controls"][test_indices[:evaluation_count]]
    initial = truth[:, 0]
    paths = model_rollouts(
        kernel,
        path_key,
        initial,
        commands,
        args.delta_t,
        args.ensemble_samples,
    )
    reference_paths = reference_rollouts(
        process,
        path_reference_key,
        initial,
        commands,
        args.delta_t,
        args.ensemble_samples,
    )
    recursive_metrics = ensemble_transition_metrics(
        paths, truth[:, 1:], reference_samples=reference_paths
    )
    validation_nll = -jnp.mean(kernel.log_prob(
        validation["x_next"],
        validation["x_current"],
        validation["delta_t"],
        validation["control"],
        validation["t_current"],
    ))
    grid = finite_time_drift_grid(
        kernel,
        process,
        grid_key,
        args.delta_t,
        args.grid_size,
        args.grid_samples,
    )

    @eqx.filter_jit
    def timed_rollout(draw_key):
        return model_rollouts(
            kernel,
            draw_key,
            initial,
            commands,
            args.delta_t,
            args.ensemble_samples,
        )

    timed_rollout(timing_key).block_until_ready()
    timing_started = time.perf_counter()
    timed_rollout(jax.random.fold_in(timing_key, 1)).block_until_ready()
    rollout_seconds = time.perf_counter() - timing_started

    eqx.tree_serialise_leaves(args.output / "model.eqx", kernel.model)
    plot_trajectory_reconstruction(
        args.output, data["times"][0], truth, commands, paths
    )
    plot_drift_grid(args.output, grid)
    np.savez_compressed(
        args.output / "reconstruction_arrays.npz",
        truth=np.asarray(truth),
        controls=np.asarray(commands),
        posterior_paths=np.asarray(paths),
        reference_paths=np.asarray(reference_paths),
        drift_states=np.asarray(grid["states"]),
        drift_reference=np.asarray(grid["truth"]),
        drift_nsf=np.asarray(grid["learned"]),
        training_losses=np.asarray(losses),
    )
    report = {
        "redistribution_status": "private-not-approved",
        "case": "controlled_stochastic_pendulum_unwrapped_angle",
        "seed": args.seed,
        "model": {
            "state_dim": 2,
            "control_dim": 1,
            "hidden_size": args.hidden_size,
            "hidden_depth": args.hidden_depth,
            "coupling_layers": args.flow_layers,
            "trainable_parameters": count_trainable_parameters(kernel),
        },
        "training": {
            **vars(training_config),
            "num_train_trajectories": len(train_indices),
            "num_validation_trajectories": len(validation_indices),
            "num_test_trajectories": len(test_indices),
            "first_batch_nll": float(losses[0]),
            "last_batch_nll": float(losses[-1]),
            "validation_nll": float(validation_nll),
            "seconds": training_seconds,
        },
        "one_step": {name: float(value) for name, value in one_metrics.items()},
        "recursive": {
            name: float(value) for name, value in recursive_metrics.items()
        },
        "finite_time_drift": {
            "delta_t": args.delta_t,
            "torque": 0.0,
            "rmse": float(grid["rmse"]),
            "angle_drift_rmse": float(grid["angle_drift_rmse"]),
            "velocity_drift_rmse": float(grid["velocity_drift_rmse"]),
            "interpretation": "conditional mean increment divided by delta_t, not an explicit learned drift",
        },
        "runtime": {
            "backend": jax.default_backend(),
            "device": jax.devices()[0].device_kind,
            "ensemble_samples": args.ensemble_samples,
            "trajectories": evaluation_count,
            "horizon": args.trajectory_steps,
            "seconds_total": rollout_seconds,
            "seconds_per_generated_trajectory": (
                rollout_seconds / (evaluation_count * args.ensemble_samples)
            ),
            "compilation_excluded": True,
        },
        "normalization": {
            "state_mean": np.asarray(state_mean).tolist(),
            "state_std": np.asarray(state_std).tolist(),
            "control_mean": np.asarray(control_mean).tolist(),
            "control_std": np.asarray(control_std).tolist(),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
