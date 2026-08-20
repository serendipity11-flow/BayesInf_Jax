"""Matched controlled Duffing benchmark for NSF and NormFlowSDE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesinf.applications.optimal_control.controlled_flow_sde import (
    ControlledRealNVPFlowSDE,
)
from bayesinf.applications.state_estimation import (
    BenchmarkTrainingConfig,
    ControlledDuffingOscillator,
    NSFTransitionAdapter,
    NeuralStochasticFlow,
    NormFlowSDETransitionAdapter,
    count_trainable_parameters,
    ensemble_transition_metrics,
    fit_normalization,
    make_transition_pairs,
    sample_controlled_duffing_data,
    train_transition_kernel_nll,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--trajectories", type=int, default=320)
    parser.add_argument("--trajectory-steps", type=int, default=40)
    parser.add_argument("--delta-t", type=float, default=0.05)
    parser.add_argument("--integration-substeps", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--train-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--flow-layers", type=int, default=4)
    parser.add_argument("--time-embedding", type=int, default=8)
    parser.add_argument("--evaluation-trajectories", type=int, default=16)
    parser.add_argument("--ensemble-samples", type=int, default=64)
    parser.add_argument("--reference-samples", type=int, default=64)
    parser.add_argument("--runtime-samples", type=int, default=64)
    parser.add_argument("--runtime-horizons", default="1,5,10,20,40")
    return parser.parse_args()


def trajectory_pairs(data, indices):
    states = data["states"][indices]
    times = data["times"][indices]
    controls = data["controls"][indices]
    pairs = make_transition_pairs(states, times, controls)
    pairs["t_current"] = times[:, :-1].reshape(-1)
    return pairs


def build_kernels(args, seed, normalization):
    state_mean, state_std, control_mean, control_std = normalization
    nsf = NeuralStochasticFlow(
        2,
        1,
        hidden_size=args.hidden_size,
        hidden_depth=args.hidden_depth,
        num_coupling_layers=args.flow_layers,
        state_mean=state_mean,
        state_std=state_std,
        control_mean=control_mean,
        control_std=control_std,
        key=jax.random.key(seed + 101),
    )
    flow = ControlledRealNVPFlowSDE(
        2,
        1,
        jax.random.key(seed + 102),
        hidden_size=args.hidden_size,
        depth=args.hidden_depth,
        diffusion_hidden=args.hidden_size,
        flow_layers=args.flow_layers,
        time_emb_dim=args.time_embedding,
        dt=args.delta_t,
    )
    return {
        "nsf": NSFTransitionAdapter(nsf),
        "normflowsde": NormFlowSDETransitionAdapter(
            flow,
            state_mean=state_mean,
            state_std=state_std,
            control_mean=control_mean,
            control_std=control_std,
            solver_name="euler",
        ),
    }


def reference_one_step(process, key, x_current, control, delta_t, samples):
    return jax.vmap(
        lambda sample_key: process.sample_step(
            sample_key, x_current, control, delta_t
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
        lambda x0, commands, path_key: kernel.sample_rollout(
            path_key,
            x0,
            delta_t,
            commands,
            num_samples=samples,
        )
    )(initial, controls, keys)
    return jnp.swapaxes(paths, 0, 1)


def numeric_metrics(metrics):
    return {name: float(value) for name, value in metrics.items()}


def evaluate_kernel(kernel, process, test, key, args):
    count = min(args.evaluation_trajectories, test["states"].shape[0])
    truth = test["states"][:count]
    controls = test["controls"][:count]
    initial = truth[:, 0]
    one_current = truth[:, :-1].reshape(-1, 2)
    one_target = truth[:, 1:].reshape(-1, 2)
    one_control = controls.reshape(-1, 1)
    one_time = test["times"][:count, :-1].reshape(-1)
    condition_count = min(128, one_current.shape[0])
    one_current = one_current[:condition_count]
    one_target = one_target[:condition_count]
    one_control = one_control[:condition_count]
    one_time = one_time[:condition_count]
    one_key, reference_key, rollout_key, rollout_reference_key = jax.random.split(
        key, 4
    )
    one_samples = kernel.sample(
        one_key,
        one_current,
        args.delta_t,
        one_control,
        num_samples=args.ensemble_samples,
        t_current=one_time,
    )
    one_reference = reference_one_step(
        process,
        reference_key,
        one_current,
        one_control,
        args.delta_t,
        args.reference_samples,
    )
    one_metrics = numeric_metrics(
        ensemble_transition_metrics(
            one_samples, one_target, reference_samples=one_reference
        )
    )
    one_metrics["nll"] = float(-jnp.mean(kernel.log_prob(
        one_target,
        one_current,
        args.delta_t,
        one_control,
        one_time,
    )))

    paths = model_rollouts(
        kernel,
        rollout_key,
        initial,
        controls,
        args.delta_t,
        args.ensemble_samples,
    )
    reference_paths = reference_rollouts(
        process,
        rollout_reference_key,
        initial,
        controls,
        args.delta_t,
        args.reference_samples,
    )
    recursive_metrics = numeric_metrics(
        ensemble_transition_metrics(
            paths, truth[:, 1:], reference_samples=reference_paths
        )
    )
    return one_metrics, recursive_metrics


def runtime_by_horizon(kernel, initial, controls, horizons, key, args):
    rows = []
    for horizon in horizons:
        selected_controls = controls[:, :horizon]
        selected_initial = initial
        keys = jax.random.split(jax.random.fold_in(key, horizon), initial.shape[0])

        @eqx.filter_jit
        def draw(draw_keys):
            return jax.vmap(
                lambda x0, commands, path_key: kernel.sample_rollout(
                    path_key,
                    x0,
                    args.delta_t,
                    commands,
                    num_samples=args.runtime_samples,
                )
            )(selected_initial, selected_controls, draw_keys)

        warmup = draw(keys)
        warmup.block_until_ready()
        started = time.perf_counter()
        paths = draw(jax.vmap(lambda item: jax.random.fold_in(item, 1))(keys))
        paths.block_until_ready()
        elapsed = time.perf_counter() - started
        rows.append({
            "horizon": horizon,
            "seconds_total": elapsed,
            "seconds_per_generated_trajectory": (
                elapsed / (initial.shape[0] * args.runtime_samples)
            ),
        })
    return rows


def aggregate(rows, model_name, section):
    selected = [row[section] for row in rows if row["model"] == model_name]
    result = {}
    for metric in selected[0]:
        values = [entry[metric] for entry in selected]
        result[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return result


def aggregate_runtime(rows, model_name):
    selected = [row for row in rows if row["model"] == model_name]
    horizons = [entry["horizon"] for entry in selected[0]["runtime"]]
    summary = []
    for horizon in horizons:
        values = [
            next(
                entry["seconds_per_generated_trajectory"]
                for entry in row["runtime"] if entry["horizon"] == horizon
            )
            for row in selected
        ]
        summary.append({
            "horizon": horizon,
            "seconds_per_generated_trajectory_mean": float(np.mean(values)),
            "seconds_per_generated_trajectory_std": (
                float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ),
        })
    return summary


def main():
    args = parse_args()
    if args.seeds < 1 or args.trajectories < 6 or args.trajectory_steps < 1:
        raise ValueError("need positive seeds/steps and at least six trajectories")
    if not 0 < args.train_fraction < 1 or not 0 <= args.validation_fraction < 1:
        raise ValueError("invalid split fractions")
    if args.train_fraction + args.validation_fraction >= 1:
        raise ValueError("train and validation fractions must leave a test split")
    process = ControlledDuffingOscillator(
        integration_substeps=args.integration_substeps
    )
    data = sample_controlled_duffing_data(
        jax.random.key(args.seed),
        process,
        args.trajectories,
        args.trajectory_steps,
        delta_t=args.delta_t,
    )
    order = np.asarray(jax.random.permutation(
        jax.random.key(args.seed + 1), args.trajectories
    ))
    train_end = int(args.train_fraction * args.trajectories)
    validation_end = train_end + int(args.validation_fraction * args.trajectories)
    train_indices = order[:train_end]
    validation_indices = order[train_end:validation_end]
    test_indices = order[validation_end:]
    train = trajectory_pairs(data, train_indices)
    validation = trajectory_pairs(data, validation_indices)
    test = {
        name: values[test_indices]
        for name, values in data.items()
    }
    normalization = fit_normalization(
        data["states"][train_indices], data["controls"][train_indices]
    )
    horizons = sorted({
        value for value in (int(item) for item in args.runtime_horizons.split(","))
        if 0 < value <= args.trajectory_steps
    })
    if not horizons:
        raise ValueError("runtime horizons must include a valid positive horizon")
    training_config = BenchmarkTrainingConfig(
        steps=args.train_steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed + 2,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed_index in range(args.seeds):
        run_seed = args.seed + 1000 * seed_index
        kernels = build_kernels(args, run_seed, normalization)
        for model_index, (model_name, kernel) in enumerate(kernels.items()):
            trained, losses = train_transition_kernel_nll(
                kernel,
                train,
                BenchmarkTrainingConfig(
                    **{**vars(training_config), "seed": run_seed + 3}
                ),
            )
            evaluation_key = jax.random.key(run_seed + 100 + model_index)
            one_step, recursive = evaluate_kernel(
                trained, process, test, evaluation_key, args
            )
            validation_nll = float(-jnp.mean(trained.log_prob(
                validation["x_next"],
                validation["x_current"],
                validation["delta_t"],
                validation["control"],
                validation["t_current"],
            )))
            runtime = runtime_by_horizon(
                trained,
                test["states"][: args.evaluation_trajectories, 0],
                test["controls"][: args.evaluation_trajectories],
                horizons,
                jax.random.fold_in(evaluation_key, 99),
                args,
            )
            parameter_count = count_trainable_parameters(trained)
            rows.append({
                "seed": run_seed,
                "model": model_name,
                "trainable_parameters": parameter_count,
                "first_batch_nll": float(losses[0]),
                "last_batch_nll": float(losses[-1]),
                "validation_nll": validation_nll,
                "one_step": one_step,
                "recursive": recursive,
                "runtime": runtime,
                "direct_long_gap": {
                    "status": "not_applicable",
                    "reason": (
                        "changing piecewise-constant control sequence is not a "
                        "single direct transition condition"
                    ),
                },
            })
            eqx.tree_serialise_leaves(
                args.output / f"seed_{seed_index}_{model_name}.eqx", trained
            )

    summary = {
        model_name: {
            "one_step": aggregate(rows, model_name, "one_step"),
            "recursive": aggregate(rows, model_name, "recursive"),
            "runtime": aggregate_runtime(rows, model_name),
            "trainable_parameters": next(
                row["trainable_parameters"] for row in rows
                if row["model"] == model_name
            ),
            "validation_nll": {
                "mean": float(np.mean([
                    row["validation_nll"] for row in rows
                    if row["model"] == model_name
                ])),
                "std": float(np.std([
                    row["validation_nll"] for row in rows
                    if row["model"] == model_name
                ], ddof=1)) if args.seeds > 1 else 0.0,
            },
        }
        for model_name in ("nsf", "normflowsde")
    }
    report = {
        "redistribution_status": "private-not-approved",
        "benchmark": "controlled_stochastic_duffing",
        "seeds": args.seeds,
        "runtime_environment": {
            "jax_backend": jax.default_backend(),
            "device_kind": jax.devices()[0].device_kind,
            "dtype": str(data["states"].dtype),
        },
        "process": {
            "damping": process.damping,
            "linear_stiffness": process.linear_stiffness,
            "cubic_stiffness": process.cubic_stiffness,
            "control_gain": process.control_gain,
            "position_diffusion": process.position_diffusion,
            "velocity_diffusion": process.velocity_diffusion,
            "integration_substeps": process.integration_substeps,
            "delta_t": args.delta_t,
        },
        "matched_protocol": {
            "same_trajectory_split": True,
            "same_normalization_from_training_only": True,
            "same_optimizer_steps_batches_and_learning_rate": True,
            "same_evaluation_conditions_samples_and_horizons": True,
            "training_config": vars(training_config),
            "ensemble_samples": args.ensemble_samples,
            "reference_samples": args.reference_samples,
            "runtime_samples": args.runtime_samples,
            "runtime_excludes_compilation": True,
        },
        "split_sizes": {
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
        },
        "rows": rows,
        "summary": summary,
    }
    (args.output / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
