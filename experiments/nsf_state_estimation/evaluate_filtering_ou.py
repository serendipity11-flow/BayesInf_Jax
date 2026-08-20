"""Evaluate a trained OU NSF in noisy and partial-observation particle filters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesinf.applications.state_estimation import (
    IdentityGaussianObservation,
    LinearGaussianObservation,
    NSFConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    ParticleFilterConfig,
    bootstrap_particle_filter_nsf,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Directory containing model.eqx and run.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trajectories", type=int, default=32)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--delta-t", type=float, default=0.1)
    parser.add_argument("--observation-std", type=float, default=0.3)
    parser.add_argument("--initial-std", type=float, default=0.25)
    parser.add_argument("--particles", type=int, default=1_000)
    parser.add_argument("--resampling-threshold", type=float, default=0.5)
    parser.add_argument(
        "--resampling-method",
        choices=["systematic", "stratified"],
        default="systematic",
    )
    parser.add_argument(
        "--missing-stride",
        type=int,
        default=2,
        help="For one-dimensional OU, retain one partial-scenario observation per stride.",
    )
    return parser.parse_args()


def sample_ou_trajectories(key, process, num_trajectories, num_steps, state_dim, delta_t):
    """Generate states including time zero from the analytic OU transition."""
    if num_trajectories < 1 or num_steps < 2 or delta_t <= 0:
        raise ValueError("need positive trajectories/delta_t and at least two times")
    initial_key, rollout_key = jax.random.split(key)
    process_mean = jnp.broadcast_to(jnp.asarray(process.mean), (state_dim,))
    initial = process_mean + jax.random.normal(
        initial_key, (num_trajectories, state_dim)
    )
    keys = jax.random.split(rollout_key, num_steps - 1)

    def step(current, step_key):
        following = process.sample(step_key, current, delta_t)
        return following, following

    _, following = jax.lax.scan(step, initial, keys)
    states = jnp.concatenate((initial[:, None], following.swapaxes(0, 1)), axis=1)
    times = jnp.arange(num_steps, dtype=states.dtype) * delta_t
    return states, times


def filtering_metrics(result, truth, elapsed_seconds):
    """Aggregate state-estimation metrics without imposing a particle KDE."""
    error = result.mean - truth
    levels = (0.5, 0.8, 0.95)
    coverage = {}
    width = {}
    for level_index, level in enumerate(levels):
        lower = result.credible_intervals[..., level_index, 0, :]
        upper = result.credible_intervals[..., level_index, 1, :]
        coverage[f"coverage_{int(100 * level)}"] = float(
            jnp.mean((truth >= lower) & (truth <= upper))
        )
        width[f"width_{int(100 * level)}"] = float(jnp.mean(upper - lower))
    return {
        "rmse": float(jnp.sqrt(jnp.mean(error**2))),
        "mae": float(jnp.mean(jnp.abs(error))),
        "mean_ess": float(jnp.mean(result.effective_sample_size)),
        "minimum_ess": float(jnp.min(result.effective_sample_size)),
        "resampling_frequency": float(jnp.mean(result.resampled)),
        "seconds_per_update": elapsed_seconds / (truth.shape[0] * truth.shape[1]),
        **coverage,
        **width,
    }


def run_scenario(model, observation_model, observations, truth, times, mask, key, config):
    """Compile once, then time one complete filtering evaluation."""
    initial = truth[:, 0]

    @eqx.filter_jit
    def run(filter_key):
        return bootstrap_particle_filter_nsf(
            observations,
            initial,
            times,
            filter_key,
            model,
            observation_model,
            mask=mask,
            config=config,
        )

    warmup = run(jax.random.fold_in(key, 0))
    warmup.mean.block_until_ready()
    started = time.perf_counter()
    result = run(jax.random.fold_in(key, 1))
    result.mean.block_until_ready()
    elapsed = time.perf_counter() - started
    return result, filtering_metrics(result, truth, elapsed)


def main():
    args = parse_args()
    if args.trajectories < 1 or args.steps < 2 or args.missing_stride < 2:
        raise ValueError("invalid trajectory count, steps, or missing stride")
    metadata = json.loads((args.run / "run.json").read_text())
    model_config = NSFConfig(**metadata["model_config"])
    model = NeuralStochasticFlow.from_config(
        model_config,
        key=jax.random.key(metadata["seed"]),
        state_mean=jnp.asarray(metadata["normalization"]["state_mean"]),
        state_std=jnp.asarray(metadata["normalization"]["state_std"]),
    )
    model = eqx.tree_deserialise_leaves(args.run / "model.eqx", model)
    process = OrnsteinUhlenbeck(**metadata["ou"])
    data_key, full_obs_key, partial_obs_key, filter_key = jax.random.split(
        jax.random.key(args.seed), 4
    )
    truth, times = sample_ou_trajectories(
        data_key,
        process,
        args.trajectories,
        args.steps,
        model.state_dim,
        args.delta_t,
    )
    config = ParticleFilterConfig(
        num_particles=args.particles,
        initial_std=args.initial_std,
        resampling_threshold=args.resampling_threshold,
        resampling_method=args.resampling_method,
        credible_levels=(0.5, 0.8, 0.95),
    )

    full_model = IdentityGaussianObservation(model.state_dim, args.observation_std)
    full_observations = full_model.sample(full_obs_key, truth)
    full_result, full_metrics = run_scenario(
        model,
        full_model,
        full_observations,
        truth,
        times,
        None,
        jax.random.fold_in(filter_key, 0),
        config,
    )

    if model.state_dim > 1:
        observed_coordinates = np.arange(0, model.state_dim, 2)
        matrix = jnp.eye(model.state_dim)[observed_coordinates]
        partial_model = LinearGaussianObservation(matrix, args.observation_std)
        partial_observations = partial_model.sample(partial_obs_key, truth)
        partial_mask = None
    else:
        observed_coordinates = np.array([0])
        partial_model = IdentityGaussianObservation(1, args.observation_std)
        partial_observations = partial_model.sample(partial_obs_key, truth)
        partial_mask = jnp.zeros_like(partial_observations, dtype=bool)
        partial_mask = partial_mask.at[:, :: args.missing_stride, :].set(True)
        partial_mask = partial_mask.at[:, 0, :].set(True)
    partial_result, partial_metrics = run_scenario(
        model,
        partial_model,
        partial_observations,
        truth,
        times,
        partial_mask,
        jax.random.fold_in(filter_key, 1),
        config,
    )

    output = args.output or args.run / "filtering_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "posterior_results.npz",
        truth=np.asarray(truth),
        times=np.asarray(times),
        full_observations=np.asarray(full_observations),
        full_mean=np.asarray(full_result.mean),
        full_covariance=np.asarray(full_result.covariance),
        full_ess=np.asarray(full_result.effective_sample_size),
        partial_observations=np.asarray(partial_observations),
        partial_mask=np.asarray(
            jnp.ones_like(partial_observations, dtype=bool)
            if partial_mask is None else partial_mask
        ),
        partial_mean=np.asarray(partial_result.mean),
        partial_covariance=np.asarray(partial_result.covariance),
        partial_ess=np.asarray(partial_result.effective_sample_size),
    )
    report = {
        "redistribution_status": "private-not-approved",
        "seed": args.seed,
        "num_trajectories": args.trajectories,
        "num_times": args.steps,
        "delta_t": args.delta_t,
        "observation_std": args.observation_std,
        "particle_filter_config": vars(config),
        "partial_observed_coordinates": observed_coordinates.tolist(),
        "one_dimensional_missing_stride": (
            args.missing_stride if model.state_dim == 1 else None
        ),
        "noisy_full": full_metrics,
        "partial": partial_metrics,
        "state_nll": "not reported: no defensible posterior density estimator",
    }
    (output / "filtering_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
