"""Benchmark state-estimation inference kernels without counting JAX compilation.

Examples
--------
Classical baselines::

    python -m evaluation.benchmark_state_estimation_inference --methods ekf pf

Learned estimators (the template options must match training)::

    python -m evaluation.benchmark_state_estimation_inference \
      --methods amortized latent flow_sde interpolant \
      --amortized-checkpoint results/results_state_estimation/AmortizedStateFlow_realnvp_lorenz96_N12_seg5_obs2_noise_0.5.pt \
      --latent-checkpoint results/results_state_estimation/LatentSSMFlow_realnvp_fixedSSM_encgru_lwp_ent0.01_lorenz96_N12_seg5_obs2_studt5.0_std0.5.pt \
      --flow-sde-checkpoint CHECKPOINT.pt --interpolant-checkpoint CHECKPOINT.pt \
      --latent-encoder gru --latent-local-window --output inference_times.csv

Inputs are deterministic synthetic arrays because data generation is outside the
timed region. Reported latency is for the whole batch and per trajectory. Each
kernel is JIT-warmed and synchronized with ``jax.block_until_ready``.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import platform
import re
import statistics
import subprocess
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bayesinf.applications.state_estimation.amortized_state_flow import AmortizedStateFlow
from bayesinf.applications.state_estimation.filtering_baselines import (
    bootstrap_particle_filter,
    extended_kalman_filter,
    unscented_kalman_filter,
)
from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE
from bayesinf.applications.state_estimation.latent_ssm_flow import LatentSSMFlow
from bayesinf.applications.state_estimation.stochastic_interpolant import StochasticInterpolantSE


ALL_METHODS = ("ekf", "ukf", "pf", "amortized", "latent", "flow_sde", "interpolant")


class CalibratedStochasticInterpolantSE(StochasticInterpolantSE):
    """Checkpoint-compatible shell used by ``train_si_lorenz96``."""
    obs_loss_weight: float = eqx.field(static=True)
    unobs_loss_weight: float = eqx.field(static=True)
    obs_source_noise_std: float = eqx.field(static=True)
    unobs_source_scale: float = eqx.field(static=True)
    obs_g0_scale: float = eqx.field(static=True)
    unobs_g0_scale: float = eqx.field(static=True)

    def __init__(self, *args, obs_loss_weight=1.0, unobs_loss_weight=1.0,
                 obs_source_noise_std=0.0, unobs_source_scale=1.0,
                 obs_g0_scale=1.0, unobs_g0_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_loss_weight = obs_loss_weight
        self.unobs_loss_weight = unobs_loss_weight
        self.obs_source_noise_std = obs_source_noise_std
        self.unobs_source_scale = unobs_source_scale
        self.obs_g0_scale = obs_g0_scale
        self.unobs_g0_scale = unobs_g0_scale


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=["ekf", "pf"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--horizon", type=int,
                   help="Defaults to a sole learned checkpoint's _hN tag, otherwise 100")
    p.add_argument("--state-dim", type=int, default=12)
    p.add_argument("--obs-interval", type=int, default=2)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--warmups", type=int, default=1)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--particles", type=int, default=100,
                   help="Posterior samples materialized by every method, including EKF")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path,
                   default=Path("evaluation/state_estimation_inference_times_detailed.csv"))
    p.add_argument("--overwrite", action="store_true",
                   help="Replace --output instead of appending rows to a compatible CSV")

    p.add_argument("--amortized-checkpoint", type=Path)
    p.add_argument("--amortized-flow", choices=("realnvp", "fm", "sbm"), default="realnvp")
    p.add_argument("--amortized-encoder", choices=("transformer", "gru"), default="transformer")
    p.add_argument("--amortized-segment-length", type=int, default=5)
    p.add_argument("--fm-steps", type=int, default=100)

    p.add_argument("--latent-checkpoint", type=Path)
    p.add_argument("--latent-flow", choices=("realnvp", "fm"), default="realnvp")
    p.add_argument("--latent-encoder", choices=("transformer", "gru", "pointwise"), default="transformer")
    p.add_argument("--latent-local-window", action="store_true")
    p.add_argument("--latent-local-window-prev-only", action="store_true")
    p.add_argument("--latent-segment-length", type=int, default=5)
    p.add_argument("--latent-max-rho", type=float, default=0.8)

    p.add_argument("--flow-sde-checkpoint", type=Path)
    p.add_argument("--flow-sde-solver", default="heun")
    p.add_argument("--observation-stride", type=int, default=5)
    p.add_argument("--flow-sde-d-c", type=int, help="Inferred from checkpoint when omitted")
    p.add_argument("--flow-sde-d-hidden-enc", type=int, help="Inferred from checkpoint when omitted")
    p.add_argument("--flow-sde-d-hidden", type=int, help="Inferred from checkpoint when omitted")
    p.add_argument("--flow-sde-d-hidden-diffusion", type=int, help="Inferred from checkpoint when omitted")

    p.add_argument("--interpolant-checkpoint", type=Path)
    p.add_argument("--interpolant-steps", type=int, default=20)
    p.add_argument("--interpolant-sampler", choices=("sde", "ode", "segmented"), default="segmented")
    p.add_argument("--interpolant-segment-length", type=int,
                   help="Defaults to the checkpoint's _msN suffix, otherwise 20")
    p.add_argument("--interpolant-d-c", type=int, default=128)
    p.add_argument("--interpolant-d-hidden", type=int, default=256)
    p.add_argument("--interpolant-depth", type=int, default=4)
    return p.parse_args()


def require_checkpoint(path, method):
    if path is None:
        raise SystemExit(f"--{method.replace('_', '-')}-checkpoint is required for {method}")
    if not path.is_file():
        raise SystemExit(f"Checkpoint does not exist: {path}")
    return path


def load_compatible_array_leaves(path, model):
    """Load newer SI checkpoints while tolerating old Equinox metadata leaves."""
    def spec(f, x):
        is_jax_array = isinstance(x, (jax.Array, jax.ShapeDtypeStruct))
        if is_jax_array or isinstance(x, np.ndarray):
            for _ in range(8):
                value = np.load(f, allow_pickle=True)
                if np.shape(value) == np.shape(x):
                    return jnp.asarray(value) if is_jax_array else value
                if np.size(value) != 1:
                    return jnp.asarray(value) if is_jax_array else value
            return jnp.asarray(value) if is_jax_array else value
        if isinstance(x, (bool, int, float, complex, np.generic)):
            position = f.tell()
            value = np.load(f, allow_pickle=True)
            if np.size(value) == 1:
                return type(x)(value.item())
            f.seek(position)
        return x
    return eqx.tree_deserialise_leaves(path, model, filter_spec=spec)


def infer_flow_sde_dimensions(path, d_x, d_y, time_emb_dim=32):
    """Infer dimensions from the stable ordering of FlowSDE array leaves."""
    arrays = []
    with path.open("rb") as f:
        while len(arrays) < 32:
            try:
                arrays.append(np.load(f, allow_pickle=True))
            except (EOFError, OSError, ValueError, pickle.UnpicklingError):
                break
    if len(arrays) < 27 or arrays[0].ndim != 2 or arrays[0].shape[1] != d_y:
        raise SystemExit("Could not infer FlowSDE architecture; provide the --flow-sde-d-* options")
    d_hidden_enc = arrays[0].shape[0] // 3
    d_c = arrays[8].shape[0]
    # First drift matrix has input width d_x + d_c + time embedding.
    drift_index = next(
        (i for i, value in enumerate(arrays)
         if value.ndim == 2 and value.shape[1] == d_x + d_c + time_emb_dim),
        None,
    )
    if drift_index is None:
        raise SystemExit("Could not identify the FlowSDE drift network in the checkpoint")
    d_hidden = arrays[drift_index].shape[0]
    drift_output_index = next(
        (i for i in range(drift_index + 1, len(arrays))
         if arrays[i].ndim == 2 and arrays[i].shape[0] == d_x),
        None,
    )
    diffusion_index = None if drift_output_index is None else next(
        (i for i in range(drift_output_index + 1, len(arrays)) if arrays[i].ndim == 2),
        None,
    )
    d_hidden_diffusion = arrays[diffusion_index].shape[0] if diffusion_index is not None else 64
    return d_c, d_hidden_enc, d_hidden, d_hidden_diffusion


def load_interpolant_class(path):
    """Select the q_eta or historical lifter architecture from checkpoint leaves."""
    arrays = []
    with path.open("rb") as f:
        while len(arrays) <= 67:
            try:
                arrays.append(np.load(f, allow_pickle=True))
            except (EOFError, OSError, ValueError, pickle.UnpicklingError):
                break
    # q_eta has mu and log-sigma heads at leaves 64--67. The legacy lifter has
    # only one 12-D output and its velocity input projection begins at leaf 66.
    legacy = len(arrays) > 66 and np.shape(arrays[66]) != np.shape(arrays[64])
    if not legacy:
        return CalibratedStochasticInterpolantSE, False
    project_root = Path(__file__).resolve().parents[1]
    try:
        source = subprocess.check_output(
            ["git", "show", "5443df1:bayesinf/applications/state_estimation/stochastic_interpolant.py"],
            cwd=project_root, text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit("Legacy interpolant checkpoint detected, but its model source is unavailable") from exc
    namespace = {}
    exec(compile(source, "legacy_stochastic_interpolant.py", "exec"), namespace)
    return namespace["StochasticInterpolantSE"], True


def benchmark(name, fn, batch_size, warmups, repeats, samples):
    for i in range(warmups):
        jax.block_until_ready(fn(jr.PRNGKey(10_000 + i)))
    values = []
    for i in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(fn(jr.PRNGKey(20_000 + i)))
        values.append(1e3 * (time.perf_counter() - start))
    median = statistics.median(values)
    return {
        "method": name,
        "batch_size": batch_size,
        "posterior_samples": samples,
        "median_batch_ms": median,
        "mean_batch_ms": statistics.mean(values),
        "stdev_batch_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min_batch_ms": min(values),
        "median_ms_per_trajectory": median / batch_size,
        "trajectories_per_second": 1000.0 * batch_size / median,
    }


def main():
    a = parse_args()
    if a.horizon is None:
        checkpoint = None
        if a.methods == ["interpolant"]:
            checkpoint = a.interpolant_checkpoint
        elif a.methods == ["flow_sde"]:
            checkpoint = a.flow_sde_checkpoint
        match = re.search(r"_h(\d+)_", checkpoint.name) if checkpoint else None
        a.horizon = int(match.group(1)) if match else 100
    if min(a.batch_size, a.horizon, a.state_dim, a.warmups, a.repeats, a.particles) < 1:
        raise SystemExit("batch size, dimensions, warmups, repeats, and particles must be positive")
    d_y = (a.state_dim + a.obs_interval - 1) // a.obs_interval
    key = jr.PRNGKey(a.seed)
    y = jr.normal(key, (a.batch_size, a.horizon, d_y))
    x0 = jnp.zeros((a.batch_size, a.state_dim))
    u = jnp.zeros((a.batch_size, a.horizon, 1))
    H = jnp.eye(a.state_dim)[:: a.obs_interval]
    rows = []

    for method in a.methods:
        if method == "ekf":
            @jax.jit
            def run(k):
                result = extended_kalman_filter(y, x0, H, dt=a.dt)
                # Draw from every filtering marginal N(m_t, P_t). These have
                # the same (B, P, T, D) materialized shape as PF particles,
                # but are not temporally correlated trajectory draws.
                eye = jnp.eye(a.state_dim, dtype=result.covariance.dtype)
                chol = jnp.linalg.cholesky(result.covariance + 1e-6 * eye)
                eps = jr.normal(
                    k, (a.batch_size, a.particles, a.horizon, a.state_dim),
                    dtype=result.mean.dtype,
                )
                return result.mean[:, None] + jnp.einsum("btij,bptj->bpti", chol, eps)
            samples = a.particles
        elif method == "ukf":
            @jax.jit
            def run(k):
                result = unscented_kalman_filter(y, x0, H, dt=a.dt)
                eye = jnp.eye(a.state_dim, dtype=result.covariance.dtype)
                chol = jnp.linalg.cholesky(result.covariance + 1e-6 * eye)
                eps = jr.normal(
                    k, (a.batch_size, a.particles, a.horizon, a.state_dim),
                    dtype=result.mean.dtype,
                )
                return result.mean[:, None] + jnp.einsum("btij,bptj->bpti", chol, eps)
            samples = a.particles
        elif method == "pf":
            run = jax.jit(lambda k: bootstrap_particle_filter(
                y, x0, H, k, dt=a.dt, num_particles=a.particles, noise_type="gmm"
            ).particles)
            samples = a.particles
        elif method == "amortized":
            path = require_checkpoint(a.amortized_checkpoint, method)
            model = AmortizedStateFlow(
                a.state_dim, d_y, 1, 64, 96, jr.PRNGKey(1), flow_type=a.amortized_flow,
                n_coupling=8, hidden_size=128, n_hidden=2,
                encoder_type=a.amortized_encoder, gru_hidden=32,
            )
            model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(path, model))
            # Match segmented training/evaluation by treating segments as the batch.
            seg = a.amortized_segment_length
            if a.horizon % seg:
                raise SystemExit("horizon must be divisible by --amortized-segment-length")
            ys = y.reshape(-1, seg, d_y)
            us = u.reshape(-1, seg, 1)
            x0s = jnp.repeat(x0, a.horizon // seg, axis=0)
            @eqx.filter_jit
            def run(k):
                return model.sample(ys, us, x0s, k, a.particles, prior="iid", fm_steps=a.fm_steps)
            samples = a.particles
        elif method == "latent":
            path = require_checkpoint(a.latent_checkpoint, method)
            model = LatentSSMFlow(
                a.state_dim, d_y, 1, 64, 96, a.latent_segment_length,
                encoder_type=a.latent_encoder, local_window=a.latent_local_window,
                local_window_prev_only=a.latent_local_window_prev_only,
                flow_type=a.latent_flow, n_coupling=8, hidden_size=128,
                n_hidden=2, max_rho=a.latent_max_rho, key=jr.PRNGKey(1),
            )
            model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(path, model))
            @eqx.filter_jit
            def run(k):
                keys = jr.split(k, a.batch_size)
                return jax.vmap(lambda yi, ui, xi, ki: model.sample_posterior(
                    yi, ui, xi, ki, a.particles
                ))(y, u, x0, keys)
            samples = a.particles
        elif method == "flow_sde":
            path = require_checkpoint(a.flow_sde_checkpoint, method)
            inferred = infer_flow_sde_dimensions(path, a.state_dim, d_y)
            d_c = a.flow_sde_d_c or inferred[0]
            d_hidden_enc = a.flow_sde_d_hidden_enc or inferred[1]
            d_hidden = a.flow_sde_d_hidden or inferred[2]
            d_hidden_diffusion = a.flow_sde_d_hidden_diffusion or inferred[3]
            print("FlowSDE template: "
                  f"d_c={d_c}, d_hidden_enc={d_hidden_enc}, "
                  f"d_hidden={d_hidden}, d_hidden_diffusion={d_hidden_diffusion}")
            model = RealNVPFlowSDE(
                a.state_dim, d_y, d_c=d_c, d_hidden_enc=d_hidden_enc,
                d_hidden=d_hidden, d_hidden_diffusion=d_hidden_diffusion, depth=3,
                observation_stride=a.observation_stride, time_step=a.dt, key=jr.PRNGKey(1),
            )
            model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(path, model))
            @eqx.filter_jit
            def run(k):
                keys = jr.split(k, a.batch_size)
                return jax.vmap(lambda yi, xi, ki: model.sample_posterior_custom_solver(
                    yi, xi, ki, a.particles, a.flow_sde_solver,
                    observation_stride=a.observation_stride,
                ))(y, x0, keys)
            samples = a.particles
        else:
            path = require_checkpoint(a.interpolant_checkpoint, method)
            match = re.search(r"_ms(\d+)(?:\.pt)?$", path.name)
            interpolant_segment_length = (
                a.interpolant_segment_length
                or (int(match.group(1)) if match else 20)
            )
            ModelClass, legacy = load_interpolant_class(path)
            print(f"Interpolant template: {'legacy lifter' if legacy else 'q_eta source'}, "
                  f"segment_length={interpolant_segment_length}")
            model = ModelClass(
                a.state_dim, d_y, d_c=a.interpolant_d_c,
                d_hidden=a.interpolant_d_hidden, depth=a.interpolant_depth,
                obs_interval=a.obs_interval, interpolant_type="linear", key=jr.PRNGKey(1),
            )
            if legacy:
                model = eqx.nn.inference_mode(eqx.tree_deserialise_leaves(path, model))
            else:
                model = eqx.nn.inference_mode(load_compatible_array_leaves(path, model))
            @eqx.filter_jit
            def run(k):
                return model.sample_posterior_batch(
                    y, k, a.particles, a.interpolant_steps,
                    stochastic=a.interpolant_sampler != "ode",
                    segmented=a.interpolant_sampler == "segmented",
                    seg_len=interpolant_segment_length,
                )
            samples = a.particles

        row = benchmark(method, run, a.batch_size, a.warmups, a.repeats, samples)
        rows.append(row)
        print(f"{method:12s} {row['median_ms_per_trajectory']:10.3f} ms/trajectory  "
              f"{row['trajectories_per_second']:10.2f} trajectories/s")

    metadata = {
        "device": str(jax.devices()[0]),
        "jax_version": jax.__version__,
        "python_version": platform.python_version(),
        "horizon": a.horizon,
        "state_dim": a.state_dim,
        "repeats": a.repeats,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) + list(metadata)
    append = a.output.exists() and not a.overwrite
    if append:
        with a.output.open(newline="") as f:
            existing_fields = csv.DictReader(f).fieldnames
        if existing_fields != fieldnames:
            raise SystemExit(
                f"Cannot append to {a.output}: CSV columns differ. "
                "Choose another --output or pass --overwrite."
            )
    with a.output.open("a" if append else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow(row | metadata)
    print(f"{'Appended to' if append else 'Saved'} {a.output}")


if __name__ == "__main__":
    main()
