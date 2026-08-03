"""Classical filtering baselines for the Lorenz-96 state-estimation tasks.

The EKF is intended for the Gaussian-observation AmortizedStateFlow experiment.
The bootstrap particle filter supports the heavy-tailed and multimodal noise used
by LatentSSMFlow, FlowSDE, and the conditional stochastic interpolant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.special import gammaln, logsumexp

from .lorenz96_dataset import rk4_step


@dataclass
class FilterResult:
    """Equally weighted filtering particles and their first two moments."""

    mean: jnp.ndarray                 # (batch, time, state)
    covariance: jnp.ndarray           # (batch, time, state, state)
    particles: Optional[jnp.ndarray] = None  # (batch, time, particle, state)


def _as_batch(y: jnp.ndarray, x0: jnp.ndarray):
    single = y.ndim == 2
    if single:
        y, x0 = y[None], x0[None]
    if y.ndim != 3 or x0.ndim != 2 or y.shape[0] != x0.shape[0]:
        raise ValueError("Expected y=(B,T,Dy), x0=(B,Dx), or their unbatched forms")
    return y, x0, single


def extended_kalman_filter(
    y: jnp.ndarray,
    x0: jnp.ndarray,
    H: jnp.ndarray,
    *,
    dt: float,
    F: float = 8.0,
    obs_std: float = 0.5,
    process_std: float = 0.05,
    initial_std: float = 0.0,
) -> FilterResult:
    """Run an EKF using the exact RK4 map and its automatic Jacobian.

    Observations correspond to states one transition after ``x0``, matching
    :func:`generate_lorenz96_dataset`. Joseph's covariance update is used for
    numerical stability.
    """
    if obs_std <= 0 or process_std < 0 or initial_std < 0:
        raise ValueError("obs_std must be positive and state-noise scales non-negative")
    y, x0, single = _as_batch(jnp.asarray(y), jnp.asarray(x0))
    H = jnp.asarray(H)
    n, dy = x0.shape[-1], y.shape[-1]
    if H.shape != (dy, n):
        raise ValueError(f"H must have shape {(dy, n)}, got {H.shape}")

    transition = lambda x: rk4_step(x, dt, F)
    jacobian = jax.jacfwd(transition)
    eye = jnp.eye(n, dtype=x0.dtype)
    Q = process_std**2 * eye
    R = obs_std**2 * jnp.eye(dy, dtype=x0.dtype)

    def one_sequence(obs, initial):
        def step(carry, measurement):
            mean, cov = carry
            A = jacobian(mean)
            pred = transition(mean)
            pred_cov = A @ cov @ A.T + Q
            innovation_cov = H @ pred_cov @ H.T + R
            # solve is more stable than explicitly forming inv(S)
            gain = jnp.linalg.solve(innovation_cov, H @ pred_cov).T
            mean = pred + gain @ (measurement - H @ pred)
            ikh = eye - gain @ H
            cov = ikh @ pred_cov @ ikh.T + gain @ R @ gain.T
            cov = 0.5 * (cov + cov.T)
            return (mean, cov), (mean, cov)

        init_cov = initial_std**2 * eye
        _, (means, covs) = jax.lax.scan(step, (initial, init_cov), obs)
        return means, covs

    means, covs = jax.vmap(one_sequence)(y, x0)
    if single:
        means, covs = means[0], covs[0]
    return FilterResult(means, covs)


def unscented_kalman_filter(
    y: jnp.ndarray,
    x0: jnp.ndarray,
    H: jnp.ndarray,
    *,
    dt: float,
    F: float = 8.0,
    obs_std: float = 0.5,
    process_std: float = 0.05,
    initial_std: float = 0.0,
    alpha: float = 0.5,
    beta: float = 2.0,
    kappa: float = 0.0,
) -> FilterResult:
    """Run a scaled unscented Kalman filter with the exact RK4 transition."""
    if obs_std <= 0 or process_std < 0 or initial_std < 0 or alpha <= 0:
        raise ValueError("Noise scales must be valid and alpha positive")
    y, x0, single = _as_batch(jnp.asarray(y), jnp.asarray(x0))
    H = jnp.asarray(H)
    n, dy = x0.shape[-1], y.shape[-1]
    if H.shape != (dy, n):
        raise ValueError(f"H must have shape {(dy, n)}, got {H.shape}")

    lam = alpha**2 * (n + kappa) - n
    scale = n + lam
    wm = jnp.full(2 * n + 1, 1.0 / (2.0 * scale), dtype=x0.dtype)
    wc = wm.at[0].set(lam / scale + (1.0 - alpha**2 + beta))
    wm = wm.at[0].set(lam / scale)
    eye = jnp.eye(n, dtype=x0.dtype)
    Q = process_std**2 * eye
    R = obs_std**2 * jnp.eye(dy, dtype=x0.dtype)

    def sigma_points(mean, covariance):
        root = jnp.linalg.cholesky(scale * (covariance + 1e-7 * eye))
        offsets = root.T
        return jnp.concatenate((mean[None], mean + offsets, mean - offsets), axis=0)

    def one_sequence(obs, initial):
        def step(carry, measurement):
            mean, covariance = carry
            sigma = sigma_points(mean, covariance)
            propagated = jax.vmap(lambda point: rk4_step(point, dt, F))(sigma)
            pred_mean = jnp.sum(wm[:, None] * propagated, axis=0)
            dx = propagated - pred_mean
            pred_cov = jnp.einsum("s,si,sj->ij", wc, dx, dx) + Q

            obs_sigma = propagated @ H.T
            obs_mean = jnp.sum(wm[:, None] * obs_sigma, axis=0)
            dz = obs_sigma - obs_mean
            innovation_cov = jnp.einsum("s,si,sj->ij", wc, dz, dz) + R
            cross_cov = jnp.einsum("s,si,sj->ij", wc, dx, dz)
            gain = jnp.linalg.solve(innovation_cov, cross_cov.T).T
            mean = pred_mean + gain @ (measurement - obs_mean)
            covariance = pred_cov - gain @ innovation_cov @ gain.T
            covariance = 0.5 * (covariance + covariance.T)
            return (mean, covariance), (mean, covariance)

        init_cov = initial_std**2 * eye
        _, (means, covs) = jax.lax.scan(step, (initial, init_cov), obs)
        return means, covs

    means, covs = jax.vmap(one_sequence)(y, x0)
    if single:
        means, covs = means[0], covs[0]
    return FilterResult(means, covs)


def _observation_log_likelihood(
    residual: jnp.ndarray,
    noise_type: str,
    obs_std: float,
    obs_df: float,
    gmm_weights: Sequence[float],
    gmm_offsets: Sequence[float],
    gmm_stds: Sequence[float],
) -> jnp.ndarray:
    """Log likelihood per particle; observation dimensions are independent."""
    if noise_type == "gaussian":
        z = residual / obs_std
        return jnp.sum(-0.5 * z**2 - jnp.log(obs_std) - 0.5 * jnp.log(2 * jnp.pi), -1)
    if noise_type == "student_t":
        nu = jnp.asarray(obs_df, residual.dtype)
        z = residual / obs_std
        c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * jnp.log(nu * jnp.pi) - jnp.log(obs_std)
        return jnp.sum(c - 0.5 * (nu + 1) * jnp.log1p(z**2 / nu), -1)
    if noise_type == "gmm":
        w = jnp.asarray(gmm_weights, residual.dtype)
        mu = obs_std * jnp.asarray(gmm_offsets, residual.dtype)
        sd = obs_std * jnp.asarray(gmm_stds, residual.dtype)
        if not (w.ndim == mu.ndim == sd.ndim == 1 and len(w) == len(mu) == len(sd)):
            raise ValueError("GMM weights, offsets, and stds must be equal-length vectors")
        z = (residual[..., None] - mu) / sd
        components = jnp.log(w / jnp.sum(w)) - jnp.log(sd) - 0.5 * (z**2 + jnp.log(2 * jnp.pi))
        return jnp.sum(logsumexp(components, axis=-1), axis=-1)
    raise ValueError(f"Unknown observation noise type: {noise_type}")


def bootstrap_particle_filter(
    y: jnp.ndarray,
    x0: jnp.ndarray,
    H: jnp.ndarray,
    key: jax.Array,
    *,
    dt: float,
    F: float = 8.0,
    obs_std: float = 0.5,
    process_std: float = 0.05,
    initial_std: float = 0.0,
    num_particles: int = 1000,
    noise_type: str = "student_t",
    obs_df: float = 5.0,
    gmm_weights: Sequence[float] = (0.70, 0.15, 0.15),
    gmm_offsets: Sequence[float] = (0.0, 2.5, -2.5),
    gmm_stds: Sequence[float] = (0.50, 1.50, 1.50),
) -> FilterResult:
    """Bootstrap/SIR particle filter with the experiment's exact noise model."""
    if num_particles < 2 or obs_std <= 0 or process_std < 0 or initial_std < 0:
        raise ValueError("Need >=2 particles, positive obs_std, and non-negative state noise")
    if noise_type == "gmm" and (
        any(value <= 0 for value in gmm_weights)
        or any(value <= 0 for value in gmm_stds)
    ):
        raise ValueError("GMM weights and standard deviations must be positive")
    y, x0, single = _as_batch(jnp.asarray(y), jnp.asarray(x0))
    H = jnp.asarray(H)
    batch, _, n = y.shape[0], y.shape[1], x0.shape[-1]
    if H.shape != (y.shape[-1], n):
        raise ValueError(f"H must have shape {(y.shape[-1], n)}, got {H.shape}")
    sequence_keys = jr.split(key, batch)

    def one_sequence(obs, initial, seq_key):
        seq_key, init_key = jr.split(seq_key)
        particles = initial + initial_std * jr.normal(init_key, (num_particles, n))

        def step(carry, measurement):
            particles, step_key = carry
            step_key, noise_key, sample_key = jr.split(step_key, 3)
            particles = jax.vmap(lambda p: rk4_step(p, dt, F))(particles)
            particles = particles + process_std * jr.normal(noise_key, particles.shape)
            residual = measurement[None, :] - particles @ H.T
            logw = _observation_log_likelihood(
                residual, noise_type, obs_std, obs_df,
                gmm_weights, gmm_offsets, gmm_stds,
            )
            logw = logw - logsumexp(logw)
            indices = jr.categorical(sample_key, logw, shape=(num_particles,))
            particles = particles[indices]
            mean = jnp.mean(particles, axis=0)
            centered = particles - mean
            cov = centered.T @ centered / (num_particles - 1)
            return (particles, step_key), (mean, cov, particles)

        _, (means, covs, history) = jax.lax.scan(step, (particles, seq_key), obs)
        return means, covs, history

    means, covs, particles = jax.vmap(one_sequence)(y, x0, sequence_keys)
    if single:
        means, covs, particles = means[0], covs[0], particles[0]
    return FilterResult(means, covs, particles)
