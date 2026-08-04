"""Synthetic lane-keeping data with an analytic, optimization-free expert."""

from dataclasses import dataclass
import math
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr


@dataclass(frozen=True)
class BicycleConfig:
    dt: float = 0.1
    velocity: float = 8.0
    wheelbase: float = 2.7
    delta_max: float = math.radians(35.0)
    obs_history: int = 20
    horizon: int = 15
    obs_std: float = 0.12
    lateral_gain: float = 0.55
    heading_gain: float = 1.4
    recovery_lateral_max: float = 1.2
    recovery_heading_max: float = 0.25
    mpc_iterations: int = 80
    mpc_learning_rate: float = 0.04


def wrap_angle(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.arctan2(jnp.sin(x), jnp.cos(x))


def bicycle_step(state: jnp.ndarray, delta: jnp.ndarray, cfg: BicycleConfig) -> jnp.ndarray:
    """Rear-axle kinematic bicycle step for state ``[x, y, psi]``."""
    x, y, psi = state
    delta = jnp.clip(delta, -cfg.delta_max, cfg.delta_max)
    return jnp.array([
        x + cfg.dt * cfg.velocity * jnp.cos(psi),
        y + cfg.dt * cfg.velocity * jnp.sin(psi),
        wrap_angle(psi + cfg.dt * cfg.velocity / cfg.wheelbase * jnp.tan(delta)),
    ])


def _path(key: jax.Array, n: int, cfg: BicycleConfig) -> Tuple[jnp.ndarray, ...]:
    """Random sinusoidal flat output, filtered by steering feasibility."""
    k1, k2, k3 = jr.split(key, 3)
    amp = jr.uniform(k1, (3,), minval=0.15, maxval=0.9)
    freq = jr.uniform(k2, (3,), minval=0.025, maxval=0.09)
    phase = jr.uniform(k3, (3,), minval=-jnp.pi, maxval=jnp.pi)
    x = jnp.arange(n) * cfg.velocity * cfg.dt
    q = x[:, None] * freq + phase
    y = jnp.sum(amp * jnp.sin(q), axis=1)
    dy = jnp.sum(amp * freq * jnp.cos(q), axis=1)
    ddy = -jnp.sum(amp * freq**2 * jnp.sin(q), axis=1)
    psi = jnp.arctan(dy)
    curvature = ddy / jnp.maximum((1.0 + dy**2) ** 1.5, 1e-6)
    delta_ff = jnp.arctan(cfg.wheelbase * curvature)
    # Scale curvature rather than silently retaining infeasible examples.
    scale = jnp.minimum(1.0, 0.9 * cfg.delta_max / (jnp.max(jnp.abs(delta_ff)) + 1e-6))
    y, dy, ddy = y * scale, dy * scale, ddy * scale
    psi = jnp.arctan(dy)
    curvature = ddy / jnp.maximum((1.0 + dy**2) ** 1.5, 1e-6)
    return x, y, psi, curvature


def expert_chunk(e_y: jnp.ndarray, e_psi: jnp.ndarray, curvature: jnp.ndarray,
                 cfg: BicycleConfig) -> jnp.ndarray:
    """Analytic feed-forward + stabilizing feedback steering rollout."""
    def step(carry, kappa):
        ey, ep = carry
        yaw_rate = cfg.velocity * kappa - cfg.lateral_gain * ey - cfg.heading_gain * ep
        delta = jnp.clip(jnp.arctan(cfg.wheelbase * yaw_rate / cfg.velocity),
                         -cfg.delta_max, cfg.delta_max)
        ey_n = ey + cfg.dt * cfg.velocity * jnp.sin(ep)
        ep_n = wrap_angle(ep + cfg.dt * (cfg.velocity / cfg.wheelbase * jnp.tan(delta)
                                         - cfg.velocity * kappa))
        return (ey_n, ep_n), delta
    return jax.lax.scan(step, (e_y, e_psi), curvature)[1]


def nonlinear_mpc_chunk(e_y: jnp.ndarray, e_psi: jnp.ndarray,
                        curvature: jnp.ndarray, previous_steering: jnp.ndarray,
                        cfg: BicycleConfig) -> jnp.ndarray:
    """Solve the bounded nonlinear MPC problem with projected Adam.

    A fixed-iteration JAX solver makes expert generation vectorizable over the
    entire dataset.  It uses the same nonlinear ``sin``/``tan`` prediction,
    quadratic tracking/control costs, terminal multiplier, steering-rate cost,
    and hard steering bounds as the SciPy comparison controller.
    """
    lateral_weight = 8.0
    heading_weight = 3.0
    steering_weight = 0.15
    steering_rate_weight = 1.0
    terminal_multiplier = 3.0

    def objective(controls):
        def step(carry, values):
            ey, ep, previous, cost, index = carry
            delta, kappa = values
            ey = ey + cfg.dt * cfg.velocity * jnp.sin(ep)
            ep = wrap_angle(ep + cfg.dt * (
                cfg.velocity / cfg.wheelbase * jnp.tan(delta)
                - cfg.velocity * kappa
            ))
            terminal = jnp.where(
                index == curvature.shape[0] - 1, terminal_multiplier, 1.0
            )
            cost = cost + terminal * (
                lateral_weight * ey**2 + heading_weight * ep**2
            )
            cost = cost + steering_weight * delta**2
            cost = cost + steering_rate_weight * (delta - previous)**2
            return (ey, ep, delta, cost, index + 1), None

        initial = (e_y, e_psi, previous_steering, jnp.array(0.0), jnp.array(0))
        final, _ = jax.lax.scan(step, initial, (controls, curvature))
        return final[3]

    # The stabilizing policy is a strong feasible warm start. Optimisation then
    # incorporates the actual MPC rate and terminal costs.
    controls = expert_chunk(e_y, e_psi, curvature, cfg)
    first_moment = jnp.zeros_like(controls)
    second_moment = jnp.zeros_like(controls)

    def optimize(index, carry):
        values, moment1, moment2 = carry
        gradient = jax.grad(objective)(values)
        moment1 = 0.9 * moment1 + 0.1 * gradient
        moment2 = 0.999 * moment2 + 0.001 * gradient**2
        step = index + 1
        corrected1 = moment1 / (1.0 - 0.9**step)
        corrected2 = moment2 / (1.0 - 0.999**step)
        values = jnp.clip(
            values - cfg.mpc_learning_rate * corrected1 / (jnp.sqrt(corrected2) + 1e-8),
            -cfg.delta_max,
            cfg.delta_max,
        )
        return values, moment1, moment2

    controls, _, _ = jax.lax.fori_loop(
        0, cfg.mpc_iterations, optimize,
        (controls, first_moment, second_moment),
    )
    return controls


def generate_dataset(key: jax.Array, n_samples: int, cfg: BicycleConfig) -> Dict[str, jnp.ndarray]:
    """Return observation histories, preview features, and bounded action chunks.

    ``reference[..., :]`` contains lateral centerline displacement relative to
    the current point, heading, and curvature. Shapes are ``(N, O, 1)``,
    ``(N, H, 3)``, and ``(N, H)``.
    """
    path_len = cfg.obs_history + cfg.horizon + 8
    keys = jr.split(key, n_samples)

    def one(k):
        kp, ke, kh, kn, ku = jr.split(k, 5)
        _, y, psi, curvature = _path(kp, path_len, cfg)
        # Sample the state at the prediction point, rather than before a long
        # stabilising rollout. Recovery states therefore remain in the data.
        e_y = jr.uniform(
            ke, (), minval=-cfg.recovery_lateral_max,
            maxval=cfg.recovery_lateral_max,
        )
        e_psi = jr.uniform(
            kh, (), minval=-cfg.recovery_heading_max,
            maxval=cfg.recovery_heading_max,
        )
        i = cfg.obs_history

        # Smooth, bounded past actuation around curvature feed-forward.
        raw_noise = jr.normal(ku, (i,))
        _, smooth_noise = jax.lax.scan(
            lambda previous, value: (0.8 * previous + 0.2 * value,
                                     0.8 * previous + 0.2 * value),
            jnp.array(0.0), raw_noise,
        )
        past_controls = jnp.clip(
            jnp.arctan(cfg.wheelbase * curvature[:i]) + 0.12 * smooth_noise,
            -cfg.delta_max, cfg.delta_max,
        )

        # Reconstruct x_0,...,x_{O-1} backward from the sampled current x_O.
        # Forward application of u_k maps x_k to measurement state x_{k+1};
        # hence token k is exactly [y_{k+1}, u_k].
        def backward(next_state, values):
            ey_next, ep_next = next_state
            delta, kappa = values
            ep_prev = wrap_angle(ep_next - cfg.dt * (
                cfg.velocity / cfg.wheelbase * jnp.tan(delta)
                - cfg.velocity * kappa
            ))
            ey_prev = ey_next - cfg.dt * cfg.velocity * jnp.sin(ep_prev)
            return (ey_prev, ep_prev), (ey_next, ep_next)

        _, reverse_states = jax.lax.scan(
            backward, (e_y, e_psi),
            (past_controls[::-1], curvature[:i][::-1]),
        )
        ey_hist = reverse_states[0][::-1]
        obs = ey_hist + cfg.obs_std * jr.normal(kn, (cfg.obs_history,))
        sl = slice(i, i + cfg.horizon)
        ref = jnp.stack([y[sl] - y[i], wrap_angle(psi[sl] - psi[i]), curvature[sl]], axis=-1)
        actions = nonlinear_mpc_chunk(
            e_y, e_psi, curvature[sl], past_controls[-1], cfg
        )
        return (obs[:, None], past_controls[:, None], past_controls[-1:],
                ref, actions, jnp.array([e_y, e_psi]))

    obs, past, previous, ref, actions, errors = jax.vmap(one)(keys)
    return {
        "observations": obs,
        "past_controls": past,
        "previous_steering": previous,
        "reference": ref,
        "actions": actions,
        "errors": errors,
    }


def split_dataset(data: Dict[str, jnp.ndarray], key: jax.Array,
                  train_fraction: float = 0.8, val_fraction: float = 0.1):
    n = data["actions"].shape[0]
    p = jr.permutation(key, n)
    n_train, n_val = int(train_fraction * n), int(val_fraction * n)
    take = lambda ids: {k: v[ids] for k, v in data.items()}
    return take(p[:n_train]), take(p[n_train:n_train + n_val]), take(p[n_train + n_val:])
