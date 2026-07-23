"""Control-conditioned RealNVP FlowSDE for forward dynamics.

The normalized-state model is

    dx = f(x, u, t) dt + G(epsilon; x, u, t) sqrt(dt),
    epsilon ~ Normal(0, I).

Unlike the state-estimation FlowSDE, this model is causal: the conditioning
variable is the command at the current step, rather than a bidirectional
encoding of an observation trajectory.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPDiffusionNet,
    SDEDriftNet,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    StepNoisePath,
    get_solver,
)


class ControlledRealNVPFlowSDE(eqx.Module):
    """Euler--Maruyama controlled FlowSDE with exact transition density."""

    drift: SDEDriftNet
    diffusion: RealNVPDiffusionNet
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    diffusion_condition_on_control: bool = eqx.field(static=True)

    def __init__(self, state_dim, control_dim, key, *, hidden_size=64, depth=2,
                 diffusion_hidden=64, flow_layers=4, time_emb_dim=16,
                 realnvp_alpha=3.0, dt=0.02,
                 diffusion_condition_on_control=True):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = float(dt)
        self.diffusion_condition_on_control = bool(diffusion_condition_on_control)
        drift_key, diffusion_key = jr.split(key)
        self.drift = SDEDriftNet(
            state_dim, control_dim, hidden_size, depth, time_emb_dim, drift_key
        )
        self.diffusion = RealNVPDiffusionNet(
            state_dim, control_dim if self.diffusion_condition_on_control else 0,
            diffusion_hidden, flow_layers,
            time_emb_dim, diffusion_key, alpha=realnvp_alpha,
        )

    def _diffusion_context(self, control):
        if self.diffusion_condition_on_control:
            return control
        return jnp.empty((0,), dtype=control.dtype)

    def transition_nll(self, x0, u0, x1, time):
        """Mean exact Euler transition NLL for arrays ending in state_dim."""
        dt = jnp.asarray(self.dt, dtype=x0.dtype)

        def one(a, control, b, t):
            residual = (b - a - self.drift(a, control, t) * dt) / jnp.sqrt(dt)
            diffusion_context = self._diffusion_context(control)
            # Centre the invertible transform at epsilon=0. This makes the
            # deterministic/stochastic decomposition identifiable: the flow
            # cannot hide deterministic motion in coupling-layer translations.
            centre = self.diffusion.apply(
                jnp.zeros_like(residual), a, diffusion_context, t
            )
            epsilon, forward_log_det = self.diffusion.inverse_and_log_det(
                residual + centre, a, diffusion_context, t
            )
            base = 0.5 * jnp.sum(epsilon ** 2)
            base += 0.5 * self.state_dim * jnp.log(2.0 * jnp.pi)
            # p(x1|x0,u) includes both the flow and sqrt(dt) Jacobians.
            return (base + forward_log_det
                    + 0.5 * self.state_dim * jnp.log(dt)) / self.state_dim

        flat_shape = (-1, self.state_dim)
        values = jax.vmap(one)(
            x0.reshape(flat_shape), u0.reshape((-1, self.control_dim)),
            x1.reshape(flat_shape), time.reshape((-1,)),
        )
        return jnp.mean(values)

    def noisy_transition_nll(self, x0, u0, x1, time, key,
                             state_noise_std=0.0, control_noise_std=0.0):
        """Denoising likelihood: perturb inputs while keeping labels clean."""
        kx, ku = jr.split(key)
        noisy_x = x0 + state_noise_std * jr.normal(kx, x0.shape)
        noisy_u = u0 + control_noise_std * jr.normal(ku, u0.shape)
        return self.transition_nll(noisy_x, noisy_u, x1, time)

    def deterministic_rollout(self, x0, controls, start_step=0):
        """Roll out the learned drift (diffusion disabled)."""
        dt = jnp.asarray(self.dt, dtype=x0.dtype)

        def step(x, item):
            k, u = item
            t = (start_step + k) * dt
            next_x = x + self.drift(x, u, t) * dt
            return next_x, next_x

        _, path = jax.lax.scan(step, x0, (jnp.arange(len(controls)), controls))
        return path

    def _sample_path(self, x0, controls, key, start_step=0,
                     diffusion_temperature=1.0, solver_name="heun"):
        """Integrate one path with the same custom solvers as state FlowSDE."""
        dt = jnp.asarray(self.dt, dtype=x0.dtype)
        noises = jr.normal(key, (len(controls), self.state_dim))
        noise_path = StepNoisePath(noises, start_step * dt, dt)
        solver = get_solver(solver_name)

        def drift_fn(t, x):
            step = jnp.clip(jnp.rint(t / dt).astype(jnp.int32) - start_step,
                            0, len(controls) - 1)
            return self.drift(x, controls[step], t)

        def diffusion_fn(t, x, epsilon):
            step = jnp.clip(jnp.rint(t / dt).astype(jnp.int32) - start_step,
                            0, len(controls) - 1)
            u = controls[step]
            diffusion_context = self._diffusion_context(u)
            zero = jnp.zeros_like(epsilon)
            return (self.diffusion.apply(
                        epsilon * diffusion_temperature, x, diffusion_context, t
                    ) - self.diffusion.apply(zero, x, diffusion_context, t))

        def step(x, k):
            t0 = (start_step + k) * dt
            x1 = solver.step(x, t0, t0 + dt, drift_fn, diffusion_fn, noise_path)
            return x1, x1

        _, path = jax.lax.scan(step, x0, jnp.arange(len(controls)))
        return path

    def sample_rollout(self, x0, controls, key, n_particles=64, start_step=0,
                       diffusion_temperature=1.0, solver_name="heun"):
        """Return stochastic paths with shape (particles, time, state_dim)."""
        return jax.vmap(lambda particle_key: self._sample_path(
            x0, controls, particle_key, start_step,
            diffusion_temperature, solver_name,
        ))(jr.split(key, n_particles))

    def multiple_shooting_loss(self, states, controls, start_steps, key, *,
                               segment_length=16, n_particles=4,
                               solver_name="heun", shooting_noise_std=0.01,
                               control_noise_std=0.01,
                               uncertainty_weight=0.5,
                               continuity_weight=0.1):
        """Pathwise energy score over stochastic multiple-shooting segments.

        ``states`` has shape (B, W+1, dx), ``controls`` (B, W, du), and W
        must be divisible by ``segment_length``. Each segment is integrated by
        the selected stochastic solver instead of evaluating one-step NLL.
        """
        window_steps = controls.shape[1]
        if window_steps % segment_length:
            raise ValueError("window steps must be divisible by segment length")
        n_segments = window_steps // segment_length
        batch_keys = jr.split(key, states.shape[0])

        def trajectory_loss(x, u, first_step, trajectory_key):
            control_key, shooting_key = jr.split(trajectory_key)
            if control_noise_std > 0:
                u = u + control_noise_std * jr.normal(control_key, u.shape)
            segment_keys = jr.split(shooting_key, n_segments)
            state_starts = x[:-1].reshape(
                n_segments, segment_length, self.state_dim
            )[:, 0]
            targets = x[1:].reshape(
                n_segments, segment_length, self.state_dim
            )
            control_segments = u.reshape(
                n_segments, segment_length, self.control_dim
            )
            segment_starts = first_step + jnp.arange(n_segments) * segment_length

            def segment_loss(x0, segment_controls, target, segment_start, seg_key):
                init_key, path_key = jr.split(seg_key)
                if shooting_noise_std > 0:
                    x0 = x0 + shooting_noise_std * jr.normal(init_key, x0.shape)
                paths = self.sample_rollout(
                    x0, segment_controls, path_key, n_particles,
                    segment_start, 1.0, solver_name,
                )
                accuracy = jnp.mean(jnp.sqrt(
                    jnp.sum((paths - target[None]) ** 2, axis=-1) + 1e-6
                ))
                pairwise = paths[:, None] - paths[None, :]
                spread = jnp.mean(jnp.sqrt(
                    jnp.sum(pairwise ** 2, axis=-1) + 1e-6
                ))
                continuity = jnp.mean(
                    (jnp.mean(paths[:, -1], axis=0) - target[-1]) ** 2
                )
                return accuracy - uncertainty_weight * spread, continuity

            scores, continuities = jax.vmap(segment_loss)(
                state_starts, control_segments, targets, segment_starts, segment_keys
            )
            return jnp.mean(scores) + continuity_weight * jnp.mean(continuities)

        return jnp.mean(jax.vmap(trajectory_loss)(
            states, controls, start_steps, batch_keys
        ))
