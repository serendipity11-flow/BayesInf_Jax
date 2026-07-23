"""Velocity-augmented, structured second-order controlled RealNVP FlowSDE."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPDiffusionNet,
    SinusoidalEmb,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    StepNoisePath,
    get_solver,
)


class AccelerationNet(eqx.Module):
    net: eqx.nn.MLP
    time_emb: SinusoidalEmb

    def __init__(self, position_dim, control_dim, hidden_size, depth,
                 time_emb_dim, key):
        self.time_emb = SinusoidalEmb(time_emb_dim)
        self.net = eqx.nn.MLP(
            2 * position_dim + control_dim + time_emb_dim, position_dim,
            hidden_size, depth, activation=jax.nn.gelu, key=key,
        )

    def __call__(self, state, control, time):
        return self.net(jnp.concatenate([state, control, self.time_emb(time)]))


class SecondOrderControlledRealNVPFlowSDE(eqx.Module):
    """dq=v dt, dv=a(q,v,u,t)dt+g(eps;q,v,t)sqrt(dt)."""

    acceleration: AccelerationNet
    diffusion: RealNVPDiffusionNet
    position_dim: int = eqx.field(static=True)
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    diffusion_condition_on_control: bool = eqx.field(static=True, default=False)
    velocity_to_position_scale: jnp.ndarray
    velocity_to_position_bias: jnp.ndarray

    def __init__(self, position_dim, control_dim, key, *, hidden_size=64,
                 depth=2, diffusion_hidden=64, flow_layers=4,
                 time_emb_dim=16, realnvp_alpha=3.0, dt=0.02,
                 velocity_to_position_scale=None,
                 velocity_to_position_bias=None):
        self.position_dim = position_dim
        self.state_dim = 2 * position_dim
        self.control_dim = control_dim
        self.dt = float(dt)
        self.diffusion_condition_on_control = False
        if velocity_to_position_scale is None:
            velocity_to_position_scale = jnp.ones(position_dim)
        if velocity_to_position_bias is None:
            velocity_to_position_bias = jnp.zeros(position_dim)
        self.velocity_to_position_scale = jnp.asarray(velocity_to_position_scale)
        self.velocity_to_position_bias = jnp.asarray(velocity_to_position_bias)
        acceleration_key, diffusion_key = jr.split(key)
        self.acceleration = AccelerationNet(
            position_dim, control_dim, hidden_size, depth,
            time_emb_dim, acceleration_key,
        )
        # q is passed as z_state and v as context: g(epsilon; q, v, t).
        self.diffusion = RealNVPDiffusionNet(
            position_dim, position_dim, diffusion_hidden, flow_layers,
            time_emb_dim, diffusion_key, alpha=realnvp_alpha,
        )

    def transition_nll(self, z0, u0, z1, time):
        dt = jnp.asarray(self.dt, dtype=z0.dtype)
        d = self.position_dim

        def one(a, control, b, t):
            q0, v0 = a[:d], a[d:]
            v1 = b[d:]
            residual = (v1 - v0 - self.acceleration(a, control, t) * dt) / jnp.sqrt(dt)
            centre = self.diffusion.apply(jnp.zeros(d), q0, v0, t)
            epsilon, log_det = self.diffusion.inverse_and_log_det(
                residual + centre, q0, v0, t
            )
            base = 0.5 * jnp.sum(epsilon ** 2) + 0.5 * d * jnp.log(2 * jnp.pi)
            return (base + log_det + 0.5 * d * jnp.log(dt)) / d

        return jnp.mean(jax.vmap(one)(
            z0.reshape(-1, self.state_dim), u0.reshape(-1, self.control_dim),
            z1.reshape(-1, self.state_dim), time.reshape(-1),
        ))

    def deterministic_rollout(self, z0, controls, start_step=0):
        dt = jnp.asarray(self.dt, dtype=z0.dtype)
        d = self.position_dim

        def step(z, item):
            k, u = item
            t = (start_step + k) * dt
            q, v = z[:d], z[d:]
            acceleration = self.acceleration(z, u, t)
            # Semi-implicit Euler is stable and matches the stochastic update.
            v1 = v + acceleration * dt
            q_rate = (self.velocity_to_position_scale * v1
                      + self.velocity_to_position_bias)
            q1 = q + q_rate * dt
            z1 = jnp.concatenate([q1, v1])
            return z1, z1
        _, path = jax.lax.scan(step, z0, (jnp.arange(len(controls)), controls))
        return path

    def _sample_path(self, z0, controls, key, start_step=0,
                     diffusion_temperature=1.0, solver_name="heun"):
        dt = jnp.asarray(self.dt, dtype=z0.dtype)
        d = self.position_dim
        noises = jr.normal(key, (len(controls), d))
        noise_path = StepNoisePath(noises, start_step * dt, dt)
        solver = get_solver(solver_name)

        def control_at(t):
            index = jnp.clip(jnp.rint(t / dt).astype(jnp.int32) - start_step,
                             0, len(controls) - 1)
            return controls[index]

        def drift_fn(t, z):
            q_rate = (self.velocity_to_position_scale * z[d:]
                      + self.velocity_to_position_bias)
            return jnp.concatenate([q_rate, self.acceleration(z, control_at(t), t)])

        def diffusion_fn(t, z, epsilon):
            q, v = z[:d], z[d:]
            zero = jnp.zeros_like(epsilon)
            noise = (self.diffusion.apply(
                epsilon * diffusion_temperature, q, v, t
            ) - self.diffusion.apply(zero, q, v, t))
            return jnp.concatenate([jnp.zeros(d, dtype=z.dtype), noise])

        def step(z, k):
            t0 = (start_step + k) * dt
            z1 = solver.step(z, t0, t0 + dt, drift_fn, diffusion_fn, noise_path)
            return z1, z1
        _, path = jax.lax.scan(step, z0, jnp.arange(len(controls)))
        return path

    def sample_rollout(self, z0, controls, key, n_particles=64, start_step=0,
                       diffusion_temperature=1.0, solver_name="heun"):
        return jax.vmap(lambda particle_key: self._sample_path(
            z0, controls, particle_key, start_step,
            diffusion_temperature, solver_name,
        ))(jr.split(key, n_particles))

    def multiple_shooting_loss(self, states, controls, start_steps, key, *,
                               segment_length=16, n_particles=4,
                               solver_name="heun", shooting_noise_std=0.01,
                               control_noise_std=0.01,
                               uncertainty_weight=0.5,
                               continuity_weight=0.1):
        window_steps = controls.shape[1]
        if window_steps % segment_length:
            raise ValueError("window steps must be divisible by segment length")
        n_segments, d = window_steps // segment_length, self.position_dim
        batch_keys = jr.split(key, len(states))

        def trajectory_loss(z, u, first_step, trajectory_key):
            control_key, segment_root = jr.split(trajectory_key)
            if control_noise_std > 0:
                u = u + control_noise_std * jr.normal(control_key, u.shape)
            starts = z[:-1].reshape(n_segments, segment_length, self.state_dim)[:, 0]
            targets = z[1:].reshape(n_segments, segment_length, self.state_dim)
            u_segments = u.reshape(n_segments, segment_length, self.control_dim)
            times = first_step + jnp.arange(n_segments) * segment_length
            keys = jr.split(segment_root, n_segments)

            def segment_loss(z0, us, target, segment_start, segment_key):
                init_key, path_key = jr.split(segment_key)
                if shooting_noise_std > 0:
                    z0 = z0 + shooting_noise_std * jr.normal(init_key, z0.shape)
                paths = self.sample_rollout(
                    z0, us, path_key, n_particles, segment_start, 1.0, solver_name
                )
                # Position accuracy dominates; velocity supplies Markov state.
                position_distance = jnp.sqrt(jnp.sum(
                    (paths[..., :d] - target[None, ..., :d]) ** 2, axis=-1
                ) + 1e-6)
                velocity_distance = jnp.sqrt(jnp.sum(
                    (paths[..., d:] - target[None, ..., d:]) ** 2, axis=-1
                ) + 1e-6)
                accuracy = jnp.mean(position_distance + 0.1 * velocity_distance)
                pairwise = paths[:, None, :, :d] - paths[None, :, :, :d]
                spread = jnp.mean(jnp.sqrt(jnp.sum(pairwise ** 2, axis=-1) + 1e-6))
                continuity = jnp.mean((jnp.mean(paths[:, -1], axis=0) - target[-1]) ** 2)
                return accuracy - uncertainty_weight * spread, continuity

            scores, continuity = jax.vmap(segment_loss)(
                starts, u_segments, targets, times, keys
            )
            return jnp.mean(scores) + continuity_weight * jnp.mean(continuity)

        return jnp.mean(jax.vmap(trajectory_loss)(
            states, controls, start_steps, batch_keys
        ))
