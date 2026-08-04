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
from collections.abc import Callable

from bayesinf.applications.state_estimation.flow_sde import (
    RealNVPDiffusionNet,
    SDEDriftNet,
)
from bayesinf.applications.state_estimation.stochastic_solvers import (
    StepNoisePath,
    get_solver,
)
ARM_INDICES = jnp.asarray((0, 1, 2, 6, 7, 8))
OBJECT_INDICES = jnp.asarray((3, 4, 5, 9, 10, 11))


class SplitRealNVPDiffusionNet(eqx.Module):
    """Independent invertible noise heads for arm and object coordinates."""

    arm: RealNVPDiffusionNet
    object: RealNVPDiffusionNet
    feature_dim: int = eqx.field(static=True)
    d_x: int = eqx.field(static=True, default=12)

    def __init__(self, control_dim, feature_dim, hidden, layers, time_dim, key,
                 alpha):
        arm_key, object_key = jr.split(key)
        context_dim = control_dim + 6 + feature_dim
        self.arm = RealNVPDiffusionNet(
            6, context_dim, hidden, layers, time_dim, arm_key, alpha=alpha
        )
        self.object = RealNVPDiffusionNet(
            6, context_dim, hidden, layers, time_dim, object_key, alpha=alpha
        )
        self.feature_dim = feature_dim

    def _parts(self, state, context):
        control = context[:-self.feature_dim] if self.feature_dim else context
        features = context[-self.feature_dim:] if self.feature_dim else context[:0]
        arm_state, object_state = state[ARM_INDICES], state[OBJECT_INDICES]
        return (arm_state, object_state,
                jnp.concatenate((control, object_state, features)),
                jnp.concatenate((control, arm_state, features)))

    @staticmethod
    def _merge(arm, object):
        out = jnp.zeros((12,), dtype=arm.dtype)
        return out.at[ARM_INDICES].set(arm).at[OBJECT_INDICES].set(object)

    def apply(self, eps, state, context, time):
        arm_state, object_state, arm_context, object_context = self._parts(
            state, context
        )
        return self._merge(
            self.arm.apply(eps[ARM_INDICES], arm_state, arm_context, time),
            self.object.apply(
                eps[OBJECT_INDICES], object_state, object_context, time
            ),
        )

    def inverse_and_log_det(self, value, state, context, time):
        arm_state, object_state, arm_context, object_context = self._parts(
            state, context
        )
        arm, arm_logdet = self.arm.inverse_and_log_det(
            value[ARM_INDICES], arm_state, arm_context, time
        )
        obj, obj_logdet = self.object.inverse_and_log_det(
            value[OBJECT_INDICES], object_state, object_context, time
        )
        return self._merge(arm, obj), arm_logdet + obj_logdet


class ControlledRealNVPFlowSDE(eqx.Module):
    """Euler--Maruyama controlled FlowSDE with exact transition density."""

    drift_full: SDEDriftNet | None
    drift_arm: SDEDriftNet | None
    drift_object: SDEDriftNet | None
    diffusion: RealNVPDiffusionNet | SplitRealNVPDiffusionNet
    state_dim: int = eqx.field(static=True)
    control_dim: int = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    diffusion_condition_on_control: bool = eqx.field(static=True)
    object_drift_no_control: bool = eqx.field(static=True)
    separate_dynamics_heads: bool = eqx.field(static=True)
    coupled_six_state_flows: bool = eqx.field(static=True)
    use_contact_geometry: bool = eqx.field(static=True)
    geometry_feature_fn: Callable | None = eqx.field(static=True)
    geometry_temperature: float = eqx.field(static=True)
    state_mean: jax.Array
    state_std: jax.Array
    geometry_mean: jax.Array
    geometry_std: jax.Array

    def __init__(self, state_dim, control_dim, key, *, hidden_size=64, depth=2,
                 diffusion_hidden=64, flow_layers=4, time_emb_dim=16,
                 realnvp_alpha=3.0, dt=0.02,
                 diffusion_condition_on_control=True,
                 object_drift_no_control=False, separate_dynamics_heads=False,
                 coupled_six_state_flows=False,
                 use_contact_geometry=False, geometry_temperature=0.02,
                 geometry_feature_fn=None,
                 state_mean=None, state_std=None, geometry_mean=None,
                 geometry_std=None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.dt = float(dt)
        self.diffusion_condition_on_control = bool(diffusion_condition_on_control)
        self.object_drift_no_control = bool(object_drift_no_control)
        self.separate_dynamics_heads = bool(separate_dynamics_heads)
        self.coupled_six_state_flows = bool(coupled_six_state_flows)
        self.use_contact_geometry = bool(use_contact_geometry)
        self.geometry_feature_fn = geometry_feature_fn
        self.geometry_temperature = float(geometry_temperature)
        self.state_mean = jnp.zeros(state_dim) if state_mean is None else jnp.asarray(state_mean)
        self.state_std = jnp.ones(state_dim) if state_std is None else jnp.asarray(state_std)
        self.geometry_mean = jnp.zeros(5) if geometry_mean is None else jnp.asarray(geometry_mean)
        self.geometry_std = jnp.ones(5) if geometry_std is None else jnp.asarray(geometry_std)
        if (self.separate_dynamics_heads or self.coupled_six_state_flows
                or self.use_contact_geometry) and state_dim != 12:
            raise ValueError("split heads/contact geometry require the 12-D pusher state")
        if self.coupled_six_state_flows and self.use_contact_geometry:
            raise ValueError("coupled six-state flows use explicit cross-state coupling")
        if self.use_contact_geometry and self.geometry_feature_fn is None:
            raise ValueError("use_contact_geometry requires geometry_feature_fn")
        drift_key, diffusion_key = jr.split(key)
        feature_dim = 5 if self.use_contact_geometry else 0
        if self.coupled_six_state_flows:
            arm_key, object_key = jr.split(drift_key)
            self.drift_full = None
            # Robot flow: r_t plus (u_t, o_t). Object flow: o_t plus
            # (u_t, r_t, predicted robot drift). Both remain jointly trained.
            self.drift_arm = SDEDriftNet(
                6, control_dim + 6, hidden_size, depth, time_emb_dim,
                arm_key, output_dim=6,
            )
            self.drift_object = SDEDriftNet(
                6, control_dim + 12, hidden_size, depth, time_emb_dim,
                object_key, output_dim=6,
            )
        elif self.object_drift_no_control or self.separate_dynamics_heads:
            if state_dim != 12:
                raise ValueError(
                    "object_drift_no_control requires the 12-D pusher state"
                )
            arm_key, object_key = jr.split(drift_key)
            self.drift_full = None
            self.drift_arm = SDEDriftNet(
                state_dim + feature_dim, control_dim, hidden_size, depth, time_emb_dim,
                arm_key, output_dim=6,
            )
            self.drift_object = SDEDriftNet(
                state_dim + feature_dim,
                0 if self.object_drift_no_control else control_dim,
                hidden_size, depth, time_emb_dim,
                object_key, output_dim=6,
            )
        else:
            self.drift_full = SDEDriftNet(
                state_dim + feature_dim, control_dim, hidden_size, depth, time_emb_dim,
                drift_key,
            )
            self.drift_arm = None
            self.drift_object = None
        diffusion_control_dim = control_dim if self.diffusion_condition_on_control else 0
        if self.separate_dynamics_heads or self.coupled_six_state_flows:
            self.diffusion = SplitRealNVPDiffusionNet(
                diffusion_control_dim, feature_dim, diffusion_hidden,
                flow_layers, time_emb_dim, diffusion_key, realnvp_alpha,
            )
        else:
            self.diffusion = RealNVPDiffusionNet(
                state_dim, diffusion_control_dim + feature_dim,
                diffusion_hidden, flow_layers,
                time_emb_dim, diffusion_key, alpha=realnvp_alpha,
            )

    def geometry_features(self, state):
        physical = state * self.state_std + self.state_mean
        raw = self.geometry_feature_fn(
            physical, temperature=self.geometry_temperature
        )
        return (raw - self.geometry_mean) / self.geometry_std

    def _augmented_state(self, state):
        if not self.use_contact_geometry:
            return state
        return jnp.concatenate((state, self.geometry_features(state)))

    def drift(self, state, control, time):
        augmented = self._augmented_state(state)
        if self.coupled_six_state_flows:
            robot_state = state[ARM_INDICES]
            object_state = state[OBJECT_INDICES]
            robot = self.drift_arm(
                robot_state, jnp.concatenate((control, object_state)), time
            )
            obj = self.drift_object(
                object_state,
                jnp.concatenate((control, robot_state, robot)), time,
            )
            return jnp.zeros_like(state).at[ARM_INDICES].set(robot).at[
                OBJECT_INDICES
            ].set(obj)
        if not (self.object_drift_no_control or self.separate_dynamics_heads):
            return self.drift_full(augmented, control, time)
        arm = self.drift_arm(augmented, control, time)
        object_control = (jnp.empty((0,), dtype=control.dtype)
                          if self.object_drift_no_control else control)
        obj = self.drift_object(augmented, object_control, time)
        # State order is [q_arm, q_object, v_arm, v_object].
        return jnp.concatenate([arm[:3], obj[:3], arm[3:], obj[3:]])

    def _diffusion_context(self, control):
        context = (control if self.diffusion_condition_on_control
                   else jnp.empty((0,), dtype=control.dtype))
        return context

    def _diffusion_context_for_state(self, state, control):
        context = self._diffusion_context(control)
        if self.use_contact_geometry:
            context = jnp.concatenate((context, self.geometry_features(state)))
        return context

    def transition_nll(self, x0, u0, x1, time):
        """Mean exact Euler transition NLL for arrays ending in state_dim."""
        dt = jnp.asarray(self.dt, dtype=x0.dtype)
        def one(a, control, b, t):
            residual = (b - a - self.drift(a, control, t) * dt) / jnp.sqrt(dt)
            diffusion_context = self._diffusion_context_for_state(a, control)
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
            diffusion_context = self._diffusion_context_for_state(x, u)
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

    def coupled_multiple_shooting_losses(
            self, states, controls, start_steps, key, *, segment_length=10,
            n_particles=2, solver_name="heun", shooting_noise_std=0.01,
            control_noise_std=0.01, path_weight=1.0, object_path_weight=1.0,
            uncertainty_weight=0.5,
            continuity_weight=0.1, object_temporal_weights=None):
        """Robot/object energy scores from one shared stochastic rollout."""
        if self.state_dim != 12:
            raise ValueError("coupled losses require the 12-D robot/object state")
        window_steps = controls.shape[1]
        if window_steps % segment_length:
            raise ValueError("window steps must be divisible by segment length")
        n_segments = window_steps // segment_length
        if object_temporal_weights is None:
            object_temporal_weights = jnp.ones(
                controls.shape[:2], dtype=states.dtype
            )
        object_temporal_weights = (
            object_temporal_weights
            / (jnp.mean(object_temporal_weights, axis=1, keepdims=True) + 1e-6)
        )
        batch_keys = jr.split(key, states.shape[0])

        def trajectory_losses(x, u, first_step, trajectory_key, object_weights):
            control_key, shooting_key = jr.split(trajectory_key)
            if control_noise_std > 0:
                u = u + control_noise_std * jr.normal(control_key, u.shape)
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
            object_weights = object_weights.reshape(n_segments, segment_length)

            def segment_losses(x0, segment_controls, target, segment_start,
                               object_time_weights, segment_key):
                init_key, path_key = jr.split(segment_key)
                if shooting_noise_std > 0:
                    x0 = x0 + shooting_noise_std * jr.normal(init_key, x0.shape)
                paths = self.sample_rollout(
                    x0, segment_controls, path_key, n_particles,
                    segment_start, 1.0, solver_name,
                )

                def group_score(indices, time_weights, group_path_weight):
                    group_paths = paths[..., indices]
                    group_target = target[..., indices]
                    accuracy_by_time = jnp.mean(jnp.sqrt(
                        jnp.sum((group_paths - group_target[None]) ** 2,
                                axis=-1) + 1e-6
                    ), axis=0)
                    pairwise = group_paths[:, None] - group_paths[None, :]
                    spread_by_time = jnp.mean(jnp.sqrt(
                        jnp.sum(pairwise ** 2, axis=-1) + 1e-6
                    ), axis=(0, 1))
                    score = jnp.mean(time_weights * (
                        accuracy_by_time - uncertainty_weight * spread_by_time
                    ))
                    continuity = jnp.mean(
                        (jnp.mean(group_paths[:, -1], axis=0)
                         - group_target[-1]) ** 2
                    )
                    return group_path_weight * score + continuity_weight * continuity

                return (
                    group_score(ARM_INDICES, jnp.ones_like(object_time_weights),
                                path_weight),
                    group_score(OBJECT_INDICES, object_time_weights,
                                object_path_weight),
                )

            robot, obj = jax.vmap(segment_losses)(
                state_starts, control_segments, targets, segment_starts,
                object_weights, jr.split(shooting_key, n_segments),
            )
            return jnp.mean(robot), jnp.mean(obj)

        robot, obj = jax.vmap(trajectory_losses)(
            states, controls, start_steps, batch_keys, object_temporal_weights
        )
        return jnp.mean(robot), jnp.mean(obj)

    def multiple_shooting_loss(self, states, controls, start_steps, key, *,
                               segment_length=16, n_particles=4,
                               solver_name="heun", shooting_noise_std=0.01,
                               control_noise_std=0.01,
                               path_weight=1.0,
                               uncertainty_weight=0.5,
                               continuity_weight=0.1,
                               state_loss_weights=None,
                               temporal_loss_weights=None,
                               contact_weight=1.0,
                               contact_center=0.5,
                               contact_width=0.15):
        """Pathwise energy score over stochastic multiple-shooting segments.

        ``states`` has shape (B, W+1, dx), ``controls`` (B, W, du), and W
        must be divisible by ``segment_length``. Each segment is integrated by
        the selected stochastic solver instead of evaluating one-step NLL.
        Optional ``temporal_loss_weights`` has shape (B, W) and is loss-only
        supervision; it is not passed to drift or diffusion networks.
        """
        window_steps = controls.shape[1]
        if window_steps % segment_length:
            raise ValueError("window steps must be divisible by segment length")
        n_segments = window_steps // segment_length
        weights = (
            jnp.ones((self.state_dim,), dtype=states.dtype)
            if state_loss_weights is None
            else jnp.asarray(state_loss_weights, dtype=states.dtype)
        )
        batch_keys = jr.split(key, states.shape[0])
        supplied_temporal_weights = temporal_loss_weights is not None
        if temporal_loss_weights is None:
            temporal_loss_weights = jnp.ones(
                controls.shape[:2], dtype=states.dtype
            )

        def trajectory_loss(x, u, first_step, trajectory_key,
                            trajectory_temporal_weights):
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
            relative_time = (
                jnp.arange(window_steps, dtype=states.dtype) + 1
            ) / window_steps
            temporal_weights = 1.0 + (contact_weight - 1.0) * jnp.exp(
                -0.5 * ((relative_time - contact_center) / contact_width) ** 2
            )
            if supplied_temporal_weights:
                # Privileged labels may shape the training loss, but are never
                # part of the learned dynamics interface or inference inputs.
                temporal_weights *= trajectory_temporal_weights
            temporal_weights /= jnp.mean(temporal_weights)
            temporal_weights = temporal_weights.reshape(
                n_segments, segment_length
            )

            def segment_loss(x0, segment_controls, target, segment_start,
                             time_weights, seg_key):
                init_key, path_key = jr.split(seg_key)
                if shooting_noise_std > 0:
                    x0 = x0 + shooting_noise_std * jr.normal(init_key, x0.shape)
                paths = self.sample_rollout(
                    x0, segment_controls, path_key, n_particles,
                    segment_start, 1.0, solver_name,
                )
                accuracy_by_time = jnp.mean(jnp.sqrt(
                    jnp.sum(
                        weights * (paths - target[None]) ** 2, axis=-1
                    ) + 1e-6
                ), axis=0)
                pairwise = paths[:, None] - paths[None, :]
                spread_by_time = jnp.mean(jnp.sqrt(
                    jnp.sum(weights * pairwise ** 2, axis=-1) + 1e-6
                ), axis=(0, 1))
                accuracy = jnp.mean(time_weights * accuracy_by_time)
                spread = jnp.mean(time_weights * spread_by_time)
                continuity = jnp.mean(
                    weights * (
                        jnp.mean(paths[:, -1], axis=0) - target[-1]
                    ) ** 2
                )
                return accuracy - uncertainty_weight * spread, continuity

            scores, continuities = jax.vmap(segment_loss)(
                state_starts, control_segments, targets, segment_starts,
                temporal_weights, segment_keys,
            )
            return (
                path_weight * jnp.mean(scores)
                + continuity_weight * jnp.mean(continuities)
            )

        return jnp.mean(jax.vmap(trajectory_loss)(
            states, controls, start_steps, batch_keys, temporal_loss_weights
        ))
