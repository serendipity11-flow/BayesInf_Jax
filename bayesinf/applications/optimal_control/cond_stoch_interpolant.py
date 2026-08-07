"""Conditional stochastic-interpolant policy for action chunks.

This optimal-control policy deliberately reuses the stochastic-interpolant
building blocks from state estimation:

* :class:`ObsTransformerEncoder` encodes the state history into one context
  vector per future action step;
* :class:`StateInferenceMLP` parameterizes the informative Gaussian source;
* :class:`SITransformerNet` parameterizes the conditional velocity field;
* with a fixed Gaussian source, the score is recovered analytically from that
  velocity and the interpolant schedule, so no score network is trained.

The offline interpolant is

    I_t = (1-t) x_0 + t x_1 + sigma_0 t(1-t) z,

where ``x_1`` is an expert action chunk and ``x_0 ~ q(x_0 | state)``.  At
inference, Euler--Maruyama transitions transport that source to an action
chunk.  Their explicit Gaussian densities support DPPO's inner-MDP PPO loss.
"""
from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.stochastic_interpolant import (
    ObsTransformerEncoder,
    SITransformerNet,
    StateInferenceMLP,
)


class UnconditionalScoreCorrection(eqx.Module):
    """Small score correction depending only on noisy actions and time."""

    first: eqx.nn.Linear
    second: eqx.nn.Linear
    output: eqx.nn.Linear
    horizon: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)

    def __init__(self, horizon, action_dim, width, key):
        k1, k2, k3 = jr.split(key, 3)
        size = horizon * action_dim
        self.first = eqx.nn.Linear(size + 1, width, key=k1)
        self.second = eqx.nn.Linear(width, width, key=k2)
        self.output = eqx.nn.Linear(width, size, key=k3)
        self.horizon, self.action_dim = horizon, action_dim

    def __call__(self, noisy_action, t):
        value = jnp.concatenate((noisy_action.reshape(-1), jnp.asarray(t).reshape(1)))
        value = jax.nn.mish(self.first(value))
        value = jax.nn.mish(self.second(value))
        return self.output(value).reshape(self.horizon, self.action_dim)

# Backward-compatible public names.  They now point at the shared
# state-estimation implementations instead of maintaining duplicate networks.
ConditionalGaussianBase = StateInferenceMLP
ConditionalVelocity = SITransformerNet


class InterpolantSample(NamedTuple):
    """Generated action chunks and their complete inner-MDP chain."""

    actions: jnp.ndarray
    chain: jnp.ndarray


class CondStochInterpolantPolicy(eqx.Module):
    """State-conditioned stochastic-interpolant action policy.

    States have shape ``(batch, cond_steps, obs_dim)`` and action chunks have
    shape ``(batch, horizon, action_dim)``.  The state history is flattened and
    repeated over the future horizon before shared transformer encoding.  The
    encoder's positional embedding makes the resulting future-step contexts
    distinct even though every step observes the same state history.
    """

    encoder: ObsTransformerEncoder
    inference_net: StateInferenceMLP | None
    vel_net: SITransformerNet
    score_net: UnconditionalScoreCorrection | None

    obs_dim: int = eqx.field(static=True)
    cond_steps: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    integration_steps: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    sigma_0: float = eqx.field(static=True)
    sampling_std: float = eqx.field(static=True)
    score_sigma_floor: float = eqx.field(static=True)
    sde_eta: float = eqx.field(static=True)
    path_loss_weight: float = eqx.field(static=True)
    path_loss_steps: int = eqx.field(static=True)
    informative_base: bool = eqx.field(static=True)
    score_loss_weight: float = eqx.field(static=True)
    score_correction_scale: float = eqx.field(static=True)
    sampler_matched_sde: bool = eqx.field(static=True)

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        key: jr.PRNGKey,
        *,
        cond_steps: int = 1,
        integration_steps: int = 5,
        informative_base: bool = True,
        sigma_0: float = 0.1,
        sampling_std: float = 0.1,
        score_sigma_floor: float = 0.02,
        sde_eta: float = 0.05,
        path_loss_weight: float = 0.0,
        path_loss_steps: int = 10,
        hidden_size: int = 256,
        depth: int = 3,
        time_dim: int = 16,
        context_dim: int = 32,
        n_heads: int = 4,
        score_head_width: int = 0,
        score_loss_weight: float = 0.0,
        score_correction_scale: float = 0.1,
        sampler_matched_sde: bool = False,
    ):
        if min(
            obs_dim,
            action_dim,
            horizon,
            cond_steps,
            integration_steps,
            hidden_size,
            depth,
            time_dim,
            context_dim,
            n_heads,
            path_loss_steps,
        ) < 1:
            raise ValueError("dimensions and integration_steps must be positive")
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if min(
            sigma_0, sampling_std, score_sigma_floor, sde_eta,
            path_loss_weight,
        ) < 0:
            raise ValueError("noise, score, and SDE parameters must be nonnegative")
        if score_sigma_floor == 0:
            raise ValueError("score_sigma_floor must be positive")

        k_encoder, k_source, k_velocity, k_score = jr.split(key, 4)
        state_size = obs_dim * cond_steps
        self.encoder = ObsTransformerEncoder(
            d_y=state_size,
            d_c=context_dim,
            d_model=hidden_size,
            n_heads=n_heads,
            depth=depth,
            time_emb_dim=time_dim,
            key=k_encoder,
        )
        self.inference_net = (
            StateInferenceMLP(
                d_y=state_size,
                d_x=action_dim,
                d_c=context_dim,
                d_hidden=hidden_size,
                key=k_source,
            )
            if informative_base
            else None
        )
        self.vel_net = SITransformerNet(
            d_x=action_dim,
            d_c=context_dim,
            d_model=hidden_size,
            n_heads=n_heads,
            depth=depth,
            time_emb_dim=time_dim,
            key=k_velocity,
        )
        self.score_net = (
            UnconditionalScoreCorrection(horizon, action_dim, score_head_width, k_score)
            if score_head_width > 0 else None
        )
        self.obs_dim, self.cond_steps = obs_dim, cond_steps
        self.action_dim, self.horizon = action_dim, horizon
        self.integration_steps, self.context_dim = integration_steps, context_dim
        self.sigma_0, self.sampling_std = float(sigma_0), float(sampling_std)
        self.score_sigma_floor = float(score_sigma_floor)
        self.sde_eta = float(sde_eta)
        self.path_loss_weight = float(path_loss_weight)
        self.path_loss_steps = int(path_loss_steps)
        self.informative_base = informative_base
        self.score_loss_weight = float(score_loss_weight)
        self.score_correction_scale = float(score_correction_scale)
        self.sampler_matched_sde = bool(sampler_matched_sde)

    @property
    def velocity(self) -> SITransformerNet:
        """Backward-compatible alias for the shared velocity network."""
        return self.vel_net

    @property
    def base(self) -> StateInferenceMLP | None:
        """Backward-compatible alias for the shared source network."""
        return self.inference_net

    @property
    def nfe(self) -> int:
        return self.integration_steps

    def _check_states(self, states: jnp.ndarray) -> None:
        if states.ndim != 3 or states.shape[1:] != (
            self.cond_steps,
            self.obs_dim,
        ):
            raise ValueError(
                f"states must have shape (batch, {self.cond_steps}, {self.obs_dim})"
            )

    def _features(
        self, state: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return repeated state input, future context, and source parameters."""
        if state.shape != (self.cond_steps, self.obs_dim):
            raise ValueError(
                f"state must have shape ({self.cond_steps}, {self.obs_dim})"
            )
        state_vector = state.reshape(-1)
        state_sequence = jnp.broadcast_to(
            state_vector, (self.horizon, state_vector.shape[0])
        )
        context = self.encoder(state_sequence)
        if self.inference_net is None:
            shape = (self.horizon, self.action_dim)
            mean, log_std = jnp.zeros(shape), jnp.zeros(shape)
        else:
            mean, log_std = self.inference_net(state_sequence, context)
        return context, mean, log_std

    def base_params(self, state: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return source mean and log standard deviation for one state history."""
        _, mean, log_std = self._features(state)
        return mean, log_std

    @staticmethod
    def _sample_source(
        mean: jnp.ndarray,
        log_std: jnp.ndarray,
        key: jr.PRNGKey,
        deterministic: bool,
    ) -> jnp.ndarray:
        return (
            mean
            if deterministic
            else mean + jnp.exp(log_std) * jr.normal(key, mean.shape)
        )

    def loss_components(
        self, actions: jnp.ndarray, states: jnp.ndarray, key: jr.PRNGKey
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Return total, velocity, and sampler-matched endpoint losses."""
        self._check_states(states)
        if actions.ndim != 3 or actions.shape[1:] != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"actions must have shape (batch, {self.horizon}, {self.action_dim})"
            )
        if len(actions) != len(states):
            raise ValueError("actions and states must have the same batch size")
        keys = jr.split(key, len(actions))

        def example_loss(action, state, sample_key):
            kt, kb, kz = jr.split(sample_key, 3)
            t = jr.uniform(kt, (), minval=1e-4, maxval=1.0 - 1e-4)
            context, mean, log_std = self._features(state)
            x0 = self._sample_source(mean, log_std, kb, False)
            noise = jr.normal(kz, action.shape)
            sigma = self.sigma_0 * t * (1.0 - t)
            sigma_dot = self.sigma_0 * (1.0 - 2.0 * t)
            interpolated = (1.0 - t) * x0 + t * action + sigma * noise
            target = action - x0 + sigma_dot * noise
            prediction = self.vel_net(interpolated, mean, context, t)
            velocity_loss = jnp.mean((prediction - target) ** 2)
            if self.score_net is None or self.score_loss_weight == 0:
                score_loss = jnp.asarray(0.0, velocity_loss.dtype)
            else:
                rho2 = (1.0 - t) ** 2 + sigma ** 2
                score_target = -(interpolated - t * action) / jnp.maximum(
                    rho2, self.score_sigma_floor ** 2
                )
                analytic = self._score_from_velocity(interpolated, prediction, t)
                corrected = analytic + self.score_correction_scale * self.score_net(interpolated, t)
                score_loss = jnp.mean((corrected - score_target) ** 2)
            return velocity_loss, score_loss

        velocity_losses, score_losses = jax.vmap(example_loss)(actions, states, keys)
        velocity_loss, score_loss = jnp.mean(velocity_losses), jnp.mean(score_losses)
        if self.path_loss_weight == 0:
            path_loss = jnp.asarray(0.0, dtype=velocity_loss.dtype)
        else:
            path_keys = jr.split(jr.fold_in(key, 1), len(actions))

            def example_path_loss(action, state, sample_key):
                context, mean, _ = self._features(state)
                k1, k2 = jr.split(sample_key)
                source1 = jr.normal(k1, action.shape)
                source2 = jr.normal(k2, action.shape)
                if self.sampler_matched_sde:
                    k1n, k2n = jr.split(jr.fold_in(sample_key, 17))
                    sample1 = self._sde_euler_single(source1, mean, context, k1n, self.path_loss_steps)
                    sample2 = self._sde_euler_single(source2, mean, context, k2n, self.path_loss_steps)
                else:
                    sample1 = self._ode_heun_single(source1, mean, context, self.path_loss_steps)
                    sample2 = self._ode_heun_single(source2, mean, context, self.path_loss_steps)
                flatten = lambda value: value.reshape(-1)
                error1 = jnp.linalg.norm(flatten(sample1 - action))
                error2 = jnp.linalg.norm(flatten(sample2 - action))
                spread = jnp.linalg.norm(flatten(sample1 - sample2))
                scale = jnp.sqrt(action.size)
                return (0.5 * (error1 + error2) - 0.5 * spread) / scale

            path_loss = jnp.mean(
                jax.vmap(example_path_loss)(actions, states, path_keys)
            )
        total = (velocity_loss + self.path_loss_weight * path_loss
                 + self.score_loss_weight * score_loss)
        return total, velocity_loss, path_loss

    def loss(
        self, actions: jnp.ndarray, states: jnp.ndarray, key: jr.PRNGKey
    ) -> jnp.ndarray:
        """Stochastic-interpolant velocity-matching objective."""
        return self.loss_components(actions, states, key)[0]

    def _corrected_score(self, x, velocity, t):
        score = self._score_from_velocity(x, velocity, t)
        if self.score_net is not None:
            score = score + self.score_correction_scale * self.score_net(x, t)
        return score

    def _sde_euler_single(self, initial, source_mean, context, key, steps):
        """Differentiable sampler identical to deployed Euler--Maruyama SDE."""
        dt = 1.0 / steps
        keys = jr.split(key, steps)
        def step(current, inputs):
            index, noise_key = inputs
            t = (index.astype(current.dtype) + .5) * dt
            velocity = self.vel_net(current, source_mean, context, t)
            score = self._corrected_score(current, velocity, t)
            following = (current + dt * (velocity + self.sde_eta * score)
                         + jnp.sqrt(2.0 * self.sde_eta * dt) * jr.normal(noise_key, current.shape))
            return following, following
        final, _ = jax.lax.scan(step, initial, (jnp.arange(steps), keys))
        return final

    def _ode_heun_single(self, initial, source_mean, context, steps):
        """Differentiable fixed-step Heun integration for one action chunk."""
        dt = 1.0 / steps

        def step(current, index):
            t = index.astype(current.dtype) * dt
            first = self.vel_net(current, source_mean, context, t)
            predictor = current + dt * first
            second = self.vel_net(
                predictor, source_mean, context, jnp.minimum(t + dt, 1.0)
            )
            following = current + 0.5 * dt * (first + second)
            return following, following

        final, generated = jax.lax.scan(step, initial, jnp.arange(steps))
        return final

    def _actual_score(self, x, source_mean, context, t):
        """Analytic score recovered from velocity for a fixed N(0,I) source.

        For ``x_t = beta*x_1 + rho*epsilon``, where
        ``rho=sqrt(alpha**2+gamma**2)``, the velocity/score identity is

        ``score = -(beta_dot*x - beta*velocity) /
                   (rho*(beta_dot*rho-rho_dot*beta))``.

        It is valid only for the fixed Gaussian source used by the Door CSI.
        """
        if self.informative_base:
            raise ValueError("analytic score requires informative_base=False")
        velocity = self.vel_net(x, source_mean, context, t)
        return self._score_from_velocity(x, velocity, t)

    def _score_from_velocity(self, x, velocity, t):
        """Evaluate the analytic identity from an already-computed velocity."""
        alpha = 1.0 - t
        alpha_dot = -1.0
        beta = t
        beta_dot = 1.0
        gamma = self.sigma_0 * t * (1.0 - t)
        gamma_dot = self.sigma_0 * (1.0 - 2.0 * t)
        rho = jnp.sqrt(alpha ** 2 + gamma ** 2)
        rho_dot = (alpha * alpha_dot + gamma * gamma_dot) / jnp.maximum(
            rho, self.score_sigma_floor
        )
        denominator = rho * (beta_dot * rho - rho_dot * beta)
        denominator = jnp.maximum(denominator, self.score_sigma_floor)
        return -(beta_dot * x - beta * velocity) / denominator

    def sample(
        self,
        states: jnp.ndarray,
        key: jr.PRNGKey,
        *,
        deterministic: bool = False,
        clip: float | None = 1.0,
    ) -> InterpolantSample:
        """Generate chunks; fixed-source CSI uses its analytic-score SDE."""
        if not self.informative_base:
            if deterministic:
                return self.sample_ode(states, key, clip=clip)
            return self.sample_sde(states, key, clip=clip)
        self._check_states(states)
        batch = len(states)
        contexts, means, log_stds = jax.vmap(self._features)(states)
        keys = jr.split(key, batch + 1)
        initial = jax.vmap(self._sample_source, in_axes=(0, 0, 0, None))(
            means, log_stds, keys[:-1], deterministic
        )
        dt = 1.0 / self.integration_steps
        step_keys = jr.split(keys[-1], self.integration_steps)

        def step(current, inputs):
            index, noise_key = inputs
            t = (index.astype(current.dtype) + 0.5) * dt
            times = jnp.full((batch,), t)
            drift = jax.vmap(self.vel_net)(current, means, contexts, times)
            mean = current + dt * drift
            if deterministic or self.sampling_std == 0:
                following = mean
            else:
                following = (
                    mean
                    + self.sampling_std
                    * jnp.sqrt(dt)
                    * jr.normal(noise_key, current.shape)
                )
            return following, following

        final, generated = jax.lax.scan(
            step, initial, (jnp.arange(self.integration_steps), step_keys)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_ode(self, states, key, *, clip=1.0):
        """Deterministic probability-flow transport using the velocity head."""
        self._check_states(states)
        batch = len(states)
        contexts, means, log_stds = jax.vmap(self._features)(states)
        keys = jr.split(key, batch + 1)
        initial = jax.vmap(self._sample_source, in_axes=(0, 0, 0, None))(
            means, log_stds, keys[:-1], True
        ) if self.informative_base else jr.normal(
            keys[-1], (batch, self.horizon, self.action_dim)
        )
        dt = 1.0 / self.integration_steps

        def step(current, index):
            t = (index.astype(current.dtype) + 0.5) * dt
            times = jnp.full((batch,), t)
            drift = jax.vmap(self.vel_net)(current, means, contexts, times)
            following = current + dt * drift
            return following, following

        final, generated = jax.lax.scan(
            step, initial, jnp.arange(self.integration_steps)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_ode_heun(self, states, key, *, clip=1.0):
        """Second-order Heun probability-flow sampling."""
        self._check_states(states)
        batch = len(states)
        contexts, means, _ = jax.vmap(self._features)(states)
        initial = jr.normal(key, (batch, self.horizon, self.action_dim))
        dt = 1.0 / self.integration_steps

        def step(current, index):
            t = index.astype(current.dtype) * dt
            times = jnp.full((batch,), t)
            first = jax.vmap(self.vel_net)(current, means, contexts, times)
            predictor = current + dt * first
            following_times = jnp.minimum(times + dt, 1.0)
            second = jax.vmap(self.vel_net)(
                predictor, means, contexts, following_times
            )
            following = current + 0.5 * dt * (first + second)
            return following, following

        final, generated = jax.lax.scan(
            step, initial, jnp.arange(self.integration_steps)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_sde(self, states, key, *, clip=1.0):
        """Euler--Maruyama transport using the analytic score correction."""
        self._check_states(states)
        batch = len(states)
        contexts, means, log_stds = jax.vmap(self._features)(states)
        keys = jr.split(key, batch + 1)
        initial = jax.vmap(self._sample_source, in_axes=(0, 0, 0, None))(
            means, log_stds, keys[:-1], False
        )
        dt = 1.0 / self.integration_steps
        step_keys = jr.split(keys[-1], self.integration_steps)

        def step(current, inputs):
            index, noise_key = inputs
            t = (index.astype(current.dtype) + 0.5) * dt
            times = jnp.full((batch,), t)
            velocity = jax.vmap(self.vel_net)(current, means, contexts, times)
            score = jax.vmap(self._corrected_score)(
                current, velocity, times
            )
            drift = velocity + self.sde_eta * score
            following = (
                current + dt * drift
                + jnp.sqrt(2.0 * self.sde_eta * dt)
                * jr.normal(noise_key, current.shape)
            )
            return following, following

        final, generated = jax.lax.scan(
            step, initial, (jnp.arange(self.integration_steps), step_keys)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def sample_sde_heun(self, states, key, *, clip=1.0):
        """Stochastic Heun sampling with shared additive noise per step."""
        self._check_states(states)
        batch = len(states)
        contexts, means, _ = jax.vmap(self._features)(states)
        initial_key, noise_key = jr.split(key)
        initial = jr.normal(initial_key, (batch, self.horizon, self.action_dim))
        dt = 1.0 / self.integration_steps
        step_keys = jr.split(noise_key, self.integration_steps)

        def drift(current, times):
            velocity = jax.vmap(self.vel_net)(current, means, contexts, times)
            score = jax.vmap(self._corrected_score)(
                current, velocity, times
            )
            return velocity + self.sde_eta * score

        def step(current, inputs):
            index, step_key = inputs
            t = index.astype(current.dtype) * dt
            times = jnp.full((batch,), t)
            noise = (
                jnp.sqrt(2.0 * self.sde_eta * dt)
                * jr.normal(step_key, current.shape)
            )
            first = drift(current, times)
            predictor = current + dt * first + noise
            second = drift(predictor, jnp.minimum(times + dt, 1.0))
            following = current + 0.5 * dt * (first + second) + noise
            return following, following

        final, generated = jax.lax.scan(
            step, initial, (jnp.arange(self.integration_steps), step_keys)
        )
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return InterpolantSample(actions, chain)

    def transition_log_prob(
        self,
        states: jnp.ndarray,
        chain: jnp.ndarray,
        indices: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Elementwise log density of selected inner transitions."""
        self._check_states(states)
        expected = (
            len(states),
            self.integration_steps + 1,
            self.horizon,
            self.action_dim,
        )
        if chain.shape != expected:
            raise ValueError(f"chain must have shape {expected}")
        if self.informative_base and self.sampling_std <= 0:
            raise ValueError("transition likelihood requires sampling_std > 0")
        if not self.informative_base and self.sde_eta <= 0:
            raise ValueError("analytic SDE likelihood requires sde_eta > 0")
        if indices is None:
            indices = jnp.arange(self.integration_steps)
        indices = jnp.asarray(indices)
        previous, following = chain[:, indices], chain[:, indices + 1]
        contexts, means, _ = jax.vmap(self._features)(states)
        dt = 1.0 / self.integration_steps
        times = (indices.astype(chain.dtype) + 0.5) * dt

        def per_transition(prev, nxt, t, source_mean, context):
            velocity = self.vel_net(prev, source_mean, context, t)
            if self.informative_base:
                drift = velocity
                std = self.sampling_std * jnp.sqrt(dt)
            else:
                score = self._corrected_score(prev, velocity, t)
                drift = velocity + self.sde_eta * score
                std = jnp.sqrt(2.0 * self.sde_eta * dt)
            transition_mean = prev + dt * drift
            return -0.5 * (
                ((nxt - transition_mean) / std) ** 2
                + 2.0 * jnp.log(std)
                + jnp.log(2.0 * jnp.pi)
            )

        def per_batch(prevs, nexts, source_mean, context):
            return jax.vmap(
                per_transition, in_axes=(0, 0, 0, None, None)
            )(prevs, nexts, times, source_mean, context)

        return jax.vmap(per_batch)(
            previous, following, means, contexts
        )
