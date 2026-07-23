"""Conditional stochastic-interpolant policy for action chunks.

This optimal-control policy deliberately reuses the stochastic-interpolant
building blocks from state estimation:

* :class:`ObsTransformerEncoder` encodes the state history into one context
  vector per future action step;
* :class:`StateInferenceMLP` parameterizes the informative Gaussian source;
* :class:`SITransformerNet` is the conditional velocity field.

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

    obs_dim: int = eqx.field(static=True)
    cond_steps: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    integration_steps: int = eqx.field(static=True)
    context_dim: int = eqx.field(static=True)
    sigma_0: float = eqx.field(static=True)
    sampling_std: float = eqx.field(static=True)
    informative_base: bool = eqx.field(static=True)

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
        hidden_size: int = 256,
        depth: int = 3,
        time_dim: int = 16,
        context_dim: int = 32,
        n_heads: int = 4,
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
        ) < 1:
            raise ValueError("dimensions and integration_steps must be positive")
        if hidden_size % n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if sigma_0 < 0 or sampling_std < 0:
            raise ValueError("sigma_0 and sampling_std must be nonnegative")

        k_encoder, k_source, k_velocity = jr.split(key, 3)
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
        self.obs_dim, self.cond_steps = obs_dim, cond_steps
        self.action_dim, self.horizon = action_dim, horizon
        self.integration_steps, self.context_dim = integration_steps, context_dim
        self.sigma_0, self.sampling_std = float(sigma_0), float(sampling_std)
        self.informative_base = informative_base

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

    def loss(
        self, actions: jnp.ndarray, states: jnp.ndarray, key: jr.PRNGKey
    ) -> jnp.ndarray:
        """Standard stochastic-interpolant velocity-matching loss."""
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
            return jnp.mean((prediction - target) ** 2)

        return jnp.mean(jax.vmap(example_loss)(actions, states, keys))

    def sample(
        self,
        states: jnp.ndarray,
        key: jr.PRNGKey,
        *,
        deterministic: bool = False,
        clip: float | None = 1.0,
    ) -> InterpolantSample:
        """Generate one action chunk per state and retain the inner chain."""
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
        if self.sampling_std <= 0:
            raise ValueError("transition likelihood requires sampling_std > 0")
        if indices is None:
            indices = jnp.arange(self.integration_steps)
        indices = jnp.asarray(indices)
        previous, following = chain[:, indices], chain[:, indices + 1]
        contexts, means, _ = jax.vmap(self._features)(states)
        dt = 1.0 / self.integration_steps
        times = (indices.astype(chain.dtype) + 0.5) * dt

        def per_transition(prev, nxt, t, source_mean, context):
            transition_mean = prev + dt * self.vel_net(
                prev, source_mean, context, t
            )
            std = self.sampling_std * jnp.sqrt(dt)
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
