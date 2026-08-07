"""Native JAX port of DPPO's state-conditioned DDPM policy."""
from __future__ import annotations

from typing import NamedTuple
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class DiffusionSample(NamedTuple):
    actions: jnp.ndarray
    chain: jnp.ndarray


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> jnp.ndarray:
    """Exact schedule used by ``third_party/dppo``."""
    count = steps + 1
    x = jnp.linspace(0.0, float(count), count)
    alpha_bar = jnp.cos(((x / count) + offset) / (1 + offset) * jnp.pi * .5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    return jnp.clip(1.0 - alpha_bar[1:] / alpha_bar[:-1], 0.0, 0.999)


def _time_embedding(t, dim):
    half = dim // 2
    frequency = jnp.exp(
        jnp.arange(half) * (-math.log(10_000.0) / (half - 1))
    )
    return jnp.concatenate([jnp.sin(t * frequency), jnp.cos(t * frequency)])


class _ResidualBlock(eqx.Module):
    first: eqx.nn.Linear
    second: eqx.nn.Linear

    def __init__(self, width, key):
        k1, k2 = jr.split(key)
        self.first = eqx.nn.Linear(width, width, key=k1)
        self.second = eqx.nn.Linear(width, width, key=k2)

    def __call__(self, x):
        residual = x
        x = self.first(jax.nn.mish(x))
        x = self.second(jax.nn.mish(x))
        return x + residual


class DPPODiffusionMLP(eqx.Module):
    """JAX equivalent of official ``DiffusionMLP(... residual_style=True)``."""

    condition_first: eqx.nn.Linear
    condition_second: eqx.nn.Linear
    time_first: eqx.nn.Linear
    time_second: eqx.nn.Linear
    input_layer: eqx.nn.Linear
    residual: _ResidualBlock
    output_layer: eqx.nn.Linear
    horizon: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    time_dim: int = eqx.field(static=True)

    def __init__(self, state_size, horizon, action_dim, key, time_dim=16):
        keys = jr.split(key, 9)
        self.condition_first = eqx.nn.Linear(state_size, 128, key=keys[0])
        self.condition_second = eqx.nn.Linear(128, 32, key=keys[1])
        self.time_first = eqx.nn.Linear(time_dim, 2 * time_dim, key=keys[2])
        self.time_second = eqx.nn.Linear(2 * time_dim, time_dim, key=keys[3])
        input_size = horizon * action_dim + 32 + time_dim
        self.input_layer = eqx.nn.Linear(input_size, 256, key=keys[4])
        self.residual = _ResidualBlock(256, keys[5])
        self.output_layer = eqx.nn.Linear(256, horizon * action_dim, key=keys[6])
        self.horizon, self.action_dim, self.time_dim = horizon, action_dim, time_dim

    def __call__(self, noisy_action, t, state):
        condition = self.condition_second(
            jax.nn.mish(self.condition_first(state.reshape(-1)))
        )
        time = self.time_second(
            jax.nn.mish(self.time_first(_time_embedding(t, self.time_dim)))
        )
        x = jnp.concatenate([noisy_action.reshape(-1), time, condition])
        x = self.input_layer(x)
        x = self.residual(x)
        return self.output_layer(x).reshape(self.horizon, self.action_dim)


class DPPODiffusionPolicy(eqx.Module):
    """Official DPPO DDPM parameterization with a JAX training interface."""

    network: DPPODiffusionMLP
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alphas_cumprod: jnp.ndarray
    alphas_cumprod_prev: jnp.ndarray
    sqrt_alphas_cumprod: jnp.ndarray
    sqrt_one_minus_alphas_cumprod: jnp.ndarray
    sqrt_recip_alphas_cumprod: jnp.ndarray
    sqrt_recipm1_alphas_cumprod: jnp.ndarray
    posterior_coef1: jnp.ndarray
    posterior_coef2: jnp.ndarray
    posterior_log_variance: jnp.ndarray
    obs_dim: int = eqx.field(static=True)
    cond_steps: int = eqx.field(static=True)
    action_dim: int = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    denoising_steps: int = eqx.field(static=True)
    denoised_clip: float = eqx.field(static=True)
    sampling_std_floor: float = eqx.field(static=True)
    logprob_std_floor: float = eqx.field(static=True)

    def __init__(
        self, obs_dim, action_dim, horizon, key, *, cond_steps=1,
        denoising_steps=20, time_dim=16, denoised_clip=1.0,
        sampling_std_floor=0.1, logprob_std_floor=0.1,
    ):
        self.network = DPPODiffusionMLP(
            obs_dim * cond_steps, horizon, action_dim, key, time_dim
        )
        self.betas = cosine_beta_schedule(denoising_steps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.alphas_cumprod_prev = jnp.concatenate(
            [jnp.ones(1), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(
            1 / self.alphas_cumprod - 1
        )
        variance = self.betas * (1 - self.alphas_cumprod_prev) / (
            1 - self.alphas_cumprod
        )
        self.posterior_log_variance = jnp.log(jnp.maximum(variance, 1e-20))
        self.posterior_coef1 = (
            self.betas * jnp.sqrt(self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
        )
        self.posterior_coef2 = (
            (1 - self.alphas_cumprod_prev) * jnp.sqrt(self.alphas)
            / (1 - self.alphas_cumprod)
        )
        self.obs_dim, self.cond_steps = obs_dim, cond_steps
        self.action_dim, self.horizon = action_dim, horizon
        self.denoising_steps = denoising_steps
        self.denoised_clip = denoised_clip
        self.sampling_std_floor = sampling_std_floor
        self.logprob_std_floor = logprob_std_floor

    @property
    def integration_steps(self):
        return self.denoising_steps

    @property
    def nfe(self):
        return self.denoising_steps

    def _check_states(self, states):
        if states.ndim != 3 or states.shape[1:] != (
            self.cond_steps, self.obs_dim
        ):
            raise ValueError(
                f"states must have shape (batch, {self.cond_steps}, {self.obs_dim})"
            )

    def loss(self, actions, states, key):
        self._check_states(states)
        kt, kn = jr.split(key)
        times = jr.randint(kt, (len(actions),), 0, self.denoising_steps)
        noise = jr.normal(kn, actions.shape)
        alpha = self.sqrt_alphas_cumprod[times, None, None]
        sigma = self.sqrt_one_minus_alphas_cumprod[times, None, None]
        noisy = alpha * actions + sigma * noise
        prediction = jax.vmap(self.network)(noisy, times, states)
        return jnp.mean((prediction - noise) ** 2)

    def _mean_log_variance(self, x, time, state):
        predicted_noise = self.network(x, time, state)
        reconstructed = (
            self.sqrt_recip_alphas_cumprod[time] * x
            - self.sqrt_recipm1_alphas_cumprod[time] * predicted_noise
        )
        reconstructed = jnp.clip(
            reconstructed, -self.denoised_clip, self.denoised_clip
        )
        mean = (
            self.posterior_coef1[time] * reconstructed
            + self.posterior_coef2[time] * x
        )
        return mean, self.posterior_log_variance[time]

    def sample(self, states, key, *, deterministic=False, clip=1.0):
        self._check_states(states)
        batch = len(states)
        key, initial_key = jr.split(key)
        initial = jr.normal(
            initial_key, (batch, self.horizon, self.action_dim)
        )
        keys = jr.split(key, self.denoising_steps)
        times = jnp.arange(self.denoising_steps - 1, -1, -1)

        def step(x, inputs):
            time, noise_key = inputs
            time_batch = jnp.full((batch,), time)
            mean, log_variance = jax.vmap(self._mean_log_variance)(
                x, time_batch, states
            )
            std = jnp.exp(.5 * log_variance)[:, None, None]
            if deterministic:
                std = jnp.where(time == 0, 0.0, jnp.maximum(std, 1e-3))
            else:
                std = jnp.maximum(std, self.sampling_std_floor)
            following = mean + std * jr.normal(noise_key, x.shape)
            return following, following

        final, generated = jax.lax.scan(step, initial, (times, keys))
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return DiffusionSample(actions, chain)

    def sample_ddim(self, states, key, *, inference_steps=None, clip=1.0):
        """Deterministic DDIM sampling on a strided subset of the trained grid.

        This is intended for behavior-cloning evaluation, not PPO: strided
        DDIM transitions do not use the Markov-chain likelihood consumed by
        :meth:`transition_log_prob`.
        """
        self._check_states(states)
        steps = self.denoising_steps if inference_steps is None else int(inference_steps)
        if not 1 <= steps <= self.denoising_steps:
            raise ValueError(
                f"inference_steps must be in [1, {self.denoising_steps}]"
            )
        batch = len(states)
        initial = jr.normal(key, (batch, self.horizon, self.action_dim))
        # Integer linspace includes both endpoints and is strictly descending
        # when steps <= the length of the trained diffusion grid.
        times = jnp.linspace(
            self.denoising_steps - 1, 0, steps, dtype=jnp.int32
        )
        previous_times = jnp.concatenate([times[1:], -jnp.ones((1,), jnp.int32)])

        def step(x, pair):
            time, previous_time = pair
            time_batch = jnp.full((batch,), time)
            predicted_noise = jax.vmap(self.network)(x, time_batch, states)
            alpha = self.alphas_cumprod[time]
            alpha_previous = jnp.where(
                previous_time >= 0,
                self.alphas_cumprod[jnp.maximum(previous_time, 0)],
                1.0,
            )
            reconstructed = (
                x - jnp.sqrt(1.0 - alpha) * predicted_noise
            ) / jnp.sqrt(alpha)
            reconstructed = jnp.clip(
                reconstructed, -self.denoised_clip, self.denoised_clip
            )
            following = (
                jnp.sqrt(alpha_previous) * reconstructed
                + jnp.sqrt(1.0 - alpha_previous) * predicted_noise
            )
            return following, following

        final, generated = jax.lax.scan(step, initial, (times, previous_times))
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return DiffusionSample(actions, chain)

    def sample_hybrid(
        self, states, key, base_policy, ft_denoising_steps, *,
        deterministic=False, clip=1.0,
    ):
        """Sample with a frozen base for early and RL actor for late steps."""
        self._check_states(states)
        batch = len(states)
        key, initial_key = jr.split(key)
        initial = jr.normal(initial_key, (batch, self.horizon, self.action_dim))
        keys = jr.split(key, self.denoising_steps)
        times = jnp.arange(self.denoising_steps - 1, -1, -1)

        def step(x, inputs):
            time, noise_key = inputs
            time_batch = jnp.full((batch,), time)
            ft_mean, ft_log_variance = jax.vmap(self._mean_log_variance)(
                x, time_batch, states
            )
            base_mean, base_log_variance = jax.vmap(base_policy._mean_log_variance)(
                x, time_batch, states
            )
            use_finetuned = time < ft_denoising_steps
            mean = jnp.where(use_finetuned, ft_mean, base_mean)
            log_variance = jnp.where(
                use_finetuned, ft_log_variance, base_log_variance
            )
            std = jnp.exp(.5 * log_variance)[:, None, None]
            if deterministic:
                std = jnp.where(time == 0, 0.0, jnp.maximum(std, 1e-3))
            else:
                std = jnp.maximum(std, self.sampling_std_floor)
            following = mean + std * jr.normal(noise_key, x.shape)
            return following, following

        final, generated = jax.lax.scan(step, initial, (times, keys))
        chain = jnp.concatenate(
            [initial[:, None], jnp.swapaxes(generated, 0, 1)], axis=1
        )
        actions = final if clip is None else jnp.clip(final, -clip, clip)
        return DiffusionSample(actions, chain)

    def hybrid_transition_log_prob(
        self, states, chain, base_policy, ft_denoising_steps,
    ):
        """Likelihood for only the late transitions controlled by the actor."""
        first = self.denoising_steps - ft_denoising_steps
        indices = jnp.arange(first, self.denoising_steps)
        return self.transition_log_prob(states, chain, indices)

    def transition_log_prob(self, states, chain, indices=None):
        self._check_states(states)
        if indices is None:
            indices = jnp.arange(self.denoising_steps)
        indices = jnp.asarray(indices)
        previous, following = chain[:, indices], chain[:, indices + 1]
        times = self.denoising_steps - 1 - indices

        def batch_log_prob(prevs, nexts, state):
            def transition(prev, nxt, time):
                mean, log_variance = self._mean_log_variance(prev, time, state)
                std = jnp.maximum(
                    jnp.exp(.5 * log_variance), self.logprob_std_floor
                )
                return -.5 * (
                    ((nxt - mean) / std) ** 2
                    + 2 * jnp.log(std) + jnp.log(2 * jnp.pi)
                )
            return jax.vmap(transition)(prevs, nexts, times)
        return jax.vmap(batch_log_prob)(previous, following, states)
