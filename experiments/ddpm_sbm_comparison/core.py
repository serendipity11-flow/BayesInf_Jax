from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr


class FourierTimeEmbedding(eqx.Module):
    frequencies: jax.Array

    def __init__(self, dim: int):
        half = dim // 2
        self.frequencies = jnp.exp(jnp.linspace(0.0, jnp.log(1000.0), half))

    def __call__(self, t: jax.Array) -> jax.Array:
        phase = t * self.frequencies * 2.0 * jnp.pi
        return jnp.concatenate((jnp.sin(phase), jnp.cos(phase)))


class MatchedScoreNetwork(eqx.Module):
    condition: eqx.nn.MLP
    time_embedding: FourierTimeEmbedding
    denoiser: eqx.nn.MLP

    def __init__(self, observation_size: int, parameter_dim: int, width: int, depth: int, key):
        k1, k2 = jr.split(key)
        time_dim = 16
        self.condition = eqx.nn.MLP(observation_size, width, width, depth=2, key=k1)
        self.time_embedding = FourierTimeEmbedding(time_dim)
        self.denoiser = eqx.nn.MLP(
            parameter_dim + width + time_dim,
            parameter_dim,
            width,
            depth=depth,
            activation=jax.nn.silu,
            key=k2,
        )

    def __call__(self, noisy_theta, t, observation):
        condition = self.condition(jnp.ravel(observation))
        features = jnp.concatenate((noisy_theta, self.time_embedding(t), condition))
        return self.denoiser(features)  # epsilon prediction for both methods


@dataclass(frozen=True)
class VPSchedule:
    beta_min: float = 0.1
    beta_max: float = 20.0

    def integrated_beta(self, t):
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t**2

    def alpha_bar(self, t):
        return jnp.exp(-self.integrated_beta(t))

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t


class MatchedDiffusion(eqx.Module):
    network: MatchedScoreNetwork
    schedule: VPSchedule = eqx.field(static=True)
    parameter_dim: int = eqx.field(static=True)
    training_method: str = eqx.field(static=True)
    training_steps: int = eqx.field(static=True)

    def __init__(self, observation_size, parameter_dim, width, depth, key, training_method="sbm", training_steps=1000, schedule=VPSchedule()):
        self.network = MatchedScoreNetwork(observation_size, parameter_dim, width, depth, key)
        self.schedule = schedule
        self.parameter_dim = parameter_dim
        self.training_method = training_method
        self.training_steps = training_steps

    def loss(self, observations, parameters, key):
        kt, ke = jr.split(key)
        if self.training_method == "ddpm":
            indices = jr.randint(kt, (parameters.shape[0],), 1, self.training_steps + 1)
            t = indices / self.training_steps
        else:
            t = jr.uniform(kt, (parameters.shape[0],), minval=1e-4, maxval=1.0)
        epsilon = jr.normal(ke, parameters.shape)
        alpha_bar = self.schedule.alpha_bar(t)[:, None]
        noisy = jnp.sqrt(alpha_bar) * parameters + jnp.sqrt(1.0 - alpha_bar) * epsilon
        prediction = jax.vmap(self.network)(noisy, t, observations)
        return jnp.mean((prediction - epsilon) ** 2)

    def _conditions(self, observations, num_samples):
        return jnp.repeat(observations, num_samples, axis=0)

    @eqx.filter_jit
    def sample_sbm(self, observations, key, num_samples: int, steps: int):
        """Euler--Maruyama reverse VP SDE followed by Tweedie denoising.

        The stochastic integration stops at epsilon=1/steps, the first DDPM
        noise level, rather than querying the score network outside its DDPM
        training grid. Both samplers then use the same deterministic x0 map.
        """
        conditions = self._conditions(observations, num_samples)
        total = conditions.shape[0]
        key, init_key = jr.split(key)
        state = jr.normal(init_key, (total, self.parameter_dim))
        dt = -1.0 / steps

        def body(carry, index):
            state, key = carry
            # Integrate from t=1 to t=1/steps using midpoint coefficients.
            t = 1.0 - (index + 0.5) / steps
            beta = self.schedule.beta(t)
            std = jnp.sqrt(jnp.maximum(1.0 - self.schedule.alpha_bar(t), 1e-8))
            eps = jax.vmap(self.network)(state, jnp.full((total,), t), conditions)
            score = -eps / std
            reverse_drift = -0.5 * beta * state - beta * score
            key, noise_key = jr.split(key)
            noise = jr.normal(noise_key, state.shape)
            return (state + reverse_drift * dt + jnp.sqrt(beta * -dt) * noise, key), None

        (state, _), _ = jax.lax.scan(body, (state, key), jnp.arange(steps - 1))

        t_min = 1.0 / steps
        alpha_bar = self.schedule.alpha_bar(t_min)
        epsilon = jax.vmap(self.network)(state, jnp.full((total,), t_min), conditions)
        state = (state - jnp.sqrt(1.0 - alpha_bar) * epsilon) / jnp.sqrt(alpha_bar)
        return state.reshape(observations.shape[0], num_samples, self.parameter_dim)

    @eqx.filter_jit
    def sample_ddpm(self, observations, key, num_samples: int, steps: int):
        """Ancestral DDPM whose discrete schedule integrates the same VP SDE."""
        conditions = self._conditions(observations, num_samples)
        total = conditions.shape[0]
        times = jnp.arange(1, steps + 1) / steps
        alpha_bars = self.schedule.alpha_bar(times)
        previous = jnp.concatenate((jnp.ones((1,)), alpha_bars[:-1]))
        alphas = alpha_bars / previous
        betas = 1.0 - alphas
        key, init_key = jr.split(key)
        state = jr.normal(init_key, (total, self.parameter_dim))

        def body(carry, reverse_index):
            state, key = carry
            i = steps - 1 - reverse_index
            t = times[i]
            eps = jax.vmap(self.network)(state, jnp.full((total,), t), conditions)
            predicted_x0 = (state - jnp.sqrt(1.0 - alpha_bars[i]) * eps) / jnp.sqrt(alpha_bars[i])
            ab_prev = previous[i]
            mean = (
                betas[i] * jnp.sqrt(ab_prev) / (1.0 - alpha_bars[i]) * predicted_x0
                + (1.0 - ab_prev) * jnp.sqrt(alphas[i]) / (1.0 - alpha_bars[i]) * state
            )
            variance = betas[i] * (1.0 - ab_prev) / (1.0 - alpha_bars[i])
            key, noise_key = jr.split(key)
            noise = jr.normal(noise_key, state.shape)
            next_state = jax.lax.cond(i > 0, lambda: mean + jnp.sqrt(variance) * noise, lambda: predicted_x0)
            return (next_state, key), None

        (state, _), _ = jax.lax.scan(body, (state, key), jnp.arange(steps))
        return state.reshape(observations.shape[0], num_samples, self.parameter_dim)


@eqx.filter_jit
def train_step(model, observations, parameters, key, optimizer, optimizer_state):
    loss, gradients = eqx.filter_value_and_grad(lambda m: m.loss(observations, parameters, key))(model)
    updates, optimizer_state = optimizer.update(gradients, optimizer_state, model)
    return eqx.apply_updates(model, updates), optimizer_state, loss


def sample(model, method: Literal["ddpm", "sbm"], observations, key, num_samples, steps):
    if method == "ddpm":
        return model.sample_ddpm(observations, key, num_samples, steps)
    return model.sample_sbm(observations, key, num_samples, steps)
