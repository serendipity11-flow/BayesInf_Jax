"""Particle rollouts, CEM stochastic MPC, and an MPPI baseline."""

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr


Array = jnp.ndarray
CostFn = Callable[[Array, Array, int], Array]


@dataclass(frozen=True)
class SamplingMPCConfig:
    horizon: int = 20
    num_candidates: int = 256
    num_particles: int = 8
    iterations: int = 5
    elite_fraction: float = 0.1
    initial_std: float = 0.25
    min_std: float = 0.01
    temperature: float = 1.0
    risk_weight: float = 0.0
    smoothness_weight: float = 0.0
    smoothness_scale: float = 1.0
    normalize_costs: bool = False


def _candidate_cost(model, x_history, u_history, candidates, keys,
                    num_particles: int, stage_cost: CostFn,
                    risk_weight: float, smoothness_weight: float = 0.0,
                    smoothness_scale: Array | float = 1.0) -> Array:
    def evaluate(actions, key):
        rollouts = model.sample_rollout(
            x_history, u_history, actions, key, num_particles
        )
        time = jnp.arange(actions.shape[0])
        costs = jax.vmap(
            lambda trajectory: jnp.sum(jax.vmap(stage_cost)(
                trajectory, actions, time
            ))
        )(rollouts)
        # Mean + lambda*std is a simple risk-sensitive objective. lambda=0 is
        # risk-neutral Monte Carlo MPC and still uses the full flow uncertainty.
        action_rate = jnp.diff(actions, axis=0) / jnp.asarray(smoothness_scale)
        smoothness_cost = smoothness_weight * jnp.sum(jnp.mean(action_rate ** 2, axis=-1))
        return jnp.mean(costs) + risk_weight * jnp.std(costs) + smoothness_cost
    return jax.vmap(evaluate)(candidates, keys)


class CEMStochasticMPC:
    """Cross-entropy MPC over action sequences and learned flow particles."""

    def __init__(self, model, action_low: Array, action_high: Array,
                 stage_cost: CostFn, config: SamplingMPCConfig = SamplingMPCConfig()):
        self.model, self.stage_cost, self.config = model, stage_cost, config
        self.low, self.high = jnp.asarray(action_low), jnp.asarray(action_high)
        self.mean = jnp.zeros((config.horizon, self.low.size))

    def reset(self):
        self.mean = jnp.zeros_like(self.mean)

    def command(self, x_history: Array, u_history: Array,
                key: jr.PRNGKey) -> Tuple[Array, Array]:
        cfg = self.config
        mean = jnp.concatenate([self.mean[1:], self.mean[-1:]], axis=0)
        std = jnp.broadcast_to(cfg.initial_std, mean.shape)
        iteration_keys = jr.split(key, cfg.iterations)
        n_elite = max(1, int(cfg.num_candidates * cfg.elite_fraction))
        for iteration_key in iteration_keys:
            sample_key, rollout_key = jr.split(iteration_key)
            candidates = jnp.clip(
                mean + std * jr.normal(sample_key, (cfg.num_candidates,) + mean.shape),
                self.low, self.high,
            )
            costs = _candidate_cost(
                self.model, x_history, u_history, candidates,
                jr.split(rollout_key, cfg.num_candidates), cfg.num_particles,
                self.stage_cost, cfg.risk_weight, cfg.smoothness_weight,
                cfg.smoothness_scale,
            )
            elite = candidates[jnp.argsort(costs)[:n_elite]]
            mean = jnp.mean(elite, axis=0)
            std = jnp.maximum(jnp.std(elite, axis=0), cfg.min_std)
        self.mean = jnp.clip(mean, self.low, self.high)
        return self.mean[0], self.mean.copy()


class MPPIController:
    """Standard path-integral-style sampling baseline using the same model.

    This implements exponential cost weighting of Gaussian perturbations.  The
    optional control-noise likelihood/covariance correction from the original
    control-affine MPPI derivation is intentionally omitted, making this the
    common practical, model-agnostic MPPI variant.
    """

    def __init__(self, model, action_low: Array, action_high: Array,
                 stage_cost: CostFn, config: SamplingMPCConfig = SamplingMPCConfig()):
        self.model, self.stage_cost, self.config = model, stage_cost, config
        self.low, self.high = jnp.asarray(action_low), jnp.asarray(action_high)
        self.plan = jnp.zeros((config.horizon, self.low.size))
        self.last_effective_sample_size = float(config.num_candidates)
        self._cost = jax.jit(lambda x_history, u_history, candidates, keys:
            _candidate_cost(
                self.model, x_history, u_history, candidates, keys,
                self.config.num_particles, self.stage_cost,
                self.config.risk_weight, self.config.smoothness_weight,
                self.config.smoothness_scale,
            ))

    def reset(self):
        self.plan = jnp.zeros_like(self.plan)

    def command(self, x_history: Array, u_history: Array,
                key: jr.PRNGKey) -> Tuple[Array, Array]:
        cfg = self.config
        nominal = jnp.concatenate([self.plan[1:], self.plan[-1:]], axis=0)
        weights = jnp.ones((cfg.num_candidates,)) / cfg.num_candidates
        for iteration in range(cfg.iterations):
            iteration_key = jr.fold_in(key, iteration)
            noise_key, rollout_key = jr.split(iteration_key)
            noise = cfg.initial_std * jr.normal(
                noise_key, (cfg.num_candidates,) + nominal.shape
            )
            candidates = jnp.clip(nominal + noise, self.low, self.high)
            costs = self._cost(
                x_history, u_history, candidates,
                jr.split(rollout_key, cfg.num_candidates),
            )
            cost_scale = jnp.where(
                cfg.normalize_costs, jnp.maximum(jnp.std(costs), 1e-6), 1.0
            )
            weights = jax.nn.softmax(
                -(costs - jnp.min(costs)) / (cfg.temperature * cost_scale)
            )
            # Use the actual clipped displacement so updates remain consistent
            # when candidates hit action bounds.
            nominal = jnp.clip(
                nominal + jnp.sum(
                    weights[:, None, None] * (candidates - nominal), axis=0
                ), self.low, self.high,
            )
        self.plan = nominal
        self.last_effective_sample_size = float(
            1.0 / jnp.sum(weights ** 2)
        )
        return self.plan[0], self.plan.copy()
