"""JIT/vectorized learned MPPI and exact-MuJoCo oracle MPPI."""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from .state import ACTION_DIM, GATE_VECTOR, OBJECT_POSITION, OBJECT_YAW
from .state import latest_frame, update_history


@dataclass(frozen=True)
class MPPIConfig:
    horizon: int = 30
    samples: int = 1000
    temperature: float = 1.0
    action_std: tuple[float, float, float] = (0.35, 0.35, 0.15)
    action_limit: float = 1.2
    goal_weight: float = 20.0
    yaw_weight: float = 1.0
    obstacle_weight: float = 30.0
    control_weight: float = 0.01
    terminal_weight: float = 8.0


class NormalizedDynamics(eqx.Module):
    """Expose physical-coordinate steps around a normalized trained model."""
    model: object
    history_mean: jax.Array
    history_std: jax.Array
    action_mean: jax.Array
    action_std: jax.Array
    delta_mean: jax.Array
    delta_std: jax.Array

    def __init__(self, model, stats):
        self.model = model
        for name in ("history_mean", "history_std", "action_mean", "action_std",
                     "delta_mean", "delta_std"):
            setattr(self, name, jnp.asarray(stats[name]))

    def step(self, history, action, key, stochastic=True):
        normalized = self.model.step((history - self.history_mean) / self.history_std,
                                     (action - self.action_mean) / self.action_std,
                                     key, stochastic)
        return normalized * self.delta_std + self.delta_mean


def pushing_cost(state, action, target, cfg, terminal=False):
    position = jnp.sum((state[OBJECT_POSITION] - target[:3]) ** 2)
    yaw_error = jnp.arctan2(jnp.sin(state[OBJECT_YAW] - target[3]),
                            jnp.cos(state[OBJECT_YAW] - target[3]))
    gate_x = jnp.exp(-0.5 * (state[GATE_VECTOR.start] / 0.11) ** 2)
    penetration = jnp.sum(jax.nn.relu(-state[-2:]) ** 2)
    cost = (cfg.goal_weight * position + cfg.yaw_weight * yaw_error ** 2
            + cfg.obstacle_weight * gate_x * penetration
            + cfg.control_weight * jnp.sum(action ** 2))
    return cost * (cfg.terminal_weight if terminal else 1.0)


def learned_candidate_costs(dynamics, history, candidates, keys, target, cfg,
                            deterministic=False):
    simulated = jnp.repeat(history[None], cfg.samples, 0)
    def scan_step(carry, inputs):
        actions, step_keys = inputs
        deltas = jax.vmap(lambda x, u, k: dynamics.step(
            x, u, k, not deterministic))(carry, actions, step_keys)
        next_states = latest_frame(carry) + deltas
        costs = jax.vmap(lambda x, u: pushing_cost(x, u, target, cfg))(next_states, actions)
        return update_history(carry, next_states, actions), costs
    simulated, stage_costs = jax.lax.scan(
        scan_step, simulated, (jnp.swapaxes(candidates, 0, 1), jnp.swapaxes(keys, 0, 1)))
    terminal = jax.vmap(lambda x, u: pushing_cost(
        x, u, target, cfg, True))(latest_frame(simulated), candidates[:, -1])
    return stage_costs.sum(0) + terminal


class MPPIPlanner:
    def __init__(self, dynamics, config=MPPIConfig(), deterministic=False):
        self.dynamics, self.config, self.deterministic = dynamics, config, deterministic
        self.plan = jnp.zeros((config.horizon, ACTION_DIM))
        self.last_effective_sample_size = float(config.samples)
        self._costs = eqx.filter_jit(lambda history, candidates, keys, target:
            learned_candidate_costs(self.dynamics, history, candidates, keys,
                                    target, self.config, self.deterministic))

    def reset(self): self.plan = jnp.zeros_like(self.plan)

    def command(self, history, target, key):
        cfg = self.config
        noise_key, rollout_key = jr.split(key)
        noise = jr.normal(noise_key, (cfg.samples, cfg.horizon, ACTION_DIM))
        candidates = jnp.clip(self.plan[None] + noise * jnp.asarray(cfg.action_std),
                              -cfg.action_limit, cfg.action_limit)
        costs = self._costs(jnp.asarray(history), candidates,
                            jr.split(rollout_key, cfg.samples * cfg.horizon).reshape(
                                cfg.samples, cfg.horizon, 2), jnp.asarray(target))
        weights = jax.nn.softmax(-(costs - costs.min()) / cfg.temperature)
        optimal = jnp.sum(weights[:, None, None] * candidates, axis=0)
        command = optimal[0]
        self.plan = jnp.concatenate((optimal[1:], optimal[-1:]))
        self.last_effective_sample_size = float(1 / jnp.sum(weights ** 2))
        return np.asarray(command), np.asarray(optimal)


class SimulatorMPPIPlanner:
    """Oracle baseline evaluating candidate paths through MuJoCo clones."""
    def __init__(self, env, config=MPPIConfig(), seed=0):
        self.env, self.config, self.rng = env, config, np.random.default_rng(seed)
        self.plan = np.zeros((config.horizon, ACTION_DIM))

    def reset(self): self.plan.fill(0)

    def command(self, history, target):
        cfg, snapshot = self.config, self.env.snapshot()
        candidates = np.clip(self.plan[None] + self.rng.normal(
            size=(cfg.samples, cfg.horizon, ACTION_DIM)) * np.asarray(cfg.action_std),
            -cfg.action_limit, cfg.action_limit)
        costs = np.zeros(cfg.samples)
        for k, controls in enumerate(candidates):
            self.env.restore(snapshot)
            for h, action in enumerate(controls):
                state = self.env.step(action, noisy=False)
                costs[k] += float(pushing_cost(jnp.asarray(state), jnp.asarray(action),
                                               jnp.asarray(target), cfg,
                                               h + 1 == cfg.horizon))
        self.env.restore(snapshot)
        weights = np.exp(-(costs - costs.min()) / cfg.temperature); weights /= weights.sum()
        optimal = np.sum(weights[:, None, None] * candidates, 0); command = optimal[0].copy()
        self.plan = np.concatenate((optimal[1:], optimal[-1:]))
        return command, optimal
