"""MPPI whose candidate paths are integrated by a controlled FlowSDE solver."""

from typing import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from .sampling_mpc import SamplingMPCConfig


def flow_sde_candidate_costs(model, x0, candidates, keys, start_step,
                             n_particles, stage_cost, risk_weight,
                             solver_name):
    """Evaluate candidate controls with solver-integrated FlowSDE particles."""
    horizon_time = jnp.arange(candidates.shape[1])

    def evaluate(controls, key):
        paths = model.sample_rollout(
            x0, controls, key, n_particles=n_particles,
            start_step=start_step, diffusion_temperature=1.0,
            solver_name=solver_name,
        )
        particle_costs = jax.vmap(lambda path: jnp.sum(jax.vmap(stage_cost)(
            path, controls, horizon_time
        )))(paths)
        return jnp.mean(particle_costs) + risk_weight * jnp.std(particle_costs)

    return jax.vmap(evaluate)(candidates, keys)


class FlowSDEMPPIController:
    """Path-integral controller using the training-time stochastic solver."""

    def __init__(self, model, action_low, action_high,
                 stage_cost: Callable, config=SamplingMPCConfig(),
                 solver_name="heun"):
        self.model, self.stage_cost, self.config = model, stage_cost, config
        self.low, self.high = jnp.asarray(action_low), jnp.asarray(action_high)
        self.solver_name = solver_name
        self.plan = jnp.zeros((config.horizon, self.low.size))
        self.last_effective_sample_size = float(config.num_candidates)
        self._cost = jax.jit(
            lambda x0, candidates, keys, start_step: flow_sde_candidate_costs(
                self.model, x0, candidates, keys, start_step,
                self.config.num_particles, self.stage_cost,
                self.config.risk_weight, self.solver_name,
            )
        )

    def reset(self):
        self.plan = jnp.zeros_like(self.plan)

    def command(self, x0, key, start_step=0):
        cfg = self.config
        nominal = jnp.concatenate([self.plan[1:], self.plan[-1:]], axis=0)
        weights = jnp.ones(cfg.num_candidates) / cfg.num_candidates
        for iteration in range(cfg.iterations):
            noise_key, rollout_key = jr.split(jr.fold_in(key, iteration))
            noise = cfg.initial_std * jr.normal(
                noise_key, (cfg.num_candidates,) + nominal.shape
            )
            candidates = jnp.clip(nominal + noise, self.low, self.high)
            costs = self._cost(
                x0, candidates, jr.split(rollout_key, cfg.num_candidates), start_step
            )
            scale = jnp.where(
                cfg.normalize_costs, jnp.maximum(jnp.std(costs), 1e-6), 1.0
            )
            weights = jax.nn.softmax(
                -(costs - jnp.min(costs)) / (cfg.temperature * scale)
            )
            nominal = jnp.clip(
                nominal + jnp.sum(
                    weights[:, None, None] * (candidates - nominal), axis=0
                ), self.low, self.high,
            )
        self.plan = nominal
        self.last_effective_sample_size = float(1.0 / jnp.sum(weights ** 2))
        return self.plan[0], self.plan.copy()
