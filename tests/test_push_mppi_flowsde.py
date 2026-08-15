import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control.conditional_flow_sde import FlowSDE
from experiments.push_mppi_flowsde.mppi_planner import MPPIConfig, learned_candidate_costs
from experiments.push_mppi_flowsde.state import (
    ACTION_DIM, HISTORY, INPUT_DIM, STATE_DIM, latest_frame, pack_history,
    update_history,
)
from experiments.push_mppi_flowsde.train import contact_weights


def small_model():
    return FlowSDE(INPUT_DIM, ACTION_DIM, STATE_DIM, jr.PRNGKey(0),
                   width=16, condition_dim=8, flow_layers=2)


def test_history_window_is_newest_first():
    states = jnp.arange(HISTORY * STATE_DIM).reshape(HISTORY, STATE_DIM)
    actions = jnp.arange((HISTORY - 1) * ACTION_DIM).reshape(HISTORY - 1, ACTION_DIM)
    history = pack_history(states, actions)
    next_state, action = jnp.full((STATE_DIM,), -1), jnp.full((ACTION_DIM,), -2)
    updated = update_history(history, next_state, action)
    assert jnp.array_equal(latest_frame(updated), next_state)
    assert jnp.array_equal(updated[STATE_DIM:2 * STATE_DIM], states[0])


def test_conditional_realnvp_round_trip_and_logdet():
    model = small_model()
    history, action = jnp.zeros(INPUT_DIM), jnp.zeros(ACTION_DIM)
    condition = model.conditioner(jnp.concatenate((history, action)))
    base = jr.normal(jr.PRNGKey(1), (STATE_DIM,))
    value, forward_logdet = model.flow.forward(base, condition)
    recovered, inverse_logdet = model.flow.inverse(value, condition)
    assert jnp.max(jnp.abs(base - recovered)) < 2e-6
    assert jnp.abs(forward_logdet + inverse_logdet) < 2e-6


def test_contact_weighting_and_vectorized_candidate_costs():
    delta = jnp.zeros((2, STATE_DIM)).at[1, 10].set(0.01)
    assert jnp.array_equal(contact_weights(delta), jnp.asarray((1.0, 10.0)))
    cfg = MPPIConfig(horizon=3, samples=4)
    candidates = jnp.zeros((cfg.samples, cfg.horizon, ACTION_DIM))
    keys = jr.split(jr.PRNGKey(2), cfg.samples * cfg.horizon).reshape(
        cfg.samples, cfg.horizon, 2
    )
    costs = learned_candidate_costs(
        small_model(), jnp.zeros(INPUT_DIM), candidates, keys,
        jnp.asarray((0.68, 0.0, 0.055, 0.0)), cfg,
    )
    assert costs.shape == (cfg.samples,)
    assert jnp.all(jnp.isfinite(costs))
