import jax.numpy as jnp
import jax.random as jr
import pytest

pytest.importorskip(
    "experiments.aloha_mppi_sanity",
    reason="This regression exercises an optional plant prototype absent from the public snapshot",
)

from bayesinf.applications.optimal_control.controlled_flow_sde import ControlledRealNVPFlowSDE
from experiments.aloha_mppi_sanity.dynamics_adapters import (
    FlowSDEDynamicsAdapter, SimplePhysicalModel,
)
from experiments.aloha_mppi_sanity.simple_aloha_plant import SimpleAlohaPlant


def limits(dim):
    return (-2 * jnp.ones(dim), 2 * jnp.ones(dim),
            -jnp.ones(dim), jnp.ones(dim))


def test_simple_plant_and_physical_adapter_shapes_and_bounds():
    state_low, state_high, action_low, action_high = limits(4)
    plant = SimpleAlohaPlant(
        0.22, state_low, state_high, action_low, action_high, seed=0
    )
    plant.reset(jnp.zeros(4))
    state = plant.step(3 * jnp.ones(4))
    assert state.shape == (4,)
    assert jnp.all(state <= state_high)

    model = SimplePhysicalModel(
        jnp.array([0.1, 0.2, 0.3, 0.4]),
        state_low, state_high, action_low, action_high,
    )
    assert model.predict(jnp.zeros(4), jnp.ones(4)).shape == (4,)
    assert model.predict(jnp.zeros((3, 4)), jnp.ones((3, 4))).shape == (3, 4)
    paths = model.sample_rollout(
        jnp.zeros((2, 4)), jnp.empty((0, 4)), jnp.ones((3, 4)),
        jr.PRNGKey(1), 2,
    )
    assert paths.shape == (2, 3, 4)


def test_flow_adapter_normalization_and_rollout_shapes():
    dim = 4
    state_low, state_high, action_low, action_high = limits(dim)
    flow = ControlledRealNVPFlowSDE(
        dim, dim, jr.PRNGKey(2), hidden_size=8, depth=1,
        diffusion_hidden=8, flow_layers=2, time_emb_dim=4,
    )
    adapter = FlowSDEDynamicsAdapter(
        flow, jnp.arange(dim), 2 * jnp.ones(dim),
        jnp.zeros(dim), jnp.ones(dim), state_low, state_high,
        action_low, action_high, "heun",
    )
    state = jnp.linspace(-0.5, 0.5, dim)
    recovered = adapter.inverse_normalize_state(adapter.normalize_state(state))
    assert jnp.allclose(recovered, state)
    assert adapter.predict(state, jnp.zeros(dim)).shape == state.shape
    assert adapter.predict(jnp.stack([state, state]), jnp.zeros((2, dim))).shape == (2, dim)
    paths = adapter.sample_rollout(
        jnp.stack([state, state]), jnp.empty((0, dim)),
        jnp.zeros((3, dim)), jr.PRNGKey(3), 2,
    )
    assert paths.shape == (2, 3, dim)
    assert jnp.all(jnp.isfinite(paths))

