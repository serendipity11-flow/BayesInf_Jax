"""Milestone 4 tests for the endpoint-consistent NSF bridge."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NSFBridge,
    NeuralStochasticFlow,
)


@pytest.mark.parametrize("state_dim", [1, 3, 4])
def test_bridge_flow_round_trip_and_analytic_log_det(state_dim):
    bridge = NSFBridge(
        state_dim,
        hidden_size=10,
        hidden_depth=1,
        num_coupling_layers=3,
        key=jax.random.key(10 + state_dim),
    )
    start = jnp.linspace(-0.4, 0.3, state_dim)
    end = jnp.linspace(0.8, -0.2, state_dim)
    before, after = jnp.asarray(0.2), jnp.asarray(0.5)
    total, tau = before + after, before / (before + after)
    condition = bridge._condition(start, end, total, tau, jnp.asarray(0.0))
    alpha = tau * (1.0 - tau)
    base = jnp.linspace(-0.7, 0.5, state_dim)

    value, forward_log_det = bridge._forward_single(base, condition, alpha)
    recovered, inverse_log_det = bridge._inverse_single(value, condition, alpha)
    jacobian = jax.jacfwd(
        lambda point: bridge._forward_single(point, condition, alpha)[0]
    )(base)
    _, autodiff_log_det = jnp.linalg.slogdet(jacobian)

    np.testing.assert_allclose(recovered, base, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(
        forward_log_det + inverse_log_det, 0.0, atol=2e-5
    )
    np.testing.assert_allclose(
        forward_log_det, autodiff_log_det, rtol=2e-5, atol=2e-5
    )


def test_bridge_is_exact_at_both_endpoints_and_rejects_endpoint_density():
    bridge = NSFBridge(
        2,
        state_mean=jnp.array([2.0, -1.0]),
        state_std=jnp.array([0.5, 3.0]),
        key=jax.random.key(20),
    )
    start = jnp.array([2.3, -3.0])
    end = jnp.array([1.5, 4.0])
    at_start = bridge.sample(jax.random.key(21), start, end, 0.0, 0.7, num_samples=5)
    at_end = bridge.sample(jax.random.key(22), start, end, 0.7, 0.0, num_samples=5)
    np.testing.assert_array_equal(at_start, jnp.broadcast_to(start, (5, 2)))
    np.testing.assert_array_equal(at_end, jnp.broadcast_to(end, (5, 2)))
    with pytest.raises(ValueError, match="strict interior"):
        bridge.log_prob(start, start, end, 0.0, 0.7)


def test_bridge_jit_vmap_and_physical_density_normalization():
    transition = NeuralStochasticFlow(
        1,
        state_mean=jnp.array([1.7]),
        state_std=jnp.array([2.4]),
        hidden_size=8,
        hidden_depth=1,
        num_coupling_layers=2,
        key=jax.random.key(30),
    )
    bridge = NSFBridge.from_transition(
        transition,
        hidden_size=8,
        hidden_depth=1,
        num_coupling_layers=2,
        key=jax.random.key(31),
    )
    starts = jnp.array([[0.2], [1.1], [-0.8]])
    ends = jnp.array([[1.0], [-0.4], [0.7]])
    middles = jnp.array([[0.5], [0.2], [-0.1]])
    before = jnp.array([0.2, 0.3, 0.4])
    after = jnp.array([0.5, 0.4, 0.2])
    compiled = eqx.filter_jit(bridge.log_prob)(
        middles, starts, ends, before, after
    )
    mapped = jax.vmap(bridge.log_prob)(middles, starts, ends, before, after)
    np.testing.assert_allclose(compiled, mapped, rtol=2e-5, atol=2e-5)
    samples = eqx.filter_jit(bridge.sample)(
        jax.random.key(32), starts, ends, before, after, num_samples=4
    )
    assert samples.shape == (4, 3, 1)
    assert jnp.isfinite(samples).all()

    grid = jnp.linspace(-20.0, 20.0, 10_001)[:, None]
    grid_start = jnp.broadcast_to(starts[0], grid.shape)
    grid_end = jnp.broadcast_to(ends[0], grid.shape)
    density = jnp.exp(
        bridge.log_prob(grid, grid_start, grid_end, 0.25, 0.35)
    )
    integral = jnp.trapezoid(density, grid[:, 0])
    np.testing.assert_allclose(integral, 1.0, rtol=2e-3, atol=2e-3)


def test_bridge_parameter_gradients_are_finite():
    bridge = NSFBridge(3, hidden_size=10, key=jax.random.key(40))
    start = jnp.array([[0.1, -0.3, 0.4], [0.5, 0.2, -0.7]])
    end = jnp.array([[0.8, 0.1, -0.2], [-0.1, 0.9, 0.3]])
    middle = jnp.array([[0.4, -0.1, 0.0], [0.2, 0.5, -0.2]])
    loss, gradients = eqx.filter_value_and_grad(
        lambda candidate: -jnp.mean(
            candidate.log_prob(middle, start, end, 0.2, 0.4)
        )
    )(bridge)
    leaves = jax.tree_util.tree_leaves(gradients)
    assert jnp.isfinite(loss)
    assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
