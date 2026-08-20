"""Milestone 2 tests for the solver-free conditional transition flow."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NSFConfig,
    NeuralStochasticFlow,
    _CouplingLayer,
)


@pytest.mark.parametrize("state_dim", [1, 2, 3, 6])
@pytest.mark.parametrize("offset", [0, 1])
def test_each_coupling_layer_round_trip_and_log_det(state_dim, offset):
    mask = (jnp.arange(state_dim) + offset) % 2 == 0
    layer = _CouplingLayer(
        state_dim=state_dim,
        condition_dim=state_dim + 1,
        hidden_size=12,
        hidden_depth=2,
        max_log_scale=2.0,
        mask=mask,
        dtype=jnp.float32,
        key=jax.random.key(100 + state_dim + offset),
    )
    value = jnp.linspace(-0.7, 0.9, state_dim)
    condition = jnp.concatenate((value, jnp.array([0.35])))
    output, forward_log_det = layer.forward_and_log_det(value, condition, 0.35)
    reconstructed, inverse_log_det = layer.inverse_and_log_det(output, condition, 0.35)
    np.testing.assert_allclose(reconstructed, value, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(forward_log_det + inverse_log_det, 0.0, atol=2e-6)


@pytest.mark.parametrize("state_dim", [2, 3, 5])
def test_full_flow_analytic_log_det_matches_autodiff(state_dim):
    model = NeuralStochasticFlow(
        state_dim=state_dim,
        hidden_size=16,
        num_coupling_layers=4,
        key=jax.random.key(20 + state_dim),
    )
    base = jnp.linspace(-0.4, 0.8, state_dim)
    current = jnp.linspace(0.2, -0.3, state_dim)
    dt = jnp.asarray(0.6)
    condition = model._condition(current, dt, jnp.empty((0,)), jnp.asarray(0.0))
    _, analytic = model._forward_single(base, condition, dt)
    jacobian = jax.jacfwd(
        lambda value: model._forward_single(value, condition, dt)[0]
    )(base)
    sign, automatic = jnp.linalg.slogdet(jacobian)
    assert float(sign) > 0
    np.testing.assert_allclose(analytic, automatic, rtol=2e-5, atol=2e-6)


def test_batched_and_unbatched_log_prob_agree_under_vmap_and_jit():
    model = NeuralStochasticFlow(
        state_dim=3,
        control_dim=2,
        hidden_size=16,
        key=jax.random.key(31),
    )
    current = jnp.array([[0.1, 0.2, -0.3], [1.0, -2.0, 0.5]])
    following = jnp.array([[0.0, 0.4, -0.2], [0.8, -1.7, 0.9]])
    control = jnp.array([[0.2, -0.1], [0.5, 0.7]])
    dt = jnp.array([0.15, 0.8])

    batched = eqx.filter_jit(model.log_prob)(following, current, dt, control)
    mapped = jax.vmap(
        lambda next_i, current_i, dt_i, control_i: model.log_prob(
            next_i, current_i, dt_i, control_i
        )
    )(following, current, dt, control)
    scalar = model.log_prob(following[0], current[0], dt[0], control[0])
    np.testing.assert_allclose(batched, mapped, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(batched[0], scalar, rtol=2e-6, atol=2e-6)


def test_sample_shapes_reproducibility_and_key_separation():
    model = NeuralStochasticFlow(state_dim=3, key=jax.random.key(40))
    current = jnp.zeros((4, 3))
    dt = jnp.full((4,), 0.2)
    key = jax.random.key(41)
    first = eqx.filter_jit(model.sample)(key, current, dt, num_samples=5)
    repeated = eqx.filter_jit(model.sample)(key, current, dt, num_samples=5)
    different = model.sample(jax.random.key(42), current, dt, num_samples=5)
    assert first.shape == (5, 4, 3)
    np.testing.assert_array_equal(first, repeated)
    assert not np.array_equal(np.asarray(first), np.asarray(different))
    assert not np.array_equal(np.asarray(first[0, 0]), np.asarray(first[0, 1]))


def test_one_dimensional_density_is_numerically_normalized():
    model = NeuralStochasticFlow(
        state_dim=1,
        hidden_size=12,
        num_coupling_layers=2,
        max_log_scale=1.0,
        key=jax.random.key(50),
    )
    grid = jnp.linspace(-30.0, 30.0, 30_001)[:, None]
    log_density = eqx.filter_jit(model.log_prob)(grid, jnp.zeros((30_001, 1)), 0.7)
    integral = np.trapezoid(np.exp(np.asarray(log_density)), np.asarray(grid[:, 0]))
    assert integral == pytest.approx(1.0, rel=3e-3, abs=3e-3)


def test_physical_density_contains_normalization_jacobian():
    key = jax.random.key(60)
    unit_model = NeuralStochasticFlow(state_dim=1, key=key)
    physical_model = NeuralStochasticFlow(
        state_dim=1,
        state_mean=jnp.array([10.0]),
        state_std=jnp.array([2.0]),
        key=key,
    )
    unit_log_prob = unit_model.log_prob(jnp.array([-0.2]), jnp.array([0.5]), 0.4)
    physical_log_prob = physical_model.log_prob(
        jnp.array([9.6]), jnp.array([11.0]), 0.4
    )
    np.testing.assert_allclose(
        physical_log_prob, unit_log_prob - jnp.log(2.0), rtol=2e-6, atol=2e-6
    )


def test_absolute_time_and_control_contracts_are_explicit():
    autonomous = NeuralStochasticFlow(state_dim=2, key=jax.random.key(70))
    with pytest.raises(ValueError, match="must be omitted"):
        autonomous.log_prob(jnp.zeros(2), jnp.zeros(2), 0.2, jnp.ones(1))

    controlled = NeuralStochasticFlow(
        state_dim=2, control_dim=1, key=jax.random.key(71)
    )
    with pytest.raises(ValueError, match="control is required"):
        controlled.log_prob(jnp.zeros(2), jnp.zeros(2), 0.2)

    nonautonomous = NeuralStochasticFlow(
        state_dim=2, include_absolute_time=True, key=jax.random.key(72)
    )
    with pytest.raises(ValueError, match="t_current is required"):
        nonautonomous.log_prob(jnp.zeros(2), jnp.zeros(2), 0.2)
    assert jnp.isfinite(
        nonautonomous.log_prob(jnp.zeros(2), jnp.zeros(2), 0.2, t_current=1.5)
    )


def test_float64_configuration_and_invalid_configuration():
    with jax.enable_x64():
        model = NeuralStochasticFlow(
            state_dim=2, dtype="float64", key=jax.random.key(80)
        )
        sample = model.sample(
            jax.random.key(81), jnp.zeros(2, dtype=jnp.float64), 0.2
        )
        assert sample.dtype == jnp.float64
        assert model.state_std.dtype == jnp.float64

    with pytest.raises(ValueError, match="dtype"):
        NSFConfig(state_dim=2, dtype="float16")
    with pytest.raises(ValueError, match="positive Python integer"):
        NeuralStochasticFlow(state_dim=2).sample(
            jax.random.key(82), jnp.zeros(2), 0.2, num_samples=0
        )
