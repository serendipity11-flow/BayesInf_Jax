import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import pytest

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    make_transition_pairs,
)


def test_ou_transition_matches_known_scalar_formula():
    ou = OrnsteinUhlenbeck(mean=2.0, kappa=0.5, diffusion=0.3)
    x = jnp.array([1.0])
    dt = jnp.array(0.7)
    expected_mean = 2.0 + (1.0 - 2.0) * np.exp(-0.5 * 0.7)
    expected_var = 0.3**2 * (1.0 - np.exp(-2.0 * 0.5 * 0.7)) / (2.0 * 0.5)
    np.testing.assert_allclose(ou.transition_mean(x, dt), [expected_mean], rtol=1e-6)
    np.testing.assert_allclose(ou.transition_variance(dt), expected_var, rtol=1e-6)


def test_ou_small_time_and_jit_vmap():
    ou = OrnsteinUhlenbeck(mean=jnp.array([0.0, 1.0]), kappa=jnp.array([1.0, 2.0]), diffusion=0.4)
    states = jnp.array([[2.0, -1.0], [0.5, 3.0]])
    dts = jnp.array([1e-7, 0.2])
    means = jax.jit(jax.vmap(ou.transition_mean))(states, dts)
    np.testing.assert_allclose(means[0], states[0], atol=5e-7)
    assert np.all(np.asarray(ou.transition_variance(jnp.array(1e-7))) > 0)


def test_ou_sampling_is_reproducible_and_density_rejects_zero():
    ou = OrnsteinUhlenbeck()
    key = jax.random.key(17)
    x = jnp.array([0.2, -0.1])
    np.testing.assert_array_equal(ou.sample(key, x, 0.3), ou.sample(key, x, 0.3))
    with pytest.raises(ValueError, match="strictly positive"):
        ou.log_prob(x, x, 0.0)


def test_transition_pairs_do_not_cross_trajectory_boundaries():
    states = jnp.array([[[0.0], [1.0], [2.0]], [[10.0], [11.0], [12.0]]])
    times = jnp.array([[0.0, 0.1, 0.4], [2.0, 2.2, 2.5]])
    pairs = make_transition_pairs(states, times)
    np.testing.assert_array_equal(pairs["x_current"][:, 0], [0.0, 1.0, 10.0, 11.0])
    np.testing.assert_array_equal(pairs["x_next"][:, 0], [1.0, 2.0, 11.0, 12.0])
    np.testing.assert_allclose(pairs["delta_t"], [0.1, 0.3, 0.2, 0.3], atol=1e-7)


def test_nsf_round_trip_log_det_and_small_time_identity():
    model = NeuralStochasticFlow(
        state_dim=3, num_coupling_layers=3, hidden_size=16,
        key=jax.random.key(2), zero_time_policy="deterministic",
    )
    x = jnp.array([0.4, -0.7, 1.2])
    dt = jnp.array(0.3)
    condition = model._condition(x, dt, jnp.empty((0,)), jnp.array(0.0))
    transformed, forward_ld = model._forward_single(x, condition, dt)
    reconstructed, inverse_ld = model._inverse_single(transformed, condition, dt)
    np.testing.assert_allclose(reconstructed, x, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(forward_ld + inverse_ld, 0.0, atol=1e-6)
    at_zero = model.sample(jax.random.key(3), x, 0.0, num_samples=2)
    np.testing.assert_array_equal(at_zero, jnp.broadcast_to(x, (2, 3)))


def test_nsf_jit_batched_sample_and_log_prob():
    model = NeuralStochasticFlow(
        state_dim=2, control_dim=1, hidden_size=16,
        key=jax.random.key(4),
    )
    current = jnp.array([[0.0, 1.0], [2.0, -1.0]])
    control = jnp.array([[0.2], [-0.3]])
    dt = jnp.array([0.1, 0.4])
    samples = jax.jit(lambda k: model.sample(k, current, dt, control, 3))(
        jax.random.key(5)
    )
    assert samples.shape == (3, 2, 2)
    log_prob = eqx.filter_jit(model.log_prob)(samples[0], current, dt, control)
    assert log_prob.shape == (2,)
    assert np.all(np.isfinite(np.asarray(log_prob)))


def test_nsf_analytic_and_automatic_coupling_jacobians_agree():
    model = NeuralStochasticFlow(
        state_dim=3, num_coupling_layers=2, hidden_size=12,
        key=jax.random.key(7),
    )
    value = jnp.array([0.2, -0.4, 0.8])
    dt = jnp.array(0.25)
    condition = model._condition(value, dt, jnp.empty((0,)), jnp.array(0.0))
    output, analytic = model._forward_single(value, condition, dt)
    del output
    jacobian = jax.jacfwd(lambda z: model._forward_single(z, condition, dt)[0])(value)
    _, automatic = jnp.linalg.slogdet(jacobian)
    np.testing.assert_allclose(analytic, automatic, rtol=2e-5, atol=2e-6)


def test_nsf_sampling_returns_physical_units():
    model = NeuralStochasticFlow(
        state_dim=2,
        state_mean=jnp.array([10.0, -3.0]),
        state_std=jnp.array([2.0, 4.0]),
        zero_time_policy="deterministic",
        key=jax.random.key(8),
    )
    physical = jnp.array([12.0, 5.0])
    samples = model.sample(jax.random.key(9), physical, 0.0, num_samples=2)
    np.testing.assert_array_equal(samples, jnp.broadcast_to(physical, (2, 2)))
