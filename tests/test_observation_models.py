"""Milestone 5 tests for configurable Gaussian observation models."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesinf.applications.state_estimation.observation_models import (
    IdentityGaussianObservation,
    LinearGaussianObservation,
    NonlinearGaussianObservation,
)


def test_identity_gaussian_log_prob_and_missing_mask():
    model = IdentityGaussianObservation(2, noise_std=jnp.array([0.5, 2.0]))
    state = jnp.array([[1.0, -2.0], [0.2, 0.4]])
    observation = jnp.array([[1.5, 0.0], [jnp.nan, -1.6]])
    mask = jnp.array([[True, True], [False, True]])
    actual = model.log_prob(observation, state, mask)
    residual = jnp.array([[0.5, 2.0], [0.0, -2.0]])
    terms = -0.5 * (
        (residual / jnp.array([0.5, 2.0])) ** 2 + jnp.log(2 * jnp.pi)
    ) - jnp.log(jnp.array([0.5, 2.0]))
    expected = jnp.array([jnp.sum(terms[0]), terms[1, 1]])
    assert jnp.isfinite(actual).all()
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_linear_partial_observation_mean_and_batched_jit():
    matrix = jnp.array([[1.0, 0.0, 0.5], [0.0, -2.0, 0.0]])
    model = LinearGaussianObservation(matrix, noise_std=0.3)
    states = jnp.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    expected = states @ matrix.T
    np.testing.assert_allclose(model.mean(states), expected)
    compiled = eqx.filter_jit(model.log_prob)(expected, states)
    mapped = jax.vmap(model.log_prob)(expected, states)
    np.testing.assert_allclose(compiled, mapped)
    assert model.observation_dim == 2 and model.state_dim == 3


def test_nonlinear_callable_observation_sampling_and_validation():
    def observation_function(state):
        return jnp.array([state[0] ** 2, jnp.sin(state[1])])

    model = NonlinearGaussianObservation(observation_function, 2, [0.1, 0.2])
    states = jnp.array([[2.0, 0.0], [-3.0, jnp.pi / 2]])
    expected = jnp.array([[4.0, 0.0], [9.0, 1.0]])
    np.testing.assert_allclose(eqx.filter_jit(model.mean)(states), expected, atol=1e-6)
    first = model.sample(jax.random.key(1), states)
    second = model.sample(jax.random.key(1), states)
    np.testing.assert_array_equal(first, second)
    assert first.shape == expected.shape

    broken = NonlinearGaussianObservation(lambda state: state[:1], 2, 0.1)
    with pytest.raises(ValueError, match="must return 2"):
        broken.mean(jnp.ones(3))


@pytest.mark.parametrize("noise_std", [0.0, -1.0, [0.1, 0.2, 0.3]])
def test_invalid_observation_noise_is_rejected(noise_std):
    with pytest.raises(ValueError):
        IdentityGaussianObservation(2, noise_std)
