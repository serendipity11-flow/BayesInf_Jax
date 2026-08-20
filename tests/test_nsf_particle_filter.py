"""Milestone 5 tests for NSF bootstrap state estimation."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesinf.applications.state_estimation import (
    IdentityGaussianObservation,
    LinearGaussianObservation,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    ParticleFilterConfig,
    bootstrap_particle_filter_nsf,
    reconstruct_particle_trajectories,
)


def test_nsf_filter_outputs_weighted_diagnostics_and_ancestors():
    model = NeuralStochasticFlow(
        2,
        hidden_size=8,
        hidden_depth=1,
        num_coupling_layers=2,
        key=jax.random.key(10),
    )
    observation_model = LinearGaussianObservation(jnp.array([[1.0, 0.0]]), 0.2)
    observations = jnp.array(
        [[[0.0], [0.1], [0.2]], [[0.5], [0.4], [0.3]]]
    )
    x0 = jnp.array([[0.0, 0.3], [0.5, -0.2]])
    result = eqx.filter_jit(bootstrap_particle_filter_nsf)(
        observations,
        x0,
        jnp.array([0.0, 0.1, 0.25]),
        jax.random.key(11),
        model,
        observation_model,
        config=ParticleFilterConfig(
            num_particles=32,
            initial_std=0.3,
            resampling_threshold=0.8,
            track_ancestors=True,
        ),
    )
    assert result.mean.shape == (2, 3, 2)
    assert result.covariance.shape == (2, 3, 2, 2)
    assert result.particles.shape == (2, 3, 32, 2)
    assert result.weights.shape == (2, 3, 32)
    assert result.effective_sample_size.shape == (2, 3)
    assert result.resampled.shape == (2, 3)
    assert result.credible_intervals.shape == (2, 3, 3, 2, 2)
    assert result.ancestors.shape == (2, 3, 32)
    np.testing.assert_allclose(jnp.sum(result.weights, axis=-1), 1.0, atol=1e-6)
    assert jnp.all((result.effective_sample_size >= 1) &
                   (result.effective_sample_size <= 32 + 1e-4))
    weighted_mean = jnp.sum(
        result.weights[..., None] * result.particles, axis=-2
    )
    np.testing.assert_allclose(result.mean, weighted_mean, rtol=2e-5, atol=2e-5)
    assert jnp.all(
        result.credible_intervals[..., 0, :] <=
        result.credible_intervals[..., 1, :]
    )
    trajectories = reconstruct_particle_trajectories(result)
    assert trajectories.shape == (2, 32, 3, 2)
    assert jnp.isfinite(trajectories).all()


def test_missing_observation_keeps_uniform_initial_weights():
    model = NeuralStochasticFlow(
        1, hidden_size=4, hidden_depth=1, num_coupling_layers=1,
        key=jax.random.key(20),
    )
    observation_model = IdentityGaussianObservation(1, 0.1)
    observations = jnp.array([[jnp.nan], [0.2]])
    mask = jnp.array([[False], [True]])
    result = bootstrap_particle_filter_nsf(
        observations,
        jnp.array([0.0]),
        jnp.array([0.0, 0.1]),
        jax.random.key(21),
        model,
        observation_model,
        mask=mask,
        config=ParticleFilterConfig(num_particles=24, initial_std=0.5),
    )
    np.testing.assert_allclose(result.weights[0], 1.0 / 24, atol=1e-7)
    np.testing.assert_allclose(result.effective_sample_size[0], 24.0, atol=2e-5)
    assert jnp.isfinite(result.mean).all()
    assert result.ancestors is None


@pytest.mark.parametrize("method", ["systematic", "stratified"])
def test_configurable_resampling_responds_to_low_ess(method):
    model = NeuralStochasticFlow(
        1, hidden_size=4, hidden_depth=1, num_coupling_layers=1,
        key=jax.random.key(30),
    )
    observation_model = IdentityGaussianObservation(1, 1e-3)
    result = bootstrap_particle_filter_nsf(
        jnp.array([[0.0], [0.0]]),
        jnp.array([0.0]),
        jnp.array([0.0, 0.1]),
        jax.random.key(31),
        model,
        observation_model,
        config=ParticleFilterConfig(
            num_particles=64,
            initial_std=1.0,
            resampling_threshold=1.0,
            resampling_method=method,
            track_ancestors=True,
        ),
    )
    assert bool(result.resampled[0])
    assert float(result.effective_sample_size[0]) < 64.0
    assert jnp.all((result.ancestors >= 0) & (result.ancestors < 64))


class _ExactOUTransition:
    state_dim = 1
    control_dim = 0

    def __init__(self, process):
        self.process = process

    def sample(self, key, x_current, delta_t, control=None, num_samples=1,
               t_current=None):
        keys = jax.random.split(key, num_samples)
        return jax.vmap(
            lambda sample_key: self.process.sample(
                sample_key, x_current, delta_t
            )
        )(keys)


def test_ou_particle_filter_improves_over_noisy_measurements():
    process = OrnsteinUhlenbeck(mean=0.4, kappa=0.9, diffusion=0.25)
    transition = _ExactOUTransition(process)
    num_trajectories, num_times = 8, 25
    initial_key, rollout_key, observation_key, filter_key = jax.random.split(
        jax.random.key(40), 4
    )
    initial = process.mean + jax.random.normal(initial_key, (num_trajectories, 1))
    step_keys = jax.random.split(rollout_key, num_times - 1)

    def step(current, key):
        following = process.sample(key, current, 0.1)
        return following, following

    _, following = jax.lax.scan(step, initial, step_keys)
    truth = jnp.concatenate((initial[:, None], following.swapaxes(0, 1)), axis=1)
    observation_model = IdentityGaussianObservation(1, 0.45)
    observations = observation_model.sample(observation_key, truth)
    result = eqx.filter_jit(bootstrap_particle_filter_nsf)(
        observations,
        truth[:, 0],
        jnp.arange(num_times) * 0.1,
        filter_key,
        transition,
        observation_model,
        config=ParticleFilterConfig(
            num_particles=256,
            initial_std=0.2,
            resampling_threshold=0.5,
        ),
    )
    filter_rmse = jnp.sqrt(jnp.mean((result.mean - truth) ** 2))
    observation_rmse = jnp.sqrt(jnp.mean((observations - truth) ** 2))
    assert float(filter_rmse) < float(observation_rmse) * 0.75
    assert float(filter_rmse) < 0.25
