"""Milestone 3 NLL training, OU validation, and small-time tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NSFTrainingConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    evaluate_ou_moments,
    fit_normalization,
    sample_ou_transition_data,
    train_nsf_nll,
)


@pytest.fixture(scope="module")
def trained_ou_model():
    process = OrnsteinUhlenbeck(mean=0.7, kappa=0.8, diffusion=0.45)
    training = sample_ou_transition_data(
        jax.random.key(100),
        process,
        8_192,
        1,
        delta_t_min=0.05,
        delta_t_max=1.0,
    )
    state_mean, state_std, _, _ = fit_normalization(
        jnp.concatenate((training["x_current"], training["x_next"]), axis=0)
    )
    model = NeuralStochasticFlow(
        state_dim=1,
        hidden_size=32,
        hidden_depth=2,
        num_coupling_layers=1,
        state_mean=state_mean,
        state_std=state_std,
        key=jax.random.key(101),
    )
    initial_nll = model.nll_loss(**training)
    trained, losses = train_nsf_nll(
        model,
        training,
        NSFTrainingConfig(
            steps=500,
            batch_size=256,
            learning_rate=2e-3,
            seed=102,
        ),
    )
    return process, training, trained, losses, initial_nll, state_mean, state_std


def test_nll_training_decreases_loss_and_freezes_normalization(trained_ou_model):
    _, training, model, losses, initial_nll, state_mean, state_std = trained_ou_model
    final_nll = model.nll_loss(**training)
    assert jnp.isfinite(losses).all()
    assert float(final_nll) < float(initial_nll) - 0.3
    np.testing.assert_array_equal(model.state_mean, state_mean)
    np.testing.assert_array_equal(model.state_std, state_std)


@pytest.mark.parametrize("delta_t", [0.1, 0.4, 1.0])
def test_trained_ou_conditional_moments(trained_ou_model, delta_t):
    process, _, model, _, _, _, _ = trained_ou_model
    metrics = evaluate_ou_moments(
        model,
        process,
        jax.random.key(200 + int(100 * delta_t)),
        jnp.array([-1.0]),
        delta_t,
        num_samples=10_000,
    )
    assert float(metrics["mean_absolute_error"]) < 0.09
    assert float(metrics["variance_absolute_error"]) < 0.018


def test_trained_nll_is_close_to_analytic_ou_density(trained_ou_model):
    process, _, model, _, _, _, _ = trained_ou_model
    held_out = sample_ou_transition_data(
        jax.random.key(300),
        process,
        4_096,
        1,
        delta_t_min=0.05,
        delta_t_max=1.0,
    )
    model_nll = -jnp.mean(model.log_prob(**held_out))
    analytic_nll = -jnp.mean(process.log_prob(
        held_out["x_next"], held_out["x_current"], held_out["delta_t"]
    ))
    assert abs(float(model_nll - analytic_nll)) < 0.12


def test_nll_training_is_reproducible():
    process = OrnsteinUhlenbeck()
    data = sample_ou_transition_data(
        jax.random.key(400), process, 128, 1, delta_t_min=0.2, delta_t_max=0.5
    )
    config = NSFTrainingConfig(steps=8, batch_size=32, seed=401)
    first = NeuralStochasticFlow(
        state_dim=1, hidden_size=8, num_coupling_layers=1,
        key=jax.random.key(402),
    )
    second = NeuralStochasticFlow(
        state_dim=1, hidden_size=8, num_coupling_layers=1,
        key=jax.random.key(402),
    )
    first, first_losses = train_nsf_nll(first, data, config)
    second, second_losses = train_nsf_nll(second, data, config)
    np.testing.assert_array_equal(first_losses, second_losses)
    for left, right in zip(
        jax.tree_util.tree_leaves(eqx.filter(first, eqx.is_array)),
        jax.tree_util.tree_leaves(eqx.filter(second, eqx.is_array)),
    ):
        np.testing.assert_array_equal(left, right)


def test_small_positive_time_is_finite_differentiable_and_concentrated():
    model = NeuralStochasticFlow(
        state_dim=2,
        hidden_size=16,
        zero_time_policy="deterministic",
        key=jax.random.key(500),
    )
    current = jnp.array([0.4, -1.2])
    tiny_dt = jnp.asarray(1e-6)
    log_prob = eqx.filter_jit(model.log_prob)(current, current, tiny_dt)
    derivative = jax.grad(lambda dt: model.log_prob(current, current, dt))(tiny_dt)
    assert jnp.isfinite(log_prob)
    assert jnp.isfinite(derivative)

    small = model.sample(jax.random.key(501), current, 1e-5, num_samples=8_000)
    larger = model.sample(jax.random.key(502), current, 1e-2, num_samples=8_000)
    np.testing.assert_allclose(jnp.mean(small, axis=0), current, atol=2e-3)
    assert float(jnp.max(jnp.var(small, axis=0))) < 5e-5
    assert jnp.all(jnp.var(small, axis=0) < jnp.var(larger, axis=0))


def test_transition_nll_parameter_gradients_are_finite():
    model = NeuralStochasticFlow(
        state_dim=3, control_dim=1, hidden_size=12, key=jax.random.key(600)
    )
    current = jnp.array([[0.1, -0.2, 0.3], [0.5, 0.7, -0.4]])
    following = current + jnp.array([[0.02, -0.01, 0.03], [-0.04, 0.02, 0.01]])
    control = jnp.array([[0.2], [-0.3]])
    delta_t = jnp.array([1e-4, 0.2])
    loss, gradients = eqx.filter_value_and_grad(
        lambda candidate: candidate.nll_loss(
            following, current, delta_t, control
        )
    )(model)
    leaves = jax.tree_util.tree_leaves(gradients)
    assert jnp.isfinite(loss)
    assert leaves
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)
