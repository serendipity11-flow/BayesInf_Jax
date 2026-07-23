import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bayesinf.applications.optimal_control import CondStochInterpolantPolicy
from bayesinf.applications.state_estimation.stochastic_interpolant import (
    ObsTransformerEncoder,
    SITransformerNet,
    StateInferenceMLP,
)


def small_policy(**kwargs):
    return CondStochInterpolantPolicy(
        obs_dim=6,
        action_dim=3,
        horizon=4,
        cond_steps=2,
        integration_steps=3,
        hidden_size=16,
        depth=2,
        time_dim=8,
        key=jr.PRNGKey(0),
        **kwargs,
    )


def batch():
    states = jr.normal(jr.PRNGKey(1), (5, 2, 6))
    actions = jnp.tanh(jr.normal(jr.PRNGKey(2), (5, 4, 3)))
    return states, actions


def test_loss_is_finite_and_differentiable():
    policy = small_policy()
    states, actions = batch()
    loss, grads = eqx.filter_value_and_grad(
        lambda model: model.loss(actions, states, jr.PRNGKey(3))
    )(policy)
    leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array))
    assert jnp.isfinite(loss)
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in leaves)


def test_policy_reuses_state_estimation_components():
    policy = small_policy()
    assert isinstance(policy.encoder, ObsTransformerEncoder)
    assert isinstance(policy.inference_net, StateInferenceMLP)
    assert isinstance(policy.vel_net, SITransformerNet)
    assert policy.base is policy.inference_net
    assert policy.velocity is policy.vel_net


def test_sample_chain_shapes_repeatability_and_nfe():
    policy = small_policy()
    states, _ = batch()
    first = policy.sample(states, jr.PRNGKey(4))
    repeated = policy.sample(states, jr.PRNGKey(4))
    other = policy.sample(states, jr.PRNGKey(5))
    deterministic = policy.sample(states, jr.PRNGKey(6), deterministic=True)
    deterministic_again = policy.sample(states, jr.PRNGKey(7), deterministic=True)
    assert first.actions.shape == (5, 4, 3)
    assert first.chain.shape == (5, 4, 4, 3)
    assert policy.nfe == 3
    assert jnp.allclose(first.chain, repeated.chain)
    assert not jnp.allclose(first.chain, other.chain)
    assert jnp.allclose(deterministic.chain, deterministic_again.chain)
    assert jnp.all(jnp.abs(first.actions) <= 1)


def test_transition_log_prob_is_finite_and_differentiable():
    policy = small_policy()
    states, _ = batch()
    chain = jax.lax.stop_gradient(policy.sample(states, jr.PRNGKey(8)).chain)
    log_prob = policy.transition_log_prob(states, chain)
    subset = policy.transition_log_prob(states, chain, jnp.array([1, 2]))
    assert log_prob.shape == (5, 3, 4, 3)
    assert subset.shape == (5, 2, 4, 3)
    assert jnp.all(jnp.isfinite(log_prob))
    _, grads = eqx.filter_value_and_grad(
        lambda model: jnp.mean(model.transition_log_prob(states, chain))
    )(policy)
    velocity_leaves = jax.tree_util.tree_leaves(
        eqx.filter(grads.velocity, eqx.is_array)
    )
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in velocity_leaves)


def test_gaussian_base_ablation_and_source_entropy():
    informative = small_policy()
    gaussian = small_policy(informative_base=False)
    states, _ = batch()
    mean, log_std = informative.base_params(states[0])
    gaussian_mean, gaussian_log_std = gaussian.base_params(states[0])
    assert mean.shape == log_std.shape == (4, 3)
    assert jnp.all(jnp.isfinite(mean)) and jnp.all(jnp.isfinite(log_std))
    assert jnp.allclose(gaussian_mean, 0)
    assert jnp.allclose(gaussian_log_std, 0)
    sample_a = informative.sample(states, jr.PRNGKey(9)).chain[:, 0]
    sample_b = informative.sample(states, jr.PRNGKey(10)).chain[:, 0]
    assert not jnp.allclose(sample_a, sample_b)


def test_input_validation_and_zero_std_likelihood_error():
    policy = small_policy(sampling_std=0.0)
    states, _ = batch()
    with pytest.raises(ValueError, match="states must have shape"):
        policy.sample(states[:, 0], jr.PRNGKey(11))
    chain = policy.sample(states, jr.PRNGKey(12)).chain
    with pytest.raises(ValueError, match="sampling_std"):
        policy.transition_log_prob(states, chain)
