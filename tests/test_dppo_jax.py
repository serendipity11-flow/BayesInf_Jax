import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control import (
    CondStochInterpolantPolicy,
    DPPOConfig,
    RolloutBatch,
    ValueCritic,
    bc_update,
    dppo_loss,
    dppo_update,
    generalized_advantage_estimate,
    make_bc_state,
    make_dppo_state,
)


def models():
    policy = CondStochInterpolantPolicy(
        6, 3, 4, jr.PRNGKey(0), integration_steps=3,
        hidden_size=16, depth=1, time_dim=8,
    )
    critic = ValueCritic(6, 1, jr.PRNGKey(1), hidden_size=16, depth=1)
    return policy, critic


def test_gae_matches_explicit_recursion():
    rewards = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
    values = jnp.array([[.1, .2], [.3, .4], [.5, .6]])
    next_value = jnp.array([.7, .8])
    terminated = jnp.array([[False, False], [False, True], [True, False]])
    advantage, returns = generalized_advantage_estimate(
        rewards, values, next_value, terminated, gamma=.9, gae_lambda=.8
    )
    expected = jnp.zeros_like(rewards)
    carry = jnp.zeros(2)
    following = jnp.concatenate([values[1:], next_value[None]])
    for index in reversed(range(3)):
        nonterminal = 1 - terminated[index]
        delta = rewards[index] + .9 * following[index] * nonterminal - values[index]
        carry = delta + .9 * .8 * nonterminal * carry
        expected = expected.at[index].set(carry)
    assert jnp.allclose(advantage, expected)
    assert jnp.allclose(returns, expected + values)


def test_bc_and_dppo_compiled_updates_are_finite():
    policy, critic = models()
    states = jr.normal(jr.PRNGKey(2), (4, 1, 6))
    actions = jr.normal(jr.PRNGKey(3), (4, 4, 3))
    bc_state, bc_optimizer = make_bc_state(policy)
    bc_state, bc_loss = bc_update(
        bc_state, bc_optimizer, actions, states, jr.PRNGKey(4)
    )
    sample = bc_state.policy.sample(states, jr.PRNGKey(5))
    old_log_probs = bc_state.policy.transition_log_prob(states, sample.chain)
    values = critic.batch(states)
    batch = RolloutBatch(
        states, sample.chain, old_log_probs, jnp.arange(4, dtype=jnp.float32),
        jnp.arange(4, dtype=jnp.float32), values,
    )
    state, actor_optimizer, critic_optimizer = make_dppo_state(
        bc_state.policy, critic
    )
    state, metrics = dppo_update(
        state, actor_optimizer, critic_optimizer, batch, DPPOConfig()
    )
    assert jnp.isfinite(bc_loss)
    assert all(jnp.isfinite(value) for value in metrics)
    assert jnp.allclose(metrics.ratio, 1, atol=1e-5)
    assert jnp.allclose(metrics.approx_kl, 0, atol=1e-5)
    assert metrics.clip_fraction == 0


def test_dppo_freezes_source_and_updates_velocity():
    policy, critic = models()
    states = jr.normal(jr.PRNGKey(10), (4, 1, 6))
    sample = policy.sample(states, jr.PRNGKey(11))
    old_log_probs = policy.transition_log_prob(states, sample.chain)
    batch = RolloutBatch(
        states, sample.chain, old_log_probs,
        jnp.array([-1., -.2, .4, 2.]),
        jnp.array([1., 2., 3., 4.]),
        critic.batch(states),
    )
    state, actor_optimizer, critic_optimizer = make_dppo_state(policy, critic)
    updated, _ = dppo_update(
        state, actor_optimizer, critic_optimizer, batch, DPPOConfig()
    )
    assert eqx.tree_equal(state.policy.encoder, updated.policy.encoder)
    assert eqx.tree_equal(state.policy.inference_net, updated.policy.inference_net)
    old_velocity = jax.tree_util.tree_leaves(eqx.filter(state.policy.vel_net, eqx.is_array))
    new_velocity = jax.tree_util.tree_leaves(eqx.filter(updated.policy.vel_net, eqx.is_array))
    assert any(not jnp.allclose(old, new) for old, new in zip(old_velocity, new_velocity))


def test_loss_rejects_no_nonfinite_values():
    policy, critic = models()
    states = jnp.zeros((2, 1, 6))
    sample = policy.sample(states, jr.PRNGKey(20))
    old = policy.transition_log_prob(states, sample.chain)
    batch = RolloutBatch(
        states, sample.chain, old, jnp.array([0., 1.]), jnp.array([0., 1.]),
        critic.batch(states),
    )
    loss, metrics = dppo_loss(policy, critic, batch, DPPOConfig())
    assert jnp.isfinite(loss)
    assert all(jnp.isfinite(value) for value in metrics)
