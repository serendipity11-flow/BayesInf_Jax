import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from bayesinf.applications.optimal_control import (
    AnalyticScorePolicy,
    FlowMatchingPolicy,
    LearnedScorePolicy,
    MLPContextEncoder,
    ScoreBasedPolicy,
)


def small_policy(policy_class=FlowMatchingPolicy, **kwargs):
    return policy_class(
        obs_dim=6, action_dim=3, horizon=4, cond_steps=2,
        integration_steps=3, hidden_size=16, depth=2, time_dim=8,
        context_dim=8, n_heads=4, encoder_type="mlp", key=jr.PRNGKey(0),
        **kwargs,
    )


def batch():
    states = jr.normal(jr.PRNGKey(1), (5, 2, 6))
    actions = jnp.tanh(jr.normal(jr.PRNGKey(2), (5, 4, 3)))
    return states, actions


def test_all_policies_use_fixed_gaussian_source_and_mlp_context():
    states, _ = batch()
    for cls in (FlowMatchingPolicy, AnalyticScorePolicy, LearnedScorePolicy):
        kwargs = ({"score_head_width": 8, "score_loss_weight": .05}
                  if cls is LearnedScorePolicy else {})
        policy = small_policy(cls, **kwargs)
        assert isinstance(policy.encoder, MLPContextEncoder)
        assert not hasattr(policy, "inference_net")
        mean, log_std = policy.base_params(states[0])
        assert jnp.allclose(mean, 0) and jnp.allclose(log_std, 0)


@pytest.mark.parametrize(
    "policy_class,kwargs",
    [(FlowMatchingPolicy, {}), (AnalyticScorePolicy, {}),
     (LearnedScorePolicy, {"score_head_width": 8, "score_loss_weight": .05})],
)
def test_standalone_policy_loss_and_sampler(policy_class, kwargs):
    policy = small_policy(policy_class, **kwargs)
    states, actions = batch()
    loss, grads = eqx.filter_value_and_grad(
        lambda model: model.loss(actions, states, jr.PRNGKey(3))
    )(policy)
    assert jnp.isfinite(loss)
    assert any(jnp.any(jnp.abs(leaf) > 0) for leaf in
               jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_array)))
    sample = (policy.sample_ode(states, jr.PRNGKey(4))
              if policy.score_mode == "flow"
              else policy.sample_sde(states, jr.PRNGKey(4)))
    assert sample.actions.shape == (5, 4, 3)
    assert sample.chain.shape == (5, 4, 4, 3)
    assert jnp.all(jnp.isfinite(sample.actions))


def test_score_network_presence_is_exact():
    assert small_policy(FlowMatchingPolicy).score_net is None
    assert small_policy(AnalyticScorePolicy).score_net is None
    assert small_policy(
        LearnedScorePolicy, score_head_width=8, score_loss_weight=.05
    ).score_net is not None


def test_learned_csi_uses_state_estimation_scaled_score_and_sde_schedule():
    policy = small_policy(
        LearnedScorePolicy, score_head_width=8, score_loss_weight=.25,
        sigma_0=1.0, score_sigma_floor=.05, sde_eta=.25,
    )
    states, actions = batch()
    total, velocity, score, path = policy.loss_components_with_score(
        actions, states, jr.PRNGKey(123)
    )
    assert jnp.isfinite(total) and jnp.isfinite(score)
    assert jnp.allclose(total, velocity + .25 * score + path)

    sample = policy.sample_sde(states, jr.PRNGKey(124), clip=None)
    log_prob = policy.transition_log_prob(states, sample.chain)
    assert jnp.all(jnp.isfinite(sample.actions))
    assert jnp.all(jnp.isfinite(log_prob))


def test_flow_policy_rejects_sde_sampling():
    states, _ = batch()
    with pytest.raises(ValueError, match="no score"):
        small_policy().sample_sde(states, jr.PRNGKey(5))


def test_flow_ppo_exploration_has_likelihood_without_a_score_model():
    states, _ = batch()
    policy = small_policy(FlowMatchingPolicy)
    sample = policy.sample(states, jr.PRNGKey(6))
    assert policy.score_net is None
    assert sample.chain.shape == (5, 4, 4, 3)
    assert jnp.all(jnp.isfinite(
        policy.transition_log_prob(states, sample.chain)
    ))


def test_score_based_policy_has_no_velocity_network():
    policy = ScoreBasedPolicy(
        6, 3, 4, jr.PRNGKey(7), cond_steps=2, denoising_steps=3,
        context_width=8, context_dim=8, hidden_size=16, depth=1,
    )
    states, actions = batch()
    assert not hasattr(policy, "vel_net")
    assert jnp.isfinite(policy.loss(actions, states, jr.PRNGKey(8)))
    sample = policy.sample(states, jr.PRNGKey(9))
    assert sample.chain.shape == (5, 4, 4, 3)
    assert jnp.all(jnp.isfinite(
        policy.transition_log_prob(states, sample.chain)
    ))
