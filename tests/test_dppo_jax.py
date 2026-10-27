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
from bayesinf.applications.optimal_control.dppo_diffusion_jax import (
    DPPODiffusionPolicy,
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


def test_interpolant_analytic_score_and_ode_sde_are_finite():
    policy = CondStochInterpolantPolicy(
        6, 3, 4, jr.PRNGKey(39), integration_steps=3,
        hidden_size=16, depth=1, time_dim=8,
    )
    states = jr.normal(jr.PRNGKey(40), (4, 1, 6))
    actions = jr.normal(jr.PRNGKey(41), (4, 4, 3))
    total, velocity, path = policy.loss_components(
        actions, states, jr.PRNGKey(42)
    )
    assert jnp.isfinite(total)
    assert jnp.isfinite(velocity)
    assert path == 0
    assert jnp.allclose(total, velocity)

    ode = policy.sample_ode(states, jr.PRNGKey(43))
    sde = policy.sample_sde(states, jr.PRNGKey(44))
    expected_chain = (4, policy.integration_steps + 1, 4, 3)
    assert ode.actions.shape == sde.actions.shape == (4, 4, 3)
    assert ode.chain.shape == sde.chain.shape == expected_chain
    assert jnp.all(jnp.isfinite(ode.actions))
    assert jnp.all(jnp.isfinite(sde.actions))
    log_prob = policy.transition_log_prob(states, sde.chain)
    assert log_prob.shape == (4, policy.integration_steps, 4, 3)
    assert jnp.all(jnp.isfinite(log_prob))

    context = policy._features(states[0])
    score = policy._actual_score(actions[0], context, jnp.asarray(0.0))
    assert jnp.allclose(score, -actions[0], atol=1e-5)


def test_interpolant_energy_path_loss_and_heun_samplers():
    policy = CondStochInterpolantPolicy(
        6, 3, 4, jr.PRNGKey(46), integration_steps=3,
        hidden_size=16, depth=1, time_dim=8,
        path_loss_weight=0.1, path_loss_steps=2,
    )
    states = jr.normal(jr.PRNGKey(47), (2, 1, 6))
    actions = jr.normal(jr.PRNGKey(48), (2, 4, 3))
    total, velocity, path = policy.loss_components(
        actions, states, jr.PRNGKey(49)
    )
    assert jnp.isfinite(total) and jnp.isfinite(path)
    assert jnp.allclose(total, velocity + 0.1 * path)
    ode = policy.sample_ode_heun(states, jr.PRNGKey(50))
    sde = policy.sample_sde_heun(states, jr.PRNGKey(51))
    assert ode.chain.shape == sde.chain.shape == (2, 4, 4, 3)
    assert jnp.all(jnp.isfinite(ode.actions))
    assert jnp.all(jnp.isfinite(sde.actions))


def test_diffusion_bc_updates_network_but_not_analytical_schedule():
    policy = DPPODiffusionPolicy(6, 3, 4, jr.PRNGKey(30), denoising_steps=5)
    states = jr.normal(jr.PRNGKey(31), (4, 1, 6))
    actions = jr.normal(jr.PRNGKey(32), (4, 4, 3))
    state, optimizer = make_bc_state(policy)
    updated, loss = bc_update(
        state, optimizer, actions, states, jr.PRNGKey(33)
    )

    assert jnp.isfinite(loss)
    for name in (
        "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
        "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
        "sqrt_recip_alphas_cumprod", "sqrt_recipm1_alphas_cumprod",
        "posterior_coef1", "posterior_coef2", "posterior_log_variance",
    ):
        assert jnp.array_equal(getattr(policy, name), getattr(updated.policy, name))
    old_network = jax.tree_util.tree_leaves(eqx.filter(policy.network, eqx.is_array))
    new_network = jax.tree_util.tree_leaves(
        eqx.filter(updated.policy.network, eqx.is_array)
    )
    assert any(not jnp.array_equal(old, new)
               for old, new in zip(old_network, new_network))


def test_diffusion_ddim_supports_strided_inference_grid():
    policy = DPPODiffusionPolicy(6, 3, 4, jr.PRNGKey(34), denoising_steps=20)
    states = jnp.zeros((2, 1, 6))
    sample = policy.sample_ddim(states, jr.PRNGKey(35), inference_steps=5)
    assert sample.actions.shape == (2, 4, 3)
    assert sample.chain.shape == (2, 6, 4, 3)
    assert jnp.all(jnp.isfinite(sample.actions))


def test_diffusion_bc_sampler_has_noise_free_final_transition():
    policy = DPPODiffusionPolicy(
        6, 3, 4, jr.PRNGKey(36), denoising_steps=5,
        sampling_std_floor=0.1,
    )
    states = jnp.zeros((2, 1, 6))
    sample = policy.sample_bc(states, jr.PRNGKey(37), clip=None)
    final_mean, _ = jax.vmap(policy._mean_log_variance)(
        sample.chain[:, -2], jnp.zeros(2, dtype=jnp.int32), states
    )
    assert jnp.allclose(sample.chain[:, -1], final_mean)
    assert jnp.allclose(sample.actions, final_mean)


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
