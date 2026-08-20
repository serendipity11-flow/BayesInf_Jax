"""Milestone 4 tests for Chapman--Kolmogorov consistency training."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesinf.applications.state_estimation.neural_stochastic_flow import (
    NSFBridge,
    NSFConsistencyConfig,
    NSFTrainingConfig,
    NeuralStochasticFlow,
    OrnsteinUhlenbeck,
    fit_normalization,
    flow_consistency_loss,
    sample_ou_transition_data,
    sample_uniform_consistency_triplets,
    train_nsf_with_consistency,
)


def _models(state_dim=2):
    model = NeuralStochasticFlow(
        state_dim,
        hidden_size=10,
        hidden_depth=1,
        num_coupling_layers=2,
        key=jax.random.key(50),
    )
    bridge = NSFBridge.from_transition(
        model,
        hidden_size=10,
        hidden_depth=1,
        num_coupling_layers=2,
        key=jax.random.key(51),
    )
    return model, bridge


@pytest.mark.parametrize("mode", ["none", "one_to_two", "two_to_one", "bidirectional"])
def test_consistency_modes_are_finite_and_route_components(mode):
    model, bridge = _models()
    losses = flow_consistency_loss(
        model,
        bridge,
        jax.random.key(52),
        jnp.array([[0.1, -0.2], [0.5, 0.3]]),
        jnp.array([0.2, 0.3]),
        jnp.array([0.4, 0.2]),
        mode=mode,
        one_to_two_weight=2.0,
        two_to_one_weight=3.0,
        num_mc_samples=3,
    )
    assert all(bool(jnp.isfinite(value)) for value in losses)
    if mode not in {"one_to_two", "bidirectional"}:
        assert float(losses.one_to_two) == 0.0
    if mode not in {"two_to_one", "bidirectional"}:
        assert float(losses.two_to_one) == 0.0
    np.testing.assert_allclose(
        losses.total,
        2.0 * losses.one_to_two + 3.0 * losses.two_to_one,
    )


def test_one_to_two_estimator_matches_log_ratio_definition():
    model, bridge = _models(state_dim=1)
    key = jax.random.key(60)
    initial, before, after = jnp.array([0.3]), 0.2, 0.5
    direct_key, middle_key, _ = jax.random.split(key, 3)
    end = model.sample(direct_key, initial, before + after)[0]
    bridge_key = jax.random.split(middle_key, 1)[0]
    middle = bridge.sample(bridge_key, initial, end, before, after)[0]
    expected = (
        model.log_prob(end, initial, before + after)
        + bridge.log_prob(middle, initial, end, before, after)
        - model.log_prob(middle, initial, before)
        - model.log_prob(end, middle, after)
    )
    actual = flow_consistency_loss(
        model, bridge, key, initial, before, after, mode="one_to_two"
    )
    np.testing.assert_allclose(actual.one_to_two, expected, rtol=2e-6, atol=2e-6)


def test_two_to_one_estimator_matches_log_ratio_definition():
    model, bridge = _models(state_dim=1)
    key = jax.random.key(61)
    initial, before, after = jnp.array([-0.4]), 0.3, 0.4
    _, middle_key, second_key = jax.random.split(key, 3)
    middle = model.sample(middle_key, initial, before)[0]
    draw_key = jax.random.split(second_key, 1)[0]
    end = model.sample(draw_key, middle, after)[0]
    expected = (
        model.log_prob(middle, initial, before)
        + model.log_prob(end, middle, after)
        - bridge.log_prob(middle, initial, end, before, after)
        - model.log_prob(end, initial, before + after)
    )
    actual = flow_consistency_loss(
        model, bridge, key, initial, before, after, mode="two_to_one"
    )
    np.testing.assert_allclose(actual.two_to_one, expected, rtol=2e-6, atol=2e-6)


def test_consistency_gradients_are_finite_for_transition_and_bridge():
    model, bridge = _models()
    initial = jnp.array([[0.1, -0.2], [0.3, 0.5]])
    arguments = (jax.random.key(62), initial, jnp.array([0.2, 0.3]), jnp.array([0.4, 0.2]))
    model_loss, model_gradients = eqx.filter_value_and_grad(
        lambda candidate: flow_consistency_loss(candidate, bridge, *arguments).total
    )(model)
    bridge_loss, bridge_gradients = eqx.filter_value_and_grad(
        lambda candidate: flow_consistency_loss(model, candidate, *arguments).total
    )(bridge)
    assert jnp.isfinite(model_loss) and jnp.isfinite(bridge_loss)
    for gradients in (model_gradients, bridge_gradients):
        leaves = jax.tree_util.tree_leaves(gradients)
        assert leaves and all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def test_consistency_weight_ablation_improves_fixed_seed_objective():
    process = OrnsteinUhlenbeck(mean=0.3, kappa=0.8, diffusion=0.5)
    transitions = sample_ou_transition_data(
        jax.random.key(70), process, 512, 1, delta_t_min=0.1, delta_t_max=0.8
    )
    state_mean, state_std, _, _ = fit_normalization(
        jnp.concatenate((transitions["x_current"], transitions["x_next"]))
    )
    triplets = sample_uniform_consistency_triplets(
        jax.random.key(71),
        transitions["x_current"],
        total_time_min=0.2,
        total_time_max=0.8,
    )

    def build():
        model = NeuralStochasticFlow(
            1,
            hidden_size=12,
            hidden_depth=1,
            num_coupling_layers=2,
            state_mean=state_mean,
            state_std=state_std,
            key=jax.random.key(72),
        )
        bridge = NSFBridge.from_transition(
            model,
            hidden_size=12,
            hidden_depth=1,
            num_coupling_layers=2,
            key=jax.random.key(73),
        )
        return model, bridge

    training = NSFTrainingConfig(
        steps=10, batch_size=32, learning_rate=3e-4, seed=74
    )
    model, bridge = build()
    disabled = train_nsf_with_consistency(
        model,
        bridge,
        transitions,
        triplets,
        training,
        NSFConsistencyConfig(
            mode="none", consistency_weight=0.0, bridge_inner_steps=2
        ),
    )
    model, bridge = build()
    enabled = train_nsf_with_consistency(
        model,
        bridge,
        transitions,
        triplets,
        training,
        NSFConsistencyConfig(
            consistency_weight=10.0,
            bridge_inner_steps=2,
            bridge_learning_rate=1e-3,
            num_mc_samples=2,
        ),
    )
    evaluation = (
        jax.random.key(75),
        triplets["x_initial"][:32],
        triplets["delta_t_before"][:32],
        triplets["delta_t_after"][:32],
    )
    disabled_loss = flow_consistency_loss(
        disabled.model, disabled.bridge, *evaluation, num_mc_samples=32
    ).total
    enabled_loss = flow_consistency_loss(
        enabled.model, enabled.bridge, *evaluation, num_mc_samples=32
    ).total
    assert float(enabled_loss) < float(disabled_loss) - 0.1
    assert set(enabled.history) == {
        "total", "nll", "flow", "one_to_two", "two_to_one", "bridge"
    }
    assert all(values.shape == (training.steps,) for values in enabled.history.values())
    assert enabled.model_opt_state is not enabled.bridge_opt_state


def test_controlled_consistency_is_rejected_until_control_path_is_defined():
    model = NeuralStochasticFlow(1, control_dim=1, key=jax.random.key(80))
    with pytest.raises(ValueError, match="control-path"):
        NSFBridge.from_transition(model, key=jax.random.key(81))
