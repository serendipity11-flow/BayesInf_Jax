"""Milestone 6 controlled rollout and common benchmark-interface tests."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from bayesinf.applications.optimal_control.controlled_flow_sde import (
    ControlledRealNVPFlowSDE,
)
from bayesinf.applications.state_estimation import (
    BenchmarkTrainingConfig,
    ControlledDuffingOscillator,
    ControlledStochasticPendulum,
    NSFTransitionAdapter,
    NeuralStochasticFlow,
    NormFlowSDETransitionAdapter,
    count_trainable_parameters,
    ensemble_transition_metrics,
    fit_normalization,
    make_transition_pairs,
    sample_controlled_duffing_data,
    sample_controlled_pendulum_data,
    train_transition_kernel_nll,
)


def _small_kernels(key=jax.random.key(1)):
    nsf_key, flow_key = jax.random.split(key)
    state_mean, state_std = jnp.array([0.3, -0.2]), jnp.array([1.5, 0.7])
    control_mean, control_std = jnp.array([0.1]), jnp.array([1.2])
    nsf = NSFTransitionAdapter(NeuralStochasticFlow(
        2,
        1,
        hidden_size=6,
        hidden_depth=1,
        num_coupling_layers=2,
        state_mean=state_mean,
        state_std=state_std,
        control_mean=control_mean,
        control_std=control_std,
        key=nsf_key,
    ))
    flow_model = ControlledRealNVPFlowSDE(
        2,
        1,
        flow_key,
        hidden_size=6,
        depth=1,
        diffusion_hidden=6,
        flow_layers=2,
        time_emb_dim=4,
        dt=0.1,
    )
    flow = NormFlowSDETransitionAdapter(
        flow_model,
        state_mean=state_mean,
        state_std=state_std,
        control_mean=control_mean,
        control_std=control_std,
    )
    return nsf, flow


def test_controlled_duffing_dataset_is_reproducible_and_nonlinear():
    process = ControlledDuffingOscillator(integration_substeps=3)
    first = sample_controlled_duffing_data(jax.random.key(2), process, 5, 7)
    second = sample_controlled_duffing_data(jax.random.key(2), process, 5, 7)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    assert first["states"].shape == (5, 8, 2)
    assert first["controls"].shape == (5, 7, 1)
    positive = process.drift(jnp.array([1.5, 0.0]), jnp.array([1.0]))
    negative = process.drift(jnp.array([1.5, 0.0]), jnp.array([-1.0]))
    assert float(positive[1]) > float(negative[1])
    # Cubic restoring force prevents this benchmark from collapsing to a
    # linear controlled Gaussian state-space model.
    assert not jnp.isclose(
        process.drift(jnp.array([2.0, 0.0]), jnp.zeros(1))[1],
        2.0 * process.drift(jnp.array([1.0, 0.0]), jnp.zeros(1))[1],
    )


def test_controlled_pendulum_data_and_drift_match_physics():
    process = ControlledStochasticPendulum(
        damping=0.3, integration_substeps=3
    )
    data = sample_controlled_pendulum_data(
        jax.random.key(200), process, 4, 6, delta_t=0.04
    )
    assert data["states"].shape == (4, 7, 2)
    assert data["controls"].shape == (4, 6, 1)
    np.testing.assert_allclose(
        data["times"][:, 1:] - data["times"][:, :-1], 0.04, atol=1e-7
    )
    downward = process.drift(jnp.array([0.0, 0.0]), jnp.array([0.0]))
    displaced = process.drift(jnp.array([jnp.pi / 2, 0.0]), jnp.array([0.0]))
    driven = process.drift(jnp.array([0.0, 0.0]), jnp.array([2.0]))
    np.testing.assert_allclose(downward, jnp.zeros(2), atol=1e-7)
    np.testing.assert_allclose(displaced[1], -9.81, rtol=1e-6)
    np.testing.assert_allclose(driven[1], 2.0, rtol=1e-6)


def test_nsf_piecewise_constant_rollout_matches_manual_recursion():
    nsf, _ = _small_kernels()
    model = nsf.model
    key = jax.random.key(3)
    initial = jnp.array([0.2, -0.4])
    controls = jnp.array([[0.1], [-0.3], [0.5]])
    gaps = jnp.array([0.1, 0.2, 0.15])
    actual = eqx.filter_jit(model.sample_rollout)(
        key, initial, gaps, controls, num_samples=5, t_start=0.7
    )
    particles = jnp.broadcast_to(initial, (5, 2))
    paths = []
    start = 0.7
    for step_key, gap, control in zip(jax.random.split(key, 3), gaps, controls):
        particles = model.sample(
            step_key,
            particles,
            gap,
            control,
            num_samples=1,
            t_current=start,
        )[0]
        paths.append(particles)
        start += float(gap)
    expected = jnp.stack(paths, axis=1)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    assert actual.shape == (5, 3, 2)


def test_common_adapters_have_matching_physical_unit_shapes_and_jit():
    nsf, flow = _small_kernels(jax.random.key(4))
    current = jnp.array([[0.2, -0.1], [0.5, 0.7]])
    control = jnp.array([[0.3], [-0.2]])
    for index, kernel in enumerate((nsf, flow)):
        samples = eqx.filter_jit(kernel.sample)(
            jax.random.key(10 + index),
            current,
            0.1,
            control,
            num_samples=4,
        )
        log_prob = eqx.filter_jit(kernel.log_prob)(
            samples[0], current, 0.1, control
        )
        rollout = eqx.filter_jit(kernel.sample_rollout)(
            jax.random.key(20 + index),
            current[0],
            0.1,
            control,
            num_samples=4,
        )
        assert samples.shape == (4, 2, 2)
        assert log_prob.shape == (2,)
        assert rollout.shape == (4, 2, 2)
        assert jnp.isfinite(samples).all() and jnp.isfinite(log_prob).all()
    assert not nsf.supports_direct_long_gap
    assert not flow.supports_direct_long_gap


def test_normflowsde_adapter_density_has_physical_jacobian():
    _, adapter = _small_kernels(jax.random.key(5))
    current = jnp.array([0.4, -0.3])
    following = jnp.array([0.45, -0.1])
    control = jnp.array([0.2])
    normalized_current = (current - adapter.state_mean) / adapter.state_std
    normalized_following = (following - adapter.state_mean) / adapter.state_std
    normalized_control = (control - adapter.control_mean) / adapter.control_std
    normalized = adapter.model.transition_log_prob(
        normalized_following,
        normalized_current,
        normalized_control,
        jnp.asarray(0.0),
    )
    expected = normalized - jnp.sum(jnp.log(adapter.state_std))
    actual = adapter.log_prob(following, current, 0.1, control)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        adapter.model.transition_nll(
            normalized_current,
            normalized_control,
            normalized_following,
            jnp.asarray(0.0),
        ),
        -normalized / 2,
        rtol=1e-6,
        atol=1e-6,
    )


def test_matched_trainer_updates_both_kernel_types_on_same_data():
    process = ControlledDuffingOscillator(integration_substeps=2)
    data = sample_controlled_duffing_data(jax.random.key(6), process, 8, 5,
                                          delta_t=0.1)
    state_mean, state_std, control_mean, control_std = fit_normalization(
        data["states"], data["controls"]
    )
    nsf, flow = _small_kernels(jax.random.key(7))
    # Rebuild only to use the data normalizers in both adapters.
    nsf = NSFTransitionAdapter(NeuralStochasticFlow(
        2, 1, hidden_size=6, hidden_depth=1, num_coupling_layers=2,
        state_mean=state_mean, state_std=state_std,
        control_mean=control_mean, control_std=control_std,
        key=jax.random.key(8),
    ))
    flow = NormFlowSDETransitionAdapter(
        flow.model,
        state_mean=state_mean,
        state_std=state_std,
        control_mean=control_mean,
        control_std=control_std,
    )
    transitions = make_transition_pairs(
        data["states"], data["times"], data["controls"]
    )
    transitions["t_current"] = data["times"][:, :-1].reshape(-1)
    config = BenchmarkTrainingConfig(steps=2, batch_size=8, seed=9)
    for kernel in (nsf, flow):
        trained, losses = train_transition_kernel_nll(kernel, transitions, config)
        assert losses.shape == (2,) and jnp.isfinite(losses).all()
        assert count_trainable_parameters(trained) > 0


def test_ensemble_metrics_match_identical_reference_distribution():
    samples = jnp.array([
        [[-1.0], [0.0]],
        [[0.0], [1.0]],
        [[1.0], [2.0]],
        [[2.0], [3.0]],
    ])
    target = jnp.array([[0.5], [1.5]])
    metrics = ensemble_transition_metrics(
        samples, target, reference_samples=samples
    )
    np.testing.assert_allclose(metrics["marginal_wasserstein"], 0.0, atol=1e-7)
    np.testing.assert_allclose(metrics["covariance_error"], 0.0, atol=1e-7)
    assert all(bool(jnp.isfinite(value)) for value in metrics.values())
