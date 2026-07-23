import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bayesinf.applications.optimal_control import (
    AmortizedSeqFlow, BicycleConfig, ConstrainedLateralMPC, LinearLateralMPC,
    LateralKalmanFilter, generate_dataset,
)
from bayesinf.applications.optimal_control.amortized_seq_flow import control_aware_loss


def test_dataset_and_flow_shapes_and_constraints():
    cfg = BicycleConfig(obs_history=8, horizon=6)
    data = generate_dataset(jr.PRNGKey(0), 5, cfg)
    assert data["observations"].shape == (5, 8, 1)
    assert data["past_controls"].shape == (5, 8, 1)
    assert data["previous_steering"].shape == (5, 1)
    assert data["reference"].shape == (5, 6, 3)
    assert data["actions"].shape == (5, 6)
    model = AmortizedSeqFlow(cfg.horizon, cfg.delta_max, jr.PRNGKey(1), hidden_size=16,
                             condition_dim=8, n_coupling=2)
    loss = model.loss(data["actions"], data["observations"], data["reference"],
                      data["past_controls"], data["previous_steering"])
    samples = model.sample(data["observations"], data["reference"], jr.PRNGKey(2), 3,
                           past_controls=data["past_controls"],
                           previous_steering=data["previous_steering"])
    assert jnp.isfinite(loss)
    assert samples.shape == (3, 5, 6)
    assert jnp.all(jnp.abs(samples) <= cfg.delta_max)


def test_history_is_aligned_and_contains_recovery_states():
    cfg = BicycleConfig(obs_history=8, horizon=6, obs_std=0.0)
    data = generate_dataset(jr.PRNGKey(11), 128, cfg)
    # The final measurement is x_t after the final recorded past control.
    assert jnp.allclose(data["observations"][:, -1, 0], data["errors"][:, 0])
    assert jnp.allclose(
        data["previous_steering"][:, 0], data["past_controls"][:, -1, 0]
    )
    assert jnp.max(jnp.abs(data["errors"][:, 0])) > 0.8
    assert jnp.max(jnp.abs(data["errors"][:, 1])) > 0.18
    assert jnp.all(jnp.abs(data["actions"]) <= cfg.delta_max + 1e-6)


def test_control_aware_loss_is_finite():
    cfg = BicycleConfig(obs_history=8, horizon=6)
    data = generate_dataset(jr.PRNGKey(20), 5, cfg)
    model = AmortizedSeqFlow(
        cfg.horizon, cfg.delta_max, jr.PRNGKey(21), hidden_size=16,
        condition_dim=8, n_coupling=2,
    )
    total, components = control_aware_loss(
        model, data["actions"], data["observations"], data["reference"],
        data["past_controls"], data["previous_steering"], data["errors"],
        dt=cfg.dt, velocity=cfg.velocity, wheelbase=cfg.wheelbase,
    )
    assert jnp.isfinite(total)
    assert all(jnp.isfinite(value) for value in components)


def test_enlarged_internal_bound_keeps_applied_controls_bounded():
    cfg = BicycleConfig(obs_history=8, horizon=6)
    data = generate_dataset(jr.PRNGKey(30), 5, cfg)
    model = AmortizedSeqFlow(
        cfg.horizon, cfg.delta_max, jr.PRNGKey(31), hidden_size=16,
        condition_dim=8, n_coupling=2, saturation_factor=1.05,
    )
    samples = model.sample(
        data["observations"], data["reference"], jr.PRNGKey(32), 8,
        past_controls=data["past_controls"],
        previous_steering=data["previous_steering"],
    )
    assert jnp.all(jnp.abs(samples) <= cfg.delta_max)
    assert jnp.all(jnp.isfinite(model.log_prob(
        data["actions"], data["observations"], data["reference"],
        data["past_controls"], data["previous_steering"],
    )))


def test_mpc_kalman_is_bounded_and_estimates_heading():
    cfg = BicycleConfig(horizon=6)
    kf = LateralKalmanFilter(cfg)
    mpc = ConstrainedLateralMPC(cfg)
    estimate = kf.update(0.5)
    delta, chunk = mpc.command(estimate, np.full(cfg.horizon, 0.015))
    assert chunk.shape == (cfg.horizon,)
    assert np.all(np.abs(chunk) <= cfg.delta_max + 1e-12)
    before = kf.mean.copy()
    kf.predict(delta, 0.015)
    kf.update(0.45)
    assert np.all(np.isfinite(kf.mean))
    assert kf.mean[1] != before[1]


def test_linear_mpc_is_bounded():
    cfg = BicycleConfig(horizon=6)
    mpc = LinearLateralMPC(cfg)
    delta, chunk = mpc.command(np.array([0.5, 0.1]), np.full(cfg.horizon, 0.015))
    assert np.isfinite(delta)
    assert chunk.shape == (cfg.horizon,)
    assert np.all(np.abs(chunk) <= cfg.delta_max + 1e-12)
