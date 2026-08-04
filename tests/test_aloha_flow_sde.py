import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control.controlled_flow_sde import (
    ControlledRealNVPFlowSDE,
)
from experiments.mujoco_pusher_optimal_control.geometry import (
    pusher_contact_features,
)


def small_model():
    return ControlledRealNVPFlowSDE(
        4, 3, jr.PRNGKey(0), hidden_size=8, depth=1,
        diffusion_hidden=8, flow_layers=2, time_emb_dim=4, dt=0.02,
    )


def test_controlled_flow_sde_loss_is_finite_with_noisy_inputs():
    model = small_model()
    x0 = jr.normal(jr.PRNGKey(1), (5, 4))
    u = jr.normal(jr.PRNGKey(2), (5, 3))
    x1 = x0 + 0.01 * jr.normal(jr.PRNGKey(3), (5, 4))
    t = jnp.arange(5) * 0.02
    assert jnp.isfinite(model.transition_nll(x0, u, x1, t))
    assert jnp.isfinite(model.noisy_transition_nll(
        x0, u, x1, t, jr.PRNGKey(4), 0.01, 0.01
    ))


def test_controlled_flow_sde_rollout_shapes_and_repeatability():
    model = small_model()
    x0, controls = jnp.zeros(4), jnp.zeros((6, 3))
    nominal = model.deterministic_rollout(x0, controls)
    samples = model.sample_rollout(x0, controls, jr.PRNGKey(5), n_particles=3)
    repeated = model.sample_rollout(x0, controls, jr.PRNGKey(5), n_particles=3)
    assert nominal.shape == (6, 4)
    assert samples.shape == (3, 6, 4)
    assert jnp.allclose(samples, repeated)
    assert jnp.all(jnp.isfinite(samples))


def test_multiple_shooting_pathwise_loss_is_finite():
    model = small_model()
    states = jr.normal(jr.PRNGKey(10), (2, 9, 4)) * 0.05
    controls = jr.normal(jr.PRNGKey(11), (2, 8, 3)) * 0.05
    loss = model.multiple_shooting_loss(
        states, controls, jnp.array([0, 20]), jr.PRNGKey(12),
        segment_length=4, n_particles=2, solver_name="heun",
    )
    assert jnp.isfinite(loss)


def test_diffusion_can_exclude_control_while_drift_uses_it():
    model = ControlledRealNVPFlowSDE(
        4, 3, jr.PRNGKey(20), hidden_size=8, depth=1,
        diffusion_hidden=8, flow_layers=2, time_emb_dim=4,
        diffusion_condition_on_control=False,
    )
    assert not model.diffusion_condition_on_control
    assert model._diffusion_context(jnp.ones(3)).shape == (0,)
    states = jr.normal(jr.PRNGKey(21), (2, 5, 4)) * 0.05
    controls = jr.normal(jr.PRNGKey(22), (2, 4, 3)) * 0.05
    loss = model.multiple_shooting_loss(
        states, controls, jnp.array([0, 8]), jr.PRNGKey(23),
        segment_length=4, n_particles=2,
    )
    assert jnp.isfinite(loss)


def test_object_drift_has_no_direct_control_path():
    model = ControlledRealNVPFlowSDE(
        12, 3, jr.PRNGKey(30), hidden_size=8, depth=1,
        diffusion_hidden=8, flow_layers=2, time_emb_dim=4,
        object_drift_no_control=True,
    )
    state = jr.normal(jr.PRNGKey(31), (12,))
    drift_zero = model.drift(state, jnp.zeros(3), jnp.asarray(0.0))
    drift_one = model.drift(state, jnp.ones(3), jnp.asarray(0.0))
    arm = jnp.asarray([0, 1, 2, 6, 7, 8])
    obj = jnp.asarray([3, 4, 5, 9, 10, 11])
    assert not jnp.allclose(drift_zero[arm], drift_one[arm])
    assert jnp.allclose(drift_zero[obj], drift_one[obj])


def test_pusher_geometry_distance_and_proximity_are_hand_checked():
    # Straight arm tip is at x=-0.72+0.34+0.30+0.22=0.14.  With an object
    # centred at x=0.50, its left face is x=0.41, hence surface gap
    # 0.41-0.14-0.055=0.215 m.
    far = jnp.asarray([0., 0., 0., .50, 0., 0., 0., 0., 0., 0., 0., 0.])
    features = pusher_contact_features(far, temperature=0.02)
    assert jnp.allclose(features[0], 0.215, atol=1e-6)
    assert features[2] < 1e-4
    assert jnp.allclose(features[3:], jnp.asarray([-1., 0.]))

    # Put the left cuboid face exactly at the tip centre: overlap equals the
    # pusher radius and the smooth proximity is close to one.
    overlap = far.at[3].set(0.23)
    overlap_features = pusher_contact_features(overlap, temperature=0.02)
    assert jnp.allclose(overlap_features[0], -0.055, atol=1e-6)
    assert overlap_features[2] > 0.9
    assert jnp.all(jnp.isfinite(jax.jacrev(pusher_contact_features)(overlap)))


def test_split_heads_and_geometry_support_drift_and_diffusion_rollout():
    model = ControlledRealNVPFlowSDE(
        12, 3, jr.PRNGKey(40), hidden_size=8, depth=1,
        diffusion_hidden=8, flow_layers=2, time_emb_dim=4,
        separate_dynamics_heads=True, use_contact_geometry=True,
        geometry_feature_fn=pusher_contact_features,
    )
    drift = model.drift(jnp.zeros(12), jnp.zeros(3), jnp.asarray(0.0))
    paths = model.sample_rollout(
        jnp.zeros(12), jnp.zeros((2, 3)), jr.PRNGKey(41), n_particles=2
    )
    assert drift.shape == (12,)
    assert paths.shape == (2, 2, 12)
    assert jnp.all(jnp.isfinite(paths))


def test_coupled_six_state_flows_roll_out_and_object_depends_on_robot_flow():
    model = ControlledRealNVPFlowSDE(
        12, 3, jr.PRNGKey(50), hidden_size=8, depth=1,
        diffusion_hidden=8, flow_layers=2, time_emb_dim=4,
        coupled_six_state_flows=True,
    )
    state = jr.normal(jr.PRNGKey(51), (12,))
    drift = model.drift(state, jnp.ones(3), jnp.asarray(0.0))
    paths = model.sample_rollout(
        state, jnp.zeros((2, 3)), jr.PRNGKey(52), n_particles=2
    )
    assert drift.shape == (12,)
    assert paths.shape == (2, 2, 12)
    assert jnp.all(jnp.isfinite(paths))
    assert model.drift_arm.net.in_size != model.drift_object.net.in_size
