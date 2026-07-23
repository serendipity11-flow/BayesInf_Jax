import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control.controlled_flow_sde import (
    ControlledRealNVPFlowSDE,
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
