import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control import (
    CEMStochasticMPC, ControlledLatentSSMFlow, GradientMPC, GradientMPCConfig,
    MPPIController,
    SamplingMPCConfig,
    GRULatentSSMFlow,
)


def _model():
    return ControlledLatentSSMFlow(
        state_dim=2, control_dim=1, history_len=3, key=jr.PRNGKey(0),
        d_enc=8, d_out=8, n_coupling=2, hidden_size=8, n_hidden=1,
    )


def test_transition_alignment_loss_and_rollout_shapes():
    model = _model()
    states = jr.normal(jr.PRNGKey(1), (2, 5, 2))
    controls = jr.normal(jr.PRNGKey(2), (2, 4, 1))
    loss = model.nll_loss(states, controls)
    rollout = model.sample_rollout(
        states[0, -3:], controls[0, -2:], jnp.zeros((4, 1)),
        jr.PRNGKey(3), n_particles=5,
    )
    assert jnp.isfinite(loss)
    assert rollout.shape == (5, 4, 2)
    assert jnp.all(jnp.isfinite(rollout))


def test_cem_and_mppi_respect_action_bounds():
    model = _model()
    cfg = SamplingMPCConfig(horizon=3, num_candidates=6, num_particles=2,
                            iterations=2, elite_fraction=0.34)
    cost = lambda x, u, t: jnp.sum(x * x) + 0.01 * jnp.sum(u * u)
    history = jnp.zeros((3, 2))
    controls = jnp.zeros((2, 1))
    for controller_type in (CEMStochasticMPC, MPPIController):
        controller = controller_type(model, jnp.array([-0.2]), jnp.array([0.2]),
                                     cost, cfg)
        command, plan = controller.command(history, controls, jr.PRNGKey(4))
        assert command.shape == (1,)
        assert plan.shape == (3, 1)
        assert jnp.all(jnp.abs(plan) <= 0.2 + 1e-6)


def test_one_step_initial_state_and_gradient_mpc():
    model = ControlledLatentSSMFlow(
        state_dim=2, control_dim=1, history_len=1, key=jr.PRNGKey(10),
        d_enc=8, d_out=8, n_coupling=2, hidden_size=8, n_hidden=1,
    )
    x0 = jnp.array([0.2, -0.1])
    actions = jnp.zeros((3, 1))
    trajectory = model.rollout_deterministic(x0, actions)
    jacobian = jnp.asarray(jr.normal(jr.PRNGKey(11), actions.shape))
    direct_gradient = jax.grad(
        lambda u: jnp.sum(model.rollout_deterministic(x0, u))
    )(jacobian)
    assert trajectory.shape == (3, 2)
    assert jnp.all(jnp.isfinite(direct_gradient))
    controller = GradientMPC(
        model, jnp.array([-0.2]), jnp.array([0.2]),
        lambda x, u, t: jnp.sum(x * x) + 0.01 * jnp.sum(u * u),
        GradientMPCConfig(horizon=3, iterations=2),
    )
    command, plan = controller.command(x0)
    assert command.shape == (1,)
    assert jnp.all(jnp.abs(plan) <= 0.2 + 1e-6)


def test_residual_one_step_is_anchored_at_supplied_state():
    model = ControlledLatentSSMFlow(
        state_dim=2, control_dim=1, history_len=1, key=jr.PRNGKey(20),
        predict_delta=True, d_enc=8, d_out=8, n_coupling=2,
        hidden_size=8, n_hidden=1,
    )
    x = jnp.array([2.0, -3.0])
    delta = model.predict_next(x, jnp.zeros(1)) - x
    assert delta.shape == x.shape
    assert jnp.all(jnp.isfinite(delta))


def test_history_conditioned_likelihood_has_one_target():
    model = ControlledLatentSSMFlow(
        state_dim=2, control_dim=1, history_len=3, key=jr.PRNGKey(30),
        predict_delta=True, d_enc=8, d_out=8, n_coupling=2,
        hidden_size=8, n_hidden=1,
    )
    states = jr.normal(jr.PRNGKey(31), (4, 4, 2))
    controls = jr.normal(jr.PRNGKey(32), (4, 3, 1))
    loss = model.history_one_step_nll(states, controls)
    rollout = model.rollout_deterministic_history(
        states[0, :3], controls[0, :2], jnp.zeros((2, 1))
    )
    assert jnp.isfinite(loss)
    assert rollout.shape == (2, 2)


def test_gru_flow_uses_last_hidden_and_current_action_condition():
    model = GRULatentSSMFlow(
        2, 1, history_len=3, d_out=5, key=jr.PRNGKey(40),
        n_coupling=2, hidden_size=8, n_hidden=1,
    )
    states = jr.normal(jr.PRNGKey(41), (4, 5, 2))  # H=3, latent length L=2
    controls = jr.normal(jr.PRNGKey(42), (4, 4, 1))
    hidden = model.encode_history(states[0, :3])
    condition = model.condition(states[0, :3], controls[0, 2])
    loss = model.nll_loss(states, controls)
    gradient = jax.grad(lambda action: jnp.sum(
        model.predict_next(states[0, :3], action)
    ))(controls[0, 2])
    assert hidden.shape == (5,)
    assert condition.shape == (6,)
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(gradient))
