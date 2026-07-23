import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control.controlled_flow_sde import (
    ControlledRealNVPFlowSDE,
)
from bayesinf.applications.optimal_control.flow_sde_mppi import (
    FlowSDEMPPIController,
    flow_sde_candidate_costs,
)
from bayesinf.applications.optimal_control.sampling_mpc import SamplingMPCConfig


def test_flow_sde_mppi_uses_solver_rollouts_and_returns_bounded_action():
    model = ControlledRealNVPFlowSDE(
        4, 3, jr.PRNGKey(0), hidden_size=8, depth=1,
        diffusion_hidden=8, flow_layers=2, time_emb_dim=4, dt=0.02,
    )
    def cost(state, action, step):
        del step
        return jnp.mean(state ** 2) + 0.01 * jnp.mean(action ** 2)
    config = SamplingMPCConfig(
        horizon=3, num_candidates=4, num_particles=2, iterations=1,
        initial_std=0.1, temperature=1.0, normalize_costs=True,
    )
    controller = FlowSDEMPPIController(
        model, -jnp.ones(3), jnp.ones(3), cost, config, solver_name="heun"
    )
    action, plan = controller.command(jnp.zeros(4), jr.PRNGKey(1), start_step=7)
    assert action.shape == (3,)
    assert plan.shape == (3, 3)
    assert jnp.all(action >= -1) and jnp.all(action <= 1)


def test_candidate_cost_shape():
    model = ControlledRealNVPFlowSDE(
        2, 2, jr.PRNGKey(2), hidden_size=4, depth=1,
        diffusion_hidden=4, flow_layers=2, time_emb_dim=4,
    )
    candidates = jnp.zeros((3, 2, 2))
    costs = flow_sde_candidate_costs(
        model, jnp.zeros(2), candidates, jr.split(jr.PRNGKey(3), 3),
        5, 2, lambda x, u, t: jnp.sum(x ** 2) + jnp.sum(u ** 2),
        0.1, "heun",
    )
    assert costs.shape == (3,)
    assert jnp.all(jnp.isfinite(costs))
