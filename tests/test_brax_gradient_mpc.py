import jax.numpy as jnp
import jax.random as jr
import pytest

from bayesinf.applications.optimal_control.gradient_based_mpc import (
    BraxGradientMPC,
    GradientMPCConfig,
)


def test_brax_gradient_mpc_differentiates_environment_rollout():
    pytest.importorskip("brax", reason="Brax is an optional simulator dependency")
    from brax import envs

    env = envs.get_environment("inverted_pendulum", backend="generalized")
    state = env.reset(jr.PRNGKey(0))
    controller = BraxGradientMPC(
        env,
        lambda next_state, action, time: (
            jnp.sum(next_state.obs ** 2) + 1e-3 * jnp.sum(action ** 2)
        ),
        GradientMPCConfig(horizon=3, iterations=2, learning_rate=0.02),
    )
    command, plan = controller.command(state)
    assert command.shape == (env.action_size,)
    assert plan.shape == (3, env.action_size)
    assert jnp.all(jnp.isfinite(plan))
    assert jnp.all(jnp.abs(plan) <= 1.0 + 1e-6)
