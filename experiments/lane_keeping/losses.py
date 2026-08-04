"""Lane-specific objectives for the generic amortized sequence flow."""

from typing import Tuple

import jax
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control.amortized_seq_flow import AmortizedSeqFlow


def control_aware_loss(
    model: AmortizedSeqFlow, actions, observations, reference, past_controls,
    previous_steering, current_state, *, dt: float, velocity: float,
    wheelbase: float, first_action_weight: float = 10.0,
    rollout_weight: float = 1.0,
) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """NLL plus first-action imitation and nonlinear bicycle rollout loss."""
    nll = model.loss(actions, observations, reference, past_controls,
                     previous_steering)
    prediction = model.sample(
        observations, reference, jr.PRNGKey(0), deterministic=True,
        past_controls=past_controls, previous_steering=previous_steering,
    )[0]
    first_action_mse = jnp.mean((prediction[:, 0] - actions[:, 0]) ** 2)
    curvature = reference[..., 2]

    def rollout_step(carry, inputs):
        ey, ep, last_control = carry
        delta, kappa = inputs
        ey = ey + dt * velocity * jnp.sin(ep)
        ep_delta = dt * (velocity / wheelbase * jnp.tan(delta) - velocity * kappa)
        ep = jnp.arctan2(jnp.sin(ep + ep_delta), jnp.cos(ep + ep_delta))
        cost = ey**2 + 0.3 * ep**2 + 0.02 * delta**2 + 0.1 * (delta - last_control)**2
        return (ey, ep, delta), cost

    _, costs = jax.lax.scan(
        rollout_step,
        (current_state[:, 0], current_state[:, 1], previous_steering.reshape(-1)),
        (jnp.swapaxes(prediction, 0, 1), jnp.swapaxes(curvature, 0, 1)),
    )
    rollout_loss = jnp.mean(costs)
    total = nll + first_action_weight * first_action_mse + rollout_weight * rollout_loss
    return total, (nll, first_action_mse, rollout_loss)
