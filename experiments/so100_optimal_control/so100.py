"""Configuration and costs for the concrete 5-DoF SO-100 experiment."""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class SO100Config:
    """Joint-space defaults; replace bounds with calibrated hardware limits.

    State is ``[q(5), dq(5)]`` and action is a 5-vector joint command (velocity,
    torque, or position increment according to the data-collection interface).
    The model remains valid for another state encoding if ``state_dim`` is
    changed consistently during training.
    """
    dof: int = 5
    dt: float = 0.05
    history_len: int = 8
    horizon: int = 20

    @property
    def state_dim(self):
        return 2 * self.dof


def joint_tracking_cost(q_reference, action_weight: float = 1e-2,
                        velocity_weight: float = 5e-2):
    """Return a vectorized MPC stage cost for a fixed joint reference."""
    q_reference = jnp.asarray(q_reference)
    dof = q_reference.shape[0]

    def cost(state, action, time_index):
        del time_index
        q, dq = state[:dof], state[dof:2 * dof]
        return (jnp.sum((q - q_reference) ** 2)
                + velocity_weight * jnp.sum(dq ** 2)
                + action_weight * jnp.sum(action ** 2))
    return cost
