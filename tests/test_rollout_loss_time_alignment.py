from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE


def test_rollout_loss_uses_initial_time_grid():
    # Exercise the real loss with a small solver double, without mutating a
    # frozen Equinox module or constructing an unrelated neural network.
    model = SimpleNamespace(time_step=1.0, observation_stride=1)

    def encode_observation_stride(y, stride):
        return jnp.zeros((y.shape[0], 1), dtype=y.dtype)

    recorded = {}

    def integrate(z0, ts, c, key, solver_name, *, c_diffusion):
        recorded["ts"] = ts
        return ts[:, None]

    model.encode_observation_stride = encode_observation_stride
    model.encode_diffusion_stride = encode_observation_stride
    model._integrate_stochastic_solver = integrate

    x = jnp.arange(4, dtype=jnp.float32)[:, None]
    y = jnp.zeros((4, 1), dtype=jnp.float32)
    key = jr.PRNGKey(0)

    loss = RealNVPFlowSDE.rollout_loss_single(
        model,
        x,
        y,
        key,
        solver_name="heun",
        use_custom_solver=True,
        loss_type="mse",
    )

    assert recorded["ts"].shape[0] == x.shape[0]
    assert recorded["ts"][0] == 0.0
    assert jnp.array_equal(recorded["ts"], jnp.arange(4, dtype=x.dtype))
    assert loss == 0.0
