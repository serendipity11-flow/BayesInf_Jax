import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.flow_sde import RealNVPFlowSDE


def test_control_is_masked_only_from_diffusion_context():
    model = RealNVPFlowSDE(
        d_x=4,
        d_y=7,
        d_u=3,
        d_c=8,
        d_hidden_enc=8,
        d_hidden=16,
        d_hidden_diffusion=8,
        time_emb_dim=8,
        exclude_control_from_diffusion=True,
        key=jr.PRNGKey(0),
    )
    observations = jr.normal(jr.PRNGKey(1), (6, 4))
    controls_a = jnp.zeros((6, 3))
    controls_b = jnp.ones((6, 3))
    y_a = jnp.concatenate((observations, controls_a), axis=-1)
    y_b = jnp.concatenate((observations, controls_b), axis=-1)

    diffusion_a = model.encode_diffusion_stride(y_a, 1)
    diffusion_b = model.encode_diffusion_stride(y_b, 1)
    drift_a = model.encode_observation_stride(y_a, 1)
    drift_b = model.encode_observation_stride(y_b, 1)

    assert jnp.array_equal(diffusion_a, diffusion_b)
    assert not jnp.allclose(drift_a, drift_b)


def test_control_remains_in_diffusion_context_by_default():
    model = RealNVPFlowSDE(
        d_x=4,
        d_y=7,
        d_u=3,
        d_c=8,
        d_hidden_enc=8,
        d_hidden=16,
        d_hidden_diffusion=8,
        time_emb_dim=8,
        key=jr.PRNGKey(2),
    )
    observations = jr.normal(jr.PRNGKey(3), (6, 4))
    y_a = jnp.concatenate((observations, jnp.zeros((6, 3))), axis=-1)
    y_b = jnp.concatenate((observations, jnp.ones((6, 3))), axis=-1)

    assert not jnp.allclose(
        model.encode_diffusion_stride(y_a, 1),
        model.encode_diffusion_stride(y_b, 1),
    )
