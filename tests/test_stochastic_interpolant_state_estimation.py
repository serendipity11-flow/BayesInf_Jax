import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.state_estimation.stochastic_interpolant import (
    StochasticInterpolantSE,
)


def _model(source_type):
    return StochasticInterpolantSE(
        d_x=4,
        d_y=2,
        d_c=8,
        d_hidden=16,
        n_heads=4,
        depth=1,
        time_emb_dim=8,
        source_type=source_type,
        key=jr.PRNGKey(0),
    )


def test_inference_network_uses_observations_only():
    model = _model("inferred")
    y = jnp.zeros((5, 2))
    mu, log_sigma = model.inference_net(y)
    assert mu.shape == (5, 4)
    assert log_sigma.shape == (5, 4)
    assert model.inference_net.fc1.in_features == 2
    assert model.vel_net.condition_on_mu


def test_gaussian_source_has_no_inference_network_or_mu_condition():
    model = _model("gaussian")
    y = jnp.zeros((5, 2))
    x = jnp.zeros((5, 4))
    assert model.inference_net is None
    assert not model.vel_net.condition_on_mu
    assert not model.score_net.condition_on_mu
    assert model.vel_net.input_proj.in_features == 4 + 8 + 2 * 8
    assert jnp.isfinite(model.loss_single(x, y, jr.PRNGKey(1)))
