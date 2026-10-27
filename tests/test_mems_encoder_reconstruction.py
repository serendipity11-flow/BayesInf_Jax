import jax.numpy as jnp
import jax.random as jr
import pytest

from bayesinf.models import networks

if not hasattr(networks, "StackedBidirectionalLSTMAutoencoder"):
    pytest.skip(
        "This regression targets an unreleased encoder variant absent from the public snapshot",
        allow_module_level=True,
    )

StackedBidirectionalLSTMAutoencoder = networks.StackedBidirectionalLSTMAutoencoder


def test_stacked_bidirectional_lstm_autoencoder_shape():
    model = StackedBidirectionalLSTMAutoencoder(
        in_size=4, hidden_size=8, latent_size=6, key=jr.PRNGKey(0),
        n_layers=2, sequence_length=96,
    )
    signals = jnp.zeros((3, 96, 4))
    assert model.encoder(signals).shape == (3, 6)
    assert model(signals).shape == signals.shape
