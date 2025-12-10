import os
from numba import cuda

if cuda.is_available():
    import jax
else:
    os.environ["JAX_PLATFORMS"] = 'cpu'
    import jax

    jax.default_device('cpu')
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

class TimeSeriesEncoder(eqx.Module):
    """Encodes a time series into a single feature vector."""
    cell: eqx.Module
    l1: eqx.Module
    l2: eqx.Module

    def __init__(self, in_size: int, hidden_size: int, out_size: int, key: jr.PRNGKey, rnn='gru'):
        key_cell, key_1, key_2 = jr.split(key, 3)
        if rnn == 'gru':
            self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=key_cell)
        else:
            self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=key_cell)
        self.l1 = eqx.nn.Linear(hidden_size, hidden_size, key=key_1)
        self.l2 = eqx.nn.Linear(hidden_size, out_size, key=key_2)

    def __rnn_call__(self, x: jax.Array):
        # x is (Time, Features)
        # h0 initialization: jnp.zeros is fine for recurrent models
        def f(carry, inp):
            return self.cell(inp, carry), None

        h0 = jnp.zeros(self.cell.hidden_size)
        if isinstance(self.cell, eqx.nn.LSTMCell):
            c0 = jnp.zeros(self.cell.hidden_size)
            final_states, _ = jax.lax.scan(f, (h0, c0), x)
            final_state = final_states[0]
        else:
            final_state, _ = jax.lax.scan(f, h0, x)
        return final_state

    def __call__(self, x: jax.Array):

        h = jax.vmap(self.__rnn_call__)(x)
        h = jax.nn.elu(h)
        h = jax.vmap(self.l1)(h)
        h = jax.nn.elu(h)
        h = jax.vmap(self.l2)(h)
        return h