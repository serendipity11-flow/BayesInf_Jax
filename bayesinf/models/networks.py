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
from typing import Union, Sequence

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


class StackedBidirectionalLSTMEncoder(eqx.Module):
    """Two-direction stacked LSTM summary encoder for BayesFlow compatibility."""
    forward_cells: tuple
    backward_cells: tuple
    projection1: eqx.nn.Linear
    projection2: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, in_size, hidden_size, out_size, key, n_layers=2):
        keys = jr.split(key, 2 * n_layers + 2)
        forward, backward = [], []
        layer_in = in_size
        for layer in range(n_layers):
            forward.append(eqx.nn.LSTMCell(layer_in, hidden_size, key=keys[2 * layer]))
            backward.append(eqx.nn.LSTMCell(layer_in, hidden_size, key=keys[2 * layer + 1]))
            layer_in = 2 * hidden_size
        self.forward_cells, self.backward_cells = tuple(forward), tuple(backward)
        self.projection1 = eqx.nn.Linear(2 * hidden_size, out_size, key=keys[-2])
        self.projection2 = eqx.nn.Linear(out_size, out_size, key=keys[-1])
        self.hidden_size = hidden_size

    def _direction(self, cell, sequence):
        initial = (jnp.zeros(self.hidden_size), jnp.zeros(self.hidden_size))
        def step(carry, value):
            carry = cell(value, carry)
            return carry, carry[0]
        final, outputs = jax.lax.scan(step, initial, sequence)
        return final[0], outputs

    def _encode_one(self, sequence):
        values = sequence
        final_forward = final_backward = None
        for forward, backward in zip(self.forward_cells, self.backward_cells):
            final_forward, out_forward = self._direction(forward, values)
            final_backward, out_backward_reversed = self._direction(backward, values[::-1])
            values = jnp.concatenate((out_forward, out_backward_reversed[::-1]), axis=-1)
        summary = jnp.concatenate((final_forward, final_backward), axis=-1)
        return self.projection2(jax.nn.elu(self.projection1(summary)))

    def __call__(self, x):
        return jax.vmap(self._encode_one)(x)
