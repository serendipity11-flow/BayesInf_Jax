import jax
import jax.numpy as jnp
import equinox as eqx
import jax.random as jr
from typing import List

# Helper for the linear block pattern: Linear -> BN -> ReLU -> Dropout
class LinearBlock(eqx.Module):
    linear: eqx.nn.Linear
    dropout: eqx.nn.Dropout
    use_dropout: bool

    def __init__(self, in_features, out_features, dropout_rate, key):
        k1, k2 = jr.split(key)
        self.linear = eqx.nn.Linear(in_features, out_features, key=k1)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)
        self.use_dropout = dropout_rate > 0

    def __call__(self, x, key=None, inference=False):
        x = self.linear(x)
        x = jax.nn.relu(x)
        if self.use_dropout and key is not None:
            x = self.dropout(x, key=key, inference=inference)
        return x

class Encoder(eqx.Module):
    lstm_cell: eqx.nn.LSTMCell
    linear_layers: List[LinearBlock]
    final_linear: eqx.nn.Linear
    linear_mu: eqx.nn.Linear
    linear_std: eqx.nn.Linear
    
    latent_dim_z: int
    theta_dim: int
    dropout_rate: float

    def __init__(self, latent_dim_z, n_time_series, lstm_hidden_dim, linear_hidden_dim,
                 n_lstm_layers=1, n_linear_layers=4, theta_dim=0, dropout=0.0, key=None):
        super().__init__()
        keys = jr.split(key, 5 + n_linear_layers)
        
        # NOTE: For simplicity, this implementation assumes a single layer LSTM as per common JAX patterns.
        # Multi-layer LSTMs in JAX require stacking cells manually in the scan loop.
        self.lstm_cell = eqx.nn.LSTMCell(n_time_series, lstm_hidden_dim, key=keys[0])
        
        self.linear_layers = []
        # First Linear Block
        self.linear_layers.append(
            LinearBlock(lstm_hidden_dim + theta_dim, linear_hidden_dim, dropout, keys[1])
        )
        
        # Subsequent Linear Blocks
        current_dim = linear_hidden_dim
        if n_linear_layers > 1:
            for i in range(1, n_linear_layers - 1):
                next_dim = linear_hidden_dim // (2**i)
                self.linear_layers.append(
                    LinearBlock(current_dim, next_dim, dropout, keys[2+i])
                )
                current_dim = next_dim
        
        self.final_linear = eqx.nn.Linear(current_dim, latent_dim_z * 2, key=keys[-3])
        self.linear_mu = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-2])
        self.linear_std = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-1])
        
        self.theta_dim = theta_dim
        self.latent_dim_z = latent_dim_z
        self.dropout_rate = dropout

    def __call__(self, x, state, theta=None, key=None, inference=False):
        # x shape: (seq_len, n_time_series)
        
        # LSTM Scan
        def scan_fn(carry, input_t):
            h, c = carry
            h_new, c_new = self.lstm_cell(input_t, (h, c))
            return (h_new, c_new), h_new

        # Initial hidden state (zeros)
        h0 = jnp.zeros((self.lstm_cell.hidden_size,))
        c0 = jnp.zeros((self.lstm_cell.hidden_size,))
        
        # Run LSTM
        (h_final, _), _ = jax.lax.scan(scan_fn, (h0, c0), x)
        
        eh1 = h_final # Take last hidden state
        
        if self.theta_dim > 0 and theta is not None:
            eh1 = jnp.concatenate([eh1, theta], axis=-1)
            
        # Linear Block Forward
        out = eh1
        keys = jr.split(key, len(self.linear_layers)) if key is not None else [None] * len(self.linear_layers)
        
        for layer, k in zip(self.linear_layers, keys):
            out, state = layer(out, state, key=k, inference=inference)
            
        eh2 = self.final_linear(out)
        mu = self.linear_mu(eh2)
        logvar = self.linear_std(eh2)
        
        return mu, logvar, state

class Regressor(eqx.Module):
    linear_layers: List[LinearBlock]
    final_linear: eqx.nn.Linear
    linear_mu: eqx.nn.Linear
    linear_std: eqx.nn.Linear
    
    def __init__(self, latent_dim_z, theta_dim, linear_hidden_dim, n_linear_layers=4, dropout=0.0, key=None):
        super().__init__()
        keys = jr.split(key, 5 + n_linear_layers)
        
        self.linear_layers = []
        # First Linear Block
        self.linear_layers.append(
            LinearBlock(theta_dim, linear_hidden_dim, dropout, keys[0])
        )
        
        current_dim = linear_hidden_dim
        if n_linear_layers > 1:
            for i in range(1, n_linear_layers - 1):
                next_dim = linear_hidden_dim // (2**i)
                self.linear_layers.append(
                    LinearBlock(current_dim, next_dim, dropout, keys[1+i])
                )
                current_dim = next_dim

        self.final_linear = eqx.nn.Linear(current_dim, latent_dim_z * 2, key=keys[-3])
        self.linear_mu = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-2])
        self.linear_std = eqx.nn.Linear(latent_dim_z * 2, latent_dim_z, key=keys[-1])

    def __call__(self, theta, state, key=None, inference=False):
        out = theta
        keys = jr.split(key, len(self.linear_layers)) if key is not None else [None] * len(self.linear_layers)
        
        for layer, k in zip(self.linear_layers, keys):
            out, state = layer(out, state, key=k, inference=inference)
            
        out = self.final_linear(out)
        mu = self.linear_mu(out)
        logvar = self.linear_std(out)
        return mu, logvar, state

class Decoder(eqx.Module):
    latent_expand: eqx.nn.Linear
    lstm_cell: eqx.nn.LSTMCell
    linear: eqx.nn.Linear
    theta_dim: int
    latent_dim_z: int
    seq_len: int
    
    def __init__(self, latent_dim_z, seq_len, n_time_series, lstm_hidden_dim, n_lstm_layers, theta_dim=0, dropout=0.0, key=None):
        super().__init__()
        k1, k2, k3 = jr.split(key, 3)
        
        self.latent_expand = eqx.nn.Linear(latent_dim_z + theta_dim, (latent_dim_z + theta_dim) * seq_len, key=k1)
        # Using LSTMCell for manual loop
        self.lstm_cell = eqx.nn.LSTMCell(latent_dim_z + theta_dim, lstm_hidden_dim, key=k2)
        self.linear = eqx.nn.Linear(lstm_hidden_dim, n_time_series, key=k3)
        
        self.theta_dim = theta_dim
        self.latent_dim_z = latent_dim_z
        self.seq_len = seq_len

    def __call__(self, z, theta=None):
        if self.theta_dim > 0 and theta is not None:
            z = jnp.concatenate([z, theta], axis=-1)
        
        # Expand and reshape to (seq_len, input_dim)
        z_expanded = self.latent_expand(z)
        z_expanded = jax.nn.relu(z_expanded)
        
        # View/Reshape: (input_dim * seq_len) -> (seq_len, input_dim)
        z_seq = z_expanded.reshape(self.seq_len, self.latent_dim_z + self.theta_dim)
        
        # LSTM Scan
        def scan_fn(carry, input_t):
            h, c = carry
            h_new, c_new = self.lstm_cell(input_t, (h, c))
            return (h_new, c_new), h_new # Return hidden state at each step

        h0 = jnp.zeros((self.lstm_cell.hidden_size,))
        c0 = jnp.zeros((self.lstm_cell.hidden_size,))
        
        _, dh1_seq = jax.lax.scan(scan_fn, (h0, c0), z_seq)
        
        # Apply final linear layer to all time steps. 
        # vmap over the time dimension (axis 0)
        xhat = jax.vmap(self.linear)(dh1_seq)
        
        return xhat

class PELS_VAE(eqx.Module):
    encoder: Encoder
    regressor: Regressor
    decoder: Decoder

    def __init__(self, theta_dim, latent_dim_z, seq_len, n_time_series, lstm_hidden_dim, linear_hidden_dim,
                 n_lstm_layers=1, n_linear_layers=4, dropout=0.0, key=None):
        super().__init__()
        k_enc, k_reg, k_dec = jr.split(key, 3)
        
        self.encoder = Encoder(latent_dim_z, n_time_series, lstm_hidden_dim, linear_hidden_dim,
                               n_lstm_layers, n_linear_layers, theta_dim, dropout, key=k_enc)
        self.regressor = Regressor(latent_dim_z, theta_dim, linear_hidden_dim,
                                   n_linear_layers, dropout, key=k_reg)
        self.decoder = Decoder(latent_dim_z, seq_len, n_time_series, lstm_hidden_dim, n_lstm_layers,
                               theta_dim, dropout, key=k_dec)

    def reparameterize(self, mu, logvar, key):
        std = jnp.exp(0.5 * logvar)
        eps = jr.normal(key, mu.shape)
        return mu + eps * std

    def __call__(self, x, state, theta=None, key=None, inference=False):
        """
        Forward pass for a SINGLE sample. 
        Use jax.vmap(model, axis_name='batch')(x_batch, state, theta_batch, keys) for batch processing.
        """
        k_enc, k_reg, k_sample = jr.split(key, 3)
        
        mu, logvar, state = self.encoder(x, state, theta=theta, key=k_enc, inference=inference)
        muhat, logvarhat, state = self.regressor(theta, state, key=k_reg, inference=inference)
        
        z = self.reparameterize(mu, logvar, k_sample)
        xhat = self.decoder(z, theta=theta)
        
        return xhat, mu, logvar, muhat, logvarhat, z, state

    def predict(self, theta, state, key=None, passes=1):
        """
        Predict xhat from theta.
        """
        muhat, logvarhat, state = self.regressor(theta, state, key=key, inference=True)
        
        def single_pass(k):
            z = self.reparameterize(muhat, logvarhat, k)
            return self.decoder(z, theta=theta)
            
        keys = jr.split(key, passes)
        xhats = jax.vmap(single_pass)(keys)
        xhat = jnp.mean(xhats, axis=0)
        
        return xhat

# Example of how to initialize and use
if __name__ == '__main__':
    # Mock config variables
    THETA_DIM = 5
    LATENT_DIM_Z = 10
    SEQ_LEN = 50
    N_TIME_SERIES = 3
    LSTM_HIDDEN = 64
    LINEAR_HIDDEN = 128
    
    key = jr.PRNGKey(0)
    model = PELS_VAE(THETA_DIM, LATENT_DIM_Z, SEQ_LEN, N_TIME_SERIES, LSTM_HIDDEN, LINEAR_HIDDEN, key=key)
    
    # Initialize State (required for BatchNorm)
    # Equinox < 0.10 style managed state manually, but eqx.nn.BatchNorm uses eqx.nn.State.
    # We must initialize the state container.
    state = eqx.nn.State(model)

    # Dummy Data (Batch size 32)
    x_batch = jnp.zeros((32, SEQ_LEN, N_TIME_SERIES))
    theta_batch = jnp.zeros((32, THETA_DIM))
    
    # Forward Pass with VMAP
    # We vmap over the batch dimension (axis 0). 
    # axis_name='batch' is required for BatchNorm to compute statistics across the batch.
    
    def forward_batch(model, state, x, theta, key):
        keys = jr.split(key, x.shape[0])
        return jax.vmap(model, in_axes=(0, None, 0, 0, None), out_axes=(0, 0, 0, 0, 0, 0, None), axis_name='batch')(
            x, state, theta, keys, False
        )

    # Execute
    key, subkey = jr.split(key)
    # Returns (xhat, mu, logvar, muhat, logvarhat, z, NEW_STATE)
    out = forward_batch(model, state, x_batch, theta_batch, subkey)
    xhat_batch = out[0]
    new_state = out[-1]
    
    print(f"Output shape: {xhat_batch.shape}")