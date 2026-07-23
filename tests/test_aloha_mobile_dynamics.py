import numpy as np
import jax.numpy as jnp
import jax.random as jr

from bayesinf.applications.optimal_control.gru_latent_ssm_flow import GRULatentSSMFlow
from bayesinf.applications.optimal_control.train_aloha_mobile_dynamics import (
    make_segments,
    split_episode_indices,
)


def test_segment_alignment_and_episode_split():
    episodes = [np.arange(60, dtype=np.float32).reshape(10, 6) + 100 * i
                for i in range(10)]
    actions = [np.arange(54, dtype=np.float32).reshape(9, 6) for _ in episodes]
    train, validation, test = split_episode_indices(10, seed=0)
    assert len(set(train) & set(validation)) == 0
    assert len(set(train) & set(test)) == 0
    zeros, ones = jnp.zeros(6), jnp.ones(6)
    x, u = make_segments(episodes, actions, train[:1], history=3,
                         segment_steps=2, stride=2, x_mean=zeros, x_std=ones,
                         u_mean=zeros, u_std=ones)
    assert x.shape[1:] == (5, 6)
    assert u.shape[1:] == (4, 6)


def test_full_segment_loss_and_full_horizon_shape():
    model = GRULatentSSMFlow(
        state_dim=4, control_dim=4, history_len=3, d_out=8,
        key=jr.PRNGKey(0), n_coupling=2, hidden_size=8, n_hidden=1,
    )
    states = jr.normal(jr.PRNGKey(1), (2, 7, 4))
    controls = jr.normal(jr.PRNGKey(2), (2, 6, 4))
    assert jnp.isfinite(model.nll_loss(states, controls))
    rollout = model.rollout_deterministic(states[0, :3], controls[0, 2:])
    assert rollout.shape == (4, 4)
    samples = model.sample_rollout(
        states[0, :3], controls[0, :2], controls[0, 2:],
        jr.PRNGKey(7), n_particles=2,
    )
    assert samples.shape == (2, 4, 4)


def test_nsf_segment_loss_and_rollout_are_finite():
    model = GRULatentSSMFlow(
        state_dim=4, control_dim=4, history_len=3, d_out=8,
        key=jr.PRNGKey(3), n_coupling=2, hidden_size=8, n_hidden=1,
        flow_type="nsf", num_bins=4,
    )
    states = jr.normal(jr.PRNGKey(4), (2, 6, 4))
    controls = jr.normal(jr.PRNGKey(5), (2, 5, 4))
    loss = model.nll_loss(states, controls)
    rollout = model.rollout_deterministic(states[0, :3], controls[0, 2:])
    assert jnp.isfinite(loss)
    assert jnp.all(jnp.isfinite(rollout))
    noisy_loss = model.noisy_context_nll_loss(
        states, controls, jr.PRNGKey(6), noise_std=0.05
    )
    assert jnp.isfinite(noisy_loss)
