from types import SimpleNamespace

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from experiments.cond_stoch_interpolant_dppo.train_jax import (
    ChunkVectorEnv,
    collect_rollout,
)
from bayesinf.applications.optimal_control import InterpolantSample


class DummyEnv:
    def __init__(self, terminate_at):
        self.terminate_at = terminate_at
        self.steps = 0
        self.resets = 0

    def reset(self, seed=None):
        self.steps = 0
        self.resets += 1
        return np.array([100 * self.resets], dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        done = self.steps >= self.terminate_at
        return np.array([self.steps], dtype=np.float32), 1.0, done, False, {}

    def close(self):
        pass


class DummyPolicy:
    def sample(self, states, key):
        batch = len(states)
        return InterpolantSample(
            jnp.zeros((batch, 4, 1)),
            jnp.zeros((batch, 3, 4, 1)),
        )

    def transition_log_prob(self, states, chains):
        return jnp.zeros((len(states), 2, 4, 1))


class DummyCritic:
    def batch(self, states):
        return jnp.zeros(len(states))


class IdentityNormalizer:
    def observation(self, value):
        return np.asarray(value, dtype=np.float32)

    def action(self, value):
        return np.asarray(value, dtype=np.float32)


def test_rollout_preserves_history_and_masks_chunk_tail_after_done():
    vector = ChunkVectorEnv([lambda: DummyEnv(1), lambda: DummyEnv(10)])
    observations, _ = vector.reset([0, 1])
    history = np.repeat(observations[:, None], 3, axis=1)
    args = SimpleNamespace(
        n_envs=2, rollout_steps=1, horizon=4, cond_steps=3,
        execution_horizon=4, reward_scale=1.0, gamma=.99, gae_lambda=.95,
    )
    batch, following_history, _, mean_reward = collect_rollout(
        DummyPolicy(), DummyCritic(), vector, history, IdentityNormalizer(),
        args, jr.PRNGKey(0),
        lambda policy, states, key: policy.sample(states, key),
        lambda policy, states, chain: policy.transition_log_prob(states, chain),
        lambda critic, states: critic.batch(states),
    )
    assert batch.states.shape == (2, 3, 1)
    # Environment zero terminates on the first action and must not receive the
    # remaining three actions from the old plan.
    assert vector.envs[0].steps == 0  # reset at the chunk boundary
    assert vector.envs[0].resets == 2
    # The new episode history is cleared to its reset observation.
    assert np.all(following_history[0] == 200)
    # Environment one remains live for all four actions and accumulates them.
    assert vector.envs[1].steps == 4
    assert np.all(following_history[1, :, 0] == np.array([2, 3, 4]))
    assert np.isclose(mean_reward, 2.5)
    vector.close()
