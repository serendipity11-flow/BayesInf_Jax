"""Train and evaluate the BayesInf JAX stochastic-interpolant policy.

Defaults are transcribed from DPPO's Kitchen MLP configs.  The update formulas
live in ``bayesinf.applications.optimal_control.dppo_jax``; this file handles
DPPO-format NPZ data, Gymnasium action-chunk rollouts, checkpoints, and CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from bayesinf.applications.optimal_control import (
    CondStochInterpolantPolicy,
    DPPODiffusionPolicy,
    DPPOConfig,
    RolloutBatch,
    ValueCritic,
    bc_update,
    dppo_update,
    generalized_advantage_estimate,
    make_bc_state,
    make_dppo_state,
)


class Chunks:
    """NumPy-only equivalent of DPPO's ``StitchedSequenceDataset``."""

    def __init__(self, path: Path, horizon: int, cond_steps: int):
        data = np.load(path, allow_pickle=False)
        self.states, self.actions = data["states"], data["actions"]
        self.horizon, self.cond_steps = horizon, cond_steps
        self.indices = []
        offset = 0
        for length in data["traj_lengths"].astype(int):
            self.indices.extend(
                (offset + within, within) for within in range(length - horizon + 1)
            )
            offset += length

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        start, within = self.indices[index]
        observations = np.stack([
            self.states[start - min(within, distance)]
            for distance in reversed(range(self.cond_steps))
        ])
        return self.actions[start:start + self.horizon], observations


def create_policy(args, key):
    if args.policy == "dppo":
        return DPPODiffusionPolicy(
            args.obs_dim, args.action_dim, args.horizon, key,
            cond_steps=args.cond_steps,
            denoising_steps=args.denoising_steps,
            time_dim=args.time_dim,
            sampling_std_floor=args.sampling_std,
            logprob_std_floor=args.sampling_std,
        )
    return CondStochInterpolantPolicy(
        args.obs_dim, args.action_dim, args.horizon, key,
        cond_steps=args.cond_steps,
        integration_steps=args.integration_steps,
        informative_base=not args.gaussian_base,
        sigma_0=args.sigma_0,
        sampling_std=args.sampling_std,
        hidden_size=args.hidden_size,
        depth=args.depth,
        time_dim=args.time_dim,
        context_dim=args.context_dim,
        n_heads=args.n_heads,
    )


def batches(dataset, batch_size, key):
    order = np.asarray(jr.permutation(key, len(dataset)))
    for start in range(0, len(order) - batch_size + 1, batch_size):
        items = [dataset[int(i)] for i in order[start:start + batch_size]]
        yield (
            jnp.asarray(np.stack([item[0] for item in items])),
            jnp.asarray(np.stack([item[1] for item in items])),
        )


def append_csv(path: Path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists: writer.writeheader()
        writer.writerow(row)


def save_policy(path: Path, policy, args, critic=None):
    path.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path / "policy.eqx", policy)
    if critic is not None:
        eqx.tree_serialise_leaves(path / "critic.eqx", critic)
    config = {key: value for key, value in vars(args).items()
              if isinstance(value, (str, int, float, bool)) or value is None}
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")


def load_policy(path: Path, args, key):
    policy = create_policy(args, key)
    return eqx.tree_deserialise_leaves(path / "policy.eqx", policy)


def pretrain(args):
    train = Chunks(args.data / "train.npz", args.horizon, args.cond_steps)
    val = Chunks(args.data / "val.npz", args.horizon, args.cond_steps)
    key = jr.PRNGKey(args.seed)
    key, model_key = jr.split(key)
    state, optimizer = make_bc_state(
        create_policy(args, model_key), args.learning_rate, args.weight_decay
    )
    # A fixed random subset is representative without making validation curves
    # noisy or repeatedly selecting correlated chunks from trajectory zero.
    validation_rng = np.random.default_rng(args.seed + 10_000)
    n_val = min(args.batch_size, len(val))
    validation_indices = validation_rng.choice(len(val), n_val, replace=False)
    val_items = [val[int(index)] for index in validation_indices]
    val_actions = jnp.asarray(np.stack([item[0] for item in val_items]))
    val_states = jnp.asarray(np.stack([item[1] for item in val_items]))
    iterator = iter(())
    start = time.perf_counter()
    previous_log_time = start
    for step in range(1, args.gradient_steps + 1):
        try:
            actions, observations = next(iterator)
        except StopIteration:
            key, shuffle_key = jr.split(key)
            iterator = iter(batches(train, args.batch_size, shuffle_key))
            actions, observations = next(iterator)
        key, update_key = jr.split(key)
        state, loss = bc_update(
            state, optimizer, actions, observations, update_key
        )
        if step == 1 or step % args.log_every == 0 or step == args.gradient_steps:
            now = time.perf_counter()
            key, val_key = jr.split(key)
            val_loss = state.policy.loss(val_actions, val_states, val_key)
            append_csv(args.output / "pretrain_metrics.csv", {
                "gradient_step": step, "train_loss": float(loss),
                "validation_loss": float(val_loss),
                "wall_seconds": time.perf_counter() - start,
                "interval_seconds": now - previous_log_time,
                "updates_per_second": (
                    (1 if step == 1 else min(args.log_every, step - 1))
                    / max(now - previous_log_time, 1e-9)
                ),
                "nfe": state.policy.nfe, "seed": args.seed,
                "policy": args.policy,
            })
            print(
                step, float(loss), float(val_loss),
                "wall_seconds", now - start,
                "updates_per_second",
                (1 if step == 1 else min(args.log_every, step - 1))
                / max(now - previous_log_time, 1e-9),
                flush=True,
            )
            previous_log_time = now
    save_policy(args.output / "checkpoint", state.policy, args)


class Normalizer:
    def __init__(self, path):
        values = np.load(path)
        self.values = {key: values[key] for key in values.files}

    def observation(self, value):
        if "obs_mean" in self.values:
            return ((value - self.values["obs_mean"]) / self.values["obs_std"]).astype(np.float32)
        return (2 * ((value - self.values["obs_min"]) /
                     (self.values["obs_max"] - self.values["obs_min"] + 1e-6) - .5)).astype(np.float32)

    def action(self, value):
        return ((value + 1) * .5 * (
            self.values["action_max"] - self.values["action_min"]
        ) + self.values["action_min"]).astype(np.float32)


class ChunkVectorEnv:
    """Synchronous environments with explicit chunk-boundary autoresets.

    Once an environment finishes, later actions from the same chunk are
    masked.  The environment is reset only after the chunk, preventing the tail
    of an old plan from being applied to a new episode.
    """

    def __init__(self, factories):
        self.envs = [factory() for factory in factories]
        self.observations = None

    def reset(self, seeds):
        results = [
            env.reset(seed=int(seed)) for env, seed in zip(self.envs, seeds)
        ]
        self.observations = np.stack([result[0] for result in results])
        return self.observations.copy(), [result[1] for result in results]

    def step_masked(self, actions, active):
        rewards = np.zeros(len(self.envs), dtype=np.float32)
        terminated = np.zeros(len(self.envs), dtype=bool)
        truncated = np.zeros(len(self.envs), dtype=bool)
        infos = [{} for _ in self.envs]
        for index, env in enumerate(self.envs):
            if not active[index]:
                continue
            observation, reward, term, trunc, info = env.step(actions[index])
            self.observations[index] = observation
            rewards[index] = reward
            terminated[index], truncated[index], infos[index] = term, trunc, info
        return self.observations.copy(), rewards, terminated, truncated, infos

    def reset_done(self, done, seeds=None):
        for index, env in enumerate(self.envs):
            if done[index]:
                kwargs = {} if seeds is None else {"seed": int(seeds[index])}
                self.observations[index], _ = env.reset(**kwargs)
        return self.observations.copy()

    def close(self):
        for env in self.envs:
            env.close()


def make_envs(args):
    import gymnasium as gym
    if "Adroit" in args.env:
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
    def factory(index):
        def make():
            kwargs = {}
            if getattr(args, "max_episode_steps", None) is not None:
                kwargs["max_episode_steps"] = args.max_episode_steps
            env = gym.make(args.env, **kwargs)
            env.reset(seed=args.seed + index)
            return env
        return make
    return ChunkVectorEnv([factory(i) for i in range(args.n_envs)])


def collect_rollout(
    policy, critic, envs, history, normalizer, args, key,
    sample_policy, transition_log_prob, critic_values,
):
    states, chains, log_probs, rewards, terminated, values = [], [], [], [], [], []
    for _ in range(args.rollout_steps):
        state = jnp.asarray(history)
        key, sample_key = jr.split(key)
        sample = sample_policy(policy, state, sample_key)
        old_log_prob = transition_log_prob(policy, state, sample.chain)
        chunk_reward = np.zeros(args.n_envs, dtype=np.float32)
        chunk_done = np.zeros(args.n_envs, dtype=bool)
        active = np.ones(args.n_envs, dtype=bool)
        for action_step in range(args.horizon):
            raw_action = normalizer.action(np.asarray(sample.actions[:, action_step]))
            stepped = active.copy()
            observation, reward, term, trunc, _ = envs.step_masked(
                raw_action, active
            )
            chunk_reward += reward
            step_done = term | trunc
            chunk_done |= step_done
            # Only live environments contribute a new history observation.
            normalized = normalizer.observation(observation)
            history[stepped] = np.concatenate(
                [history[stepped, 1:], normalized[stepped, None]], axis=1
            )
            active &= ~step_done
            if not np.any(active):
                break
        states.append(state); chains.append(sample.chain)
        log_probs.append(old_log_prob); rewards.append(chunk_reward)
        terminated.append(chunk_done); values.append(critic_values(critic, state))
        # Reset completed environments at the action-chunk boundary and clear
        # their history so the next policy call sees only the new episode.
        if np.any(chunk_done):
            observation = envs.reset_done(chunk_done)
            normalized = normalizer.observation(observation)
            history[chunk_done] = np.repeat(
                normalized[chunk_done, None], args.cond_steps, axis=1
            )
    following = jnp.asarray(history)
    next_value = critic_values(critic, following)
    reward_array = jnp.asarray(np.stack(rewards))
    value_array = jnp.stack(values)
    terminal_array = jnp.asarray(np.stack(terminated))
    advantage, returns = generalized_advantage_estimate(
        reward_array, value_array, next_value, terminal_array,
        gamma=args.gamma, gae_lambda=args.gae_lambda,
    )
    flatten = lambda value: value.reshape((-1,) + value.shape[2:])
    return (
        RolloutBatch(
            flatten(jnp.stack(states)),
            flatten(jnp.stack(chains)),
            flatten(jnp.stack(log_probs)),
            advantage.reshape(-1),
            returns.reshape(-1),
            value_array.reshape(-1),
        ),
        history,
        key,
        float(jnp.mean(reward_array)),
    )


def finetune(args):
    key = jr.PRNGKey(args.seed)
    key, policy_key, critic_key = jr.split(key, 3)
    policy = load_policy(args.checkpoint, args, policy_key)
    critic = ValueCritic(
        args.obs_dim, args.cond_steps, critic_key,
        hidden_size=args.critic_hidden_size, depth=args.critic_depth,
    )
    state, actor_optimizer, critic_optimizer = make_dppo_state(
        policy, critic, args.actor_learning_rate, args.critic_learning_rate,
        args.max_grad_norm,
    )
    config = DPPOConfig(
        gamma=args.gamma, gae_lambda=args.gae_lambda,
        gamma_denoising=args.gamma_denoising,
        clip_ploss_coef=args.clip,
        clip_ploss_coef_base=args.clip_base,
        clip_ploss_coef_rate=args.clip_rate,
        vf_coef=args.vf_coef, target_kl=args.target_kl,
        reward_horizon=args.horizon,
    )
    normalizer = Normalizer(args.data / "normalization.npz")
    envs = make_envs(args)
    observation, _ = envs.reset([args.seed + i for i in range(args.n_envs)])
    normalized = normalizer.observation(observation)
    history = np.repeat(
        normalized[:, None], args.cond_steps, axis=1
    )
    # Reuse compiled rollout kernels. Without these wrappers, every environment
    # chunk pays Python tracing/dispatch overhead for the complete inner MDP.
    sample_policy = eqx.filter_jit(
        lambda policy_, states_, key_: policy_.sample(states_, key_)
    )
    transition_log_prob = eqx.filter_jit(
        lambda policy_, states_, chain_: policy_.transition_log_prob(
            states_, chain_
        )
    )
    critic_values = eqx.filter_jit(
        lambda critic_, states_: critic_.batch(states_)
    )
    env_steps = 0
    start = time.perf_counter()
    for iteration in range(1, args.iterations + 1):
        batch, history, key, mean_reward = collect_rollout(
            state.policy, state.critic, envs, history, normalizer, args, key,
            sample_policy, transition_log_prob, critic_values,
        )
        for _ in range(args.update_epochs):
            state, metrics = dppo_update(
                state, actor_optimizer, critic_optimizer, batch, config
            )
            if float(metrics.approx_kl) > args.target_kl: break
        env_steps += args.rollout_steps * args.n_envs * args.horizon
        append_csv(args.output / "finetune_metrics.csv", {
            "iteration": iteration, "environment_steps": env_steps,
            "mean_chunk_reward": mean_reward,
            "loss": float(metrics.total_loss),
            "policy_loss": float(metrics.policy_loss),
            "value_loss": float(metrics.value_loss),
            "approx_kl": float(metrics.approx_kl),
            "clip_fraction": float(metrics.clip_fraction),
            "ratio": float(metrics.ratio),
            "wall_seconds": time.perf_counter() - start,
            "nfe": state.policy.nfe, "seed": args.seed,
        })
        print(iteration, env_steps, mean_reward, float(metrics.total_loss))
        if args.checkpoint_every and iteration % args.checkpoint_every == 0:
            save_policy(
                args.output / "checkpoints" / f"iteration_{iteration:04d}"
                / "checkpoint",
                state.policy, args, state.critic,
            )
    save_policy(args.output / "checkpoint", state.policy, args, state.critic)
    envs.close()


def evaluate(args):
    import gymnasium as gym
    if "Adroit" in args.env:
        import gymnasium_robotics
        gym.register_envs(gymnasium_robotics)
    policy = load_policy(args.checkpoint, args, jr.PRNGKey(args.seed))
    normalizer = Normalizer(args.data / "normalization.npz")
    kwargs = {}
    if args.max_episode_steps is not None:
        kwargs["max_episode_steps"] = args.max_episode_steps
    env = gym.make(args.env, **kwargs)
    returns, successes, lengths, latencies = [], [], [], []
    for episode in range(args.eval_episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        history = [normalizer.observation(observation)] * args.cond_steps
        episode_return, episode_success, episode_length = 0.0, False, 0
        done = False
        eval_key = jr.PRNGKey(args.seed + episode)
        while not done:
            state = jnp.asarray(np.stack(history)[None])
            eval_key, sample_key = jr.split(eval_key)
            start = time.perf_counter()
            sample = policy.sample(
                state, sample_key, deterministic=True
            )
            jax.block_until_ready(sample.actions)
            latencies.append(1e3 * (time.perf_counter() - start))
            for action_step in range(args.horizon):
                raw_action = normalizer.action(np.asarray(sample.actions[0, action_step]))
                observation, reward, terminated, truncated, info = env.step(raw_action)
                history = history[1:] + [normalizer.observation(observation)]
                episode_return += float(reward)
                episode_length += 1
                episode_success |= bool(info.get("success", False))
                done = terminated or truncated
                if done: break
        returns.append(episode_return)
        successes.append(episode_success)
        lengths.append(episode_length)
        append_csv(args.output / "evaluation_episodes.csv", {
            "episode": episode, "return": episode_return,
            "success": int(episode_success), "length": episode_length,
            "seed": args.seed + episode, "nfe": policy.nfe,
        })
    summary = {
        "episodes": args.eval_episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "length_mean": float(np.mean(lengths)),
        "latency_ms_mean": float(np.mean(latencies[1:] or latencies)),
        "latency_ms_std": float(np.std(latencies[1:] or latencies)),
        "nfe": policy.nfe,
        "deterministic": True,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    env.close()


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("pretrain", "finetune", "evaluate"))
    p.add_argument("--policy", choices=("interpolant", "dppo"), default="interpolant")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--env", default="AdroitHandDoor-v2")
    p.add_argument("--max-episode-steps", type=int)
    p.add_argument("--obs-dim", type=int, required=True)
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--cond-steps", type=int, default=1)
    p.add_argument("--integration-steps", type=int, default=5)
    p.add_argument("--denoising-steps", type=int, default=20)
    p.add_argument("--gaussian-base", action="store_true")
    p.add_argument("--sigma-0", type=float, default=.1)
    p.add_argument("--sampling-std", type=float, default=.1)
    # 64x2 gives 226,548 total SI parameters for Door, closely matching the
    # official DPPO MLP actor's 211,904 parameters.  The previous 256x3
    # transformer had 4.89M parameters and was not a fair efficiency baseline.
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--time-dim", type=int, default=16)
    p.add_argument("--context-dim", type=int, default=32)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gradient-steps", type=int, default=100000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--iterations", type=int, default=301)
    p.add_argument(
        "--checkpoint-every", type=int, default=10,
        help="Save a fine-tuning policy/critic checkpoint every N iterations; 0 disables.",
    )
    p.add_argument("--n-envs", type=int, default=40)
    p.add_argument("--rollout-steps", type=int, default=70)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--actor-learning-rate", type=float, default=1e-4)
    p.add_argument("--critic-learning-rate", type=float, default=1e-3)
    p.add_argument("--critic-hidden-size", type=int, default=256)
    p.add_argument("--critic-depth", type=int, default=3)
    p.add_argument("--max-grad-norm", type=float)
    p.add_argument("--gamma", type=float, default=.99)
    p.add_argument("--gae-lambda", type=float, default=.95)
    p.add_argument("--gamma-denoising", type=float, default=.99)
    p.add_argument("--clip", type=float, default=.01)
    p.add_argument("--clip-base", type=float, default=.01)
    p.add_argument("--clip-rate", type=float, default=3)
    p.add_argument("--vf-coef", type=float, default=.5)
    p.add_argument("--target-kl", type=float, default=1)
    p.add_argument("--eval-episodes", type=int, default=50)
    return p


if __name__ == "__main__":
    args = parser().parse_args()
    if args.mode == "pretrain":
        pretrain(args)
    elif args.mode == "finetune":
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for finetune")
        finetune(args)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for evaluate")
        evaluate(args)
