"""PPO fine-tuning with DPPO's outer/inner two-layer MDP.

This runner is for both SI variants.  It mirrors the official DPPO objective:
GAE on environment transitions; the same advantage is assigned to each
selected inner transition, discounted by ``gamma_denoising``; likelihood
ratios are averaged across action chunk coordinates before PPO clipping.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from .metrics import MetricLogger, benchmark_latency, peak_gpu_mb
from .policies import CondStochInterpolant


class Critic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.Mish(),
                                 nn.Linear(256, 256), nn.Mish(),
                                 nn.Linear(256, 256), nn.Mish(), nn.Linear(256, 1))
    def forward(self, state): return self.net(state.flatten(1)).squeeze(-1)


def make_envs(env_id, n, seed):
    import gymnasium as gym
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    def factory(i):
        def make():
            env = gym.make(env_id)
            env.reset(seed=seed + i)
            return env
        return make
    return gym.vector.SyncVectorEnv([factory(i) for i in range(n)])


def normalize_obs(obs, norm):
    return ((obs - norm["obs_mean"]) / norm["obs_std"]).astype(np.float32)


def denormalize_action(action, norm):
    return norm["action_min"] + (action + 1) * .5 * (
        norm["action_max"] - norm["action_min"]
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="AdroitHandDoor-v2")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--normalization", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["si_informative", "si_gaussian"], required=True)
    p.add_argument("--obs-dim", type=int, default=39)
    p.add_argument("--action-dim", type=int, default=28)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--ft-steps", type=int, default=5)
    p.add_argument("--n-envs", type=int, default=40)
    p.add_argument("--rollout-steps", type=int, default=70)
    p.add_argument("--environment-steps", type=int, default=840_000)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=5600)
    p.add_argument("--actor-lr", type=float, default=1e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--clip", type=float, default=.01)
    p.add_argument("--gamma-denoising", type=float, default=.99)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    assert 0 < a.ft_steps <= a.steps
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    norm = np.load(a.normalization)
    model = CondStochInterpolant(a.obs_dim, a.action_dim, a.horizon, a.steps,
        informative_base=a.method == "si_informative").to(device)
    model.load_state_dict(torch.load(a.checkpoint, map_location=device,
                                     weights_only=True)["model"])
    # Initial distribution is not an inner action in DPPO; freeze it for matched credit.
    if model.base is not None:
        for parameter in model.base.parameters(): parameter.requires_grad_(False)
    critic = Critic(a.obs_dim).to(device)
    actor_opt = torch.optim.AdamW(model.velocity.parameters(), lr=a.actor_lr)
    critic_opt = torch.optim.AdamW(critic.parameters(), lr=a.critic_lr)
    envs = make_envs(a.env, a.n_envs, a.seed)
    obs, _ = envs.reset(seed=[a.seed + i for i in range(a.n_envs)])
    a.output.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(a.output / "metrics.csv", {
        "environment": a.env, "method": a.method, "seed": a.seed,
        "nfe": a.steps,
    })
    total_steps = updates = 0
    start_all = time.perf_counter()
    while total_steps < a.environment_steps:
        update_start = time.perf_counter()
        states, chains, old_lp, rewards, dones, values = [], [], [], [], [], []
        completed_returns = np.zeros(a.n_envs); running_returns = np.zeros(a.n_envs)
        completed_success = []
        model.eval()
        for _ in range(a.rollout_steps):
            state = torch.as_tensor(normalize_obs(obs, norm), device=device)[:, None]
            with torch.no_grad():
                sample = model.sample_chain({"state": state})
                lp = model.transition_log_prob({"state": state}, sample.chains)
                value = critic(state)
            env_reward = np.zeros(a.n_envs)
            done = np.zeros(a.n_envs, dtype=bool)
            info = {}
            for chunk_i in range(a.horizon):
                act = denormalize_action(sample.trajectories[:, chunk_i].cpu().numpy(), norm)
                obs, r, term, trunc, info = envs.step(act)
                env_reward += r; done |= term | trunc; total_steps += a.n_envs
                if done.all() or total_steps >= a.environment_steps: break
            running_returns += env_reward
            final_info = info.get("final_info", [])
            for i, finished in enumerate(done):
                if finished:
                    completed_returns[i] = running_returns[i]; running_returns[i] = 0
                    fi = final_info[i] if i < len(final_info) and final_info[i] else {}
                    completed_success.append(float(fi.get("success", False)))
            states.append(state.cpu()); chains.append(sample.chains.cpu())
            old_lp.append(lp.cpu()); rewards.append(torch.tensor(env_reward))
            dones.append(torch.tensor(done)); values.append(value.cpu())
            if total_steps >= a.environment_steps: break
        next_state = torch.as_tensor(normalize_obs(obs, norm), device=device)[:, None]
        with torch.no_grad(): next_value = critic(next_state).cpu()
        rewards_t, dones_t, values_t = map(torch.stack, (rewards, dones, values))
        adv = torch.zeros_like(rewards_t); last = torch.zeros(a.n_envs)
        for t in reversed(range(len(rewards_t))):
            nv = next_value if t == len(rewards_t)-1 else values_t[t+1]
            delta = rewards_t[t] + .99 * nv * (~dones_t[t]) - values_t[t]
            last = delta + .99 * .95 * (~dones_t[t]) * last; adv[t] = last
        returns = adv + values_t
        st = torch.cat(states).to(device); ch = torch.cat(chains).to(device)
        olp = torch.cat(old_lp).to(device); ad = adv.flatten().to(device)
        ret = returns.flatten().to(device); oval = values_t.flatten().to(device)
        n = len(st); order = np.arange(n)
        model.train()
        for _ in range(a.update_epochs):
            np.random.shuffle(order)
            for begin in range(0, n, a.minibatch_size):
                ix = torch.as_tensor(order[begin:begin+a.minibatch_size], device=device)
                cond = {"state": st[ix]}
                newlp = model.transition_log_prob(cond, ch[ix])
                # Fine-tune final ft_steps, identical selection convention to DPPO.
                newlp, ref = newlp[:, -a.ft_steps:], olp[ix, -a.ft_steps:]
                logratio = newlp.clamp(-5, 2).mean((-1, -2)) - ref.clamp(-5, 2).mean((-1, -2))
                ratio = logratio.exp()
                advantage = (ad[ix] - ad[ix].mean()) / (ad[ix].std() + 1e-8)
                discount = a.gamma_denoising ** torch.arange(
                    a.ft_steps - 1, -1, -1, device=device)
                aa = advantage[:, None] * discount
                frac = torch.arange(a.ft_steps, device=device) / max(a.ft_steps - 1, 1)
                clip = .001 + (a.clip - .001) * (torch.exp(3*frac)-1)/(np.exp(3)-1)
                pg = torch.maximum(-aa*ratio, -aa*ratio.clamp(1-clip, 1+clip)).mean()
                v = critic(st[ix]); vf = .5 * (v-ret[ix]).square().mean()
                actor_opt.zero_grad(); pg.backward(); actor_opt.step()
                critic_opt.zero_grad(); vf.backward(); critic_opt.step()
        updates += 1
        elapsed = time.perf_counter() - update_start
        logger.log(phase="train", env_steps=total_steps, gradient_steps=updates,
                   episode_return=float(completed_returns[completed_returns != 0].mean())
                     if np.any(completed_returns != 0) else "",
                   success=float(np.mean(completed_success)) if completed_success else "",
                   wall_seconds=time.perf_counter()-start_all, update_seconds=elapsed,
                   env_step_seconds=elapsed/max(1, len(rewards)*a.n_envs*a.horizon),
                   peak_gpu_mb=peak_gpu_mb(device))
        if updates % a.eval_every == 0:
            lat, _ = benchmark_latency(model, {"state": next_state[:1]})
            logger.log(phase="latency", env_steps=total_steps, gradient_steps=updates,
                       latency_ms=lat, peak_gpu_mb=peak_gpu_mb(device))
    torch.save({"model": model.state_dict(), "critic": critic.state_dict(), "args": vars(a)},
               a.output / "state.pt")
    envs.close()


if __name__ == "__main__":
    main()
