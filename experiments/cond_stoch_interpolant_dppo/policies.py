"""Generative policies with a common PPO-facing Markov-chain interface.

The PPO code treats every generative transition as an inner-MDP action.  Both
policies therefore expose ``sample_chain`` and ``transition_log_prob`` with
identical tensor shapes.  The SI base encoder is trained by BC and frozen
during PPO: its sample is the initial inner state, exactly as Gaussian noise is
the (non-policy) initial state for DDPM.  This keeps the PPO action count and
credit assignment matched.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.distributions import Normal


def _state(cond: Mapping[str, Tensor]) -> Tensor:
    return cond["state"].flatten(1)


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device) * (-math.log(10_000) / max(half - 1, 1))
        )
        emb = t.float().reshape(-1, 1) * freq.reshape(1, -1)
        out = torch.cat((emb.sin(), emb.cos()), -1)
        return torch.nn.functional.pad(out, (0, self.dim - out.shape[-1]))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class MatchedMLP(nn.Module):
    """DPPO-style 3x256 residual MLP with a [128, 32] condition encoder."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        hidden: int = 256,
        depth: int = 3,
        time_dim: int = 16,
    ):
        super().__init__()
        self.action_dim, self.horizon = action_dim, horizon
        self.cond = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Mish(), nn.Linear(128, 32)
        )
        self.time = nn.Sequential(
            SinusoidalTime(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.Mish(),
            nn.Linear(time_dim * 2, time_dim),
        )
        self.input = nn.Linear(action_dim * horizon + 32 + time_dim, hidden)
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(depth)])
        self.output = nn.Linear(hidden, action_dim * horizon)

    def forward(self, x: Tensor, t: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        z = torch.cat((x.flatten(1), self.time(t), self.cond(_state(cond))), -1)
        return self.output(self.blocks(torch.nn.functional.mish(self.input(z)))).reshape_as(x)


class InformativeBase(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, horizon: int):
        super().__init__()
        out = action_dim * horizon
        self.action_dim, self.horizon = action_dim, horizon
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Mish(), nn.Linear(128, 128), nn.Mish()
        )
        self.mean = nn.Linear(128, out)
        self.log_std = nn.Linear(128, out)

    def params(self, cond: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        h = self.net(_state(cond))
        shape = (-1, self.horizon, self.action_dim)
        return self.mean(h).reshape(shape), self.log_std(h).reshape(shape).clamp(-5, 1)

    def forward(self, cond: Mapping[str, Tensor], deterministic: bool = False) -> Tensor:
        mean, log_std = self.params(cond)
        return mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)


@dataclass
class PolicySample:
    trajectories: Tensor
    chains: Tensor


class CondStochInterpolant(nn.Module):
    """Linear stochastic interpolant and Euler--Maruyama inner policy.

    BC uses I_t=(1-t)x0+t*x1+gamma(t)z and regresses its path derivative.
    Online sampling uses the learned probability-flow drift with a fixed,
    explicitly logged exploration std per transition.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        horizon: int,
        steps: int,
        informative_base: bool = True,
        bridge_sigma: float = 0.1,
        sampling_std: float = 0.1,
    ):
        super().__init__()
        self.obs_dim, self.action_dim, self.horizon = obs_dim, action_dim, horizon
        self.steps, self.nfe = steps, steps
        self.bridge_sigma, self.sampling_std = bridge_sigma, sampling_std
        self.velocity = MatchedMLP(obs_dim, action_dim, horizon)
        self.base = (
            InformativeBase(obs_dim, action_dim, horizon) if informative_base else None
        )

    def base_sample(self, cond: Mapping[str, Tensor], deterministic: bool = False) -> Tensor:
        if self.base is None:
            b = len(cond["state"])
            return torch.zeros(
                b, self.horizon, self.action_dim, device=cond["state"].device
            ) if deterministic else torch.randn(
                b, self.horizon, self.action_dim, device=cond["state"].device
            )
        return self.base(cond, deterministic)

    def loss(self, actions: Tensor, cond: Mapping[str, Tensor]) -> Tensor:
        b = len(actions)
        eps = 1e-4
        t = torch.rand(b, device=actions.device).clamp(eps, 1 - eps)
        x0 = self.base_sample(cond)
        z = torch.randn_like(actions)
        view = (b,) + (1,) * (actions.ndim - 1)
        tv = t.view(view)
        gamma = self.bridge_sigma * torch.sqrt(2 * tv * (1 - tv))
        gamma_dot = self.bridge_sigma * (1 - 2 * tv) / torch.sqrt(
            2 * tv * (1 - tv)
        )
        xt = (1 - tv) * x0 + tv * actions + gamma * z
        target = actions - x0 + gamma_dot * z
        return torch.nn.functional.mse_loss(self.velocity(xt, t, cond), target)

    @torch.no_grad()
    def sample_chain(
        self, cond: Mapping[str, Tensor], deterministic: bool = False
    ) -> PolicySample:
        x = self.base_sample(cond, deterministic)
        chain = [x]
        dt = 1.0 / self.steps
        for k in range(self.steps):
            t = torch.full((len(x),), (k + 0.5) * dt, device=x.device)
            mean = x + dt * self.velocity(x, t, cond)
            x = mean if deterministic else mean + self.sampling_std * math.sqrt(dt) * torch.randn_like(x)
            chain.append(x)
        return PolicySample(x.clamp(-1, 1), torch.stack(chain, 1))

    def transition_log_prob(
        self, cond: Mapping[str, Tensor], chains: Tensor, indices: Tensor | None = None
    ) -> Tensor:
        prev, nxt = chains[:, :-1], chains[:, 1:]
        b, k = prev.shape[:2]
        if indices is None:
            indices = torch.arange(k, device=prev.device)
        prev, nxt = prev[:, indices], nxt[:, indices]
        ki = len(indices)
        flat_cond = {
            key: value[:, None].expand(b, ki, *value.shape[1:]).flatten(0, 1)
            for key, value in cond.items()
        }
        dt = 1.0 / self.steps
        t = ((indices.float() + 0.5) * dt)[None].expand(b, ki).flatten()
        mean = prev.flatten(0, 1) + dt * self.velocity(prev.flatten(0, 1), t, flat_cond)
        std = torch.full_like(mean, max(self.sampling_std * math.sqrt(dt), 1e-6))
        return Normal(mean, std).log_prob(nxt.flatten(0, 1)).reshape(b, ki, *mean.shape[1:])

    forward = sample_chain


def parameter_report(model: nn.Module) -> dict[str, int]:
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

