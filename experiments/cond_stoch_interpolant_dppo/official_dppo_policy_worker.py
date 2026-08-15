"""Serve the pinned official DPPO Kitchen policy over a binary pipe."""
from __future__ import annotations

import argparse
import pickle
import struct
import sys
from pathlib import Path


def receive(stream):
    size_bytes = stream.read(8)
    if not size_bytes:
        raise EOFError
    size = struct.unpack("!Q", size_bytes)[0]
    payload = stream.read(size)
    if len(payload) != size:
        raise EOFError("truncated policy request")
    return pickle.loads(payload)


def send(stream, value):
    payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)
    stream.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import numpy as np
    import torch
    from model.diffusion.diffusion import DiffusionModel
    from model.diffusion.mlp_diffusion import DiffusionMLP

    network = DiffusionMLP(
        action_dim=9, horizon_steps=4, cond_dim=60, time_dim=16,
        mlp_dims=[256, 256, 256], cond_mlp_dims=[128, 32],
        residual_style=True,
    )
    policy = DiffusionModel(
        network=network, horizon_steps=4, obs_dim=60, action_dim=9,
        denoising_steps=20, predict_epsilon=True,
        denoised_clip_value=1.0, device=args.device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    policy.load_state_dict(checkpoint["ema"], strict=True)
    policy.eval()

    @torch.no_grad()
    def sample(state):
        """Pinned DDPM sampler, excluding an unused stale keyword upstream."""
        from model.diffusion.sampling import make_timesteps
        batch = len(state)
        x = torch.randn((batch, 4, 9), device=args.device)
        for t in reversed(range(20)):
            times = make_timesteps(batch, t, args.device)
            mean, logvar = policy.p_mean_var(
                x=x, t=times, cond={"state": state},
                index=make_timesteps(batch, 19 - t, args.device),
            )
            std = torch.zeros_like(mean) if t == 0 else torch.exp(0.5 * logvar).clamp(min=1e-3)
            noise = torch.randn_like(x).clamp(-policy.randn_clip_value, policy.randn_clip_value)
            x = mean + std * noise
        return x

    source, sink = sys.stdin.buffer, sys.stdout.buffer
    try:
        while True:
            command, payload = receive(source)
            if command == "sample":
                observation, seed = payload
                torch.manual_seed(int(seed))
                state = torch.as_tensor(
                    np.asarray(observation, dtype=np.float32), device=args.device
                ).reshape(1, 1, 60)
                action = sample(state)
                send(sink, action[0].detach().cpu().numpy())
            elif command == "close":
                send(sink, None)
                return
            else:
                raise ValueError(f"unknown command: {command}")
    except EOFError:
        return


if __name__ == "__main__":
    main()
