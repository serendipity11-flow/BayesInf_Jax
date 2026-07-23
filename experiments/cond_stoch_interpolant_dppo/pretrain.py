"""Step-matched behavior cloning for the SI policies.

Official DPPO pretraining is intentionally invoked through its own
``script/run.py``; see README.md.  This runner uses the identical NPZ chunks,
batch size, optimizer and explicit *gradient-step* budget for both SI variants.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .metrics import MetricLogger, peak_gpu_mb
from .policies import CondStochInterpolant, parameter_report


class Chunks(Dataset):
    def __init__(self, path: Path, horizon: int, cond_steps: int):
        data = np.load(path)
        self.states = torch.from_numpy(data["states"]).float()
        self.actions = torch.from_numpy(data["actions"]).float()
        self.items = []
        offset = 0
        for length in data["traj_lengths"]:
            for i in range(int(length) - horizon + 1):
                self.items.append((offset + i, i))
            offset += int(length)
        self.horizon, self.cond_steps = horizon, cond_steps

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        start, within = self.items[index]
        obs = torch.stack([
            self.states[start - min(within, d)]
            for d in reversed(range(self.cond_steps))
        ])
        return self.actions[start:start + self.horizon], {"state": obs}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["si_informative", "si_gaussian"], required=True)
    p.add_argument("--obs-dim", type=int, required=True)
    p.add_argument("--action-dim", type=int, required=True)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--cond-steps", type=int, default=1)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--gradient-steps", type=int, default=100_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    data = Chunks(a.data, a.horizon, a.cond_steps)
    loader = DataLoader(data, batch_size=a.batch_size, shuffle=True, drop_last=True)
    model = CondStochInterpolant(
        a.obs_dim, a.action_dim, a.horizon, a.steps,
        informative_base=a.method == "si_informative",
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-6)
    a.output.mkdir(parents=True, exist_ok=True)
    logger = MetricLogger(a.output / "metrics.csv", {
        "method": a.method, "seed": a.seed, "phase": "pretrain", "nfe": a.steps,
    })
    start = time.perf_counter()
    iterator = iter(loader)
    for step in range(1, a.gradient_steps + 1):
        try: actions, cond = next(iterator)
        except StopIteration: iterator = iter(loader); actions, cond = next(iterator)
        actions = actions.to(device); cond = {k: v.to(device) for k, v in cond.items()}
        loss = model.loss(actions, cond)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step == 1 or step % 1000 == 0 or step == a.gradient_steps:
            logger.log(gradient_steps=step, wall_seconds=time.perf_counter() - start,
                       peak_gpu_mb=peak_gpu_mb(device))
    torch.save({"model": model.state_dict(), "args": vars(a)}, a.output / "state.pt")
    (a.output / "parameter_count.json").write_text(json.dumps(parameter_report(model), indent=2))


if __name__ == "__main__":
    main()

