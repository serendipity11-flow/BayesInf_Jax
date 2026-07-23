import json
from pathlib import Path
import numpy as np

from experiments.cond_stoch_interpolant_dppo.pretrain import Chunks
from experiments.cond_stoch_interpolant_dppo.validate_data import validate


def check_prepared(root: Path, obs_dim: int, action_dim: int):
    report = validate(root, obs_dim, action_dim, horizon=4)
    manifest = json.loads((root / "manifest.json").read_text())
    assert set(manifest["train_episode_ids"]).isdisjoint(manifest["validation_episode_ids"])
    for split in ("train", "val"):
        dataset = Chunks(root / f"{split}.npz", horizon=4, cond_steps=1)
        actions, cond = dataset[0]
        assert actions.shape == (4, action_dim)
        assert cond["state"].shape == (1, obs_dim)
        assert len(dataset) == report[split]["chunks"]
        assert np.isfinite(actions.numpy()).all()


def test_door_prepared():
    check_prepared(Path("data/dppo/door-human-v2"), 39, 28)


def test_kitchen_prepared():
    check_prepared(Path("data/dppo/kitchen-complete-v0"), 60, 9)
