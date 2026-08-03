# MuJoCo pusher pseudo-process dataset

Synthetic 12-state, three-action pusher trajectories with contact-force labels
and correlated pseudo-process disturbances. The snapshot has 128 episodes of
250 transitions (32,000 rows).

Recreate the exact default snapshot and its JSON metadata sidecar:

```bash
python -m experiments.mujoco_pusher_flowsde.generate_data \
  --model experiments/mujoco_pusher_flowsde/pusher_passage.xml \
  --output data/mujoco_pusher/pseudo_process.parquet \
  --episodes 128 --steps 250 --seed 0 --exploration 0.7 \
  --process-rho 0.97 --process-sigma 0.35 \
  --lateral-bias 0.08 --regime-steps 50
python -m experiments.mujoco_pusher_flowsde.plot_data
```

The Parquet and generated overview image stay outside Git.
