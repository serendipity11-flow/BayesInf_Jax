# ALOHA MuJoCo trajectories

Synthetic ALOHA trajectories generated from the vendored MuJoCo Menagerie
scene. `trajectories.parquet` contains the default 28-dimensional state run;
`mujoco_random_lerobot_matched.parquet` contains 14 joint positions and
correlated contact-dependent joint disturbances matched to the LeRobot horizon.

Recreate both snapshots from the repository root:

```bash
python -m experiments.aloha_mujoco_flow.generate_simulator_data
python -m experiments.aloha_mujoco_flow.generate_simulator_data \
  --output data/aloha_mujoco/mujoco_random_lerobot_matched.parquet \
  --steps 499 --smooth-episodes 30 --nominal-episodes 15 --mppi-episodes 5 \
  --positions-only --joint-disturbance
```

Exact parameters and checksums are recorded in `provenance.yaml`. The generator
also writes a detailed JSON sidecar next to each Parquet file.
