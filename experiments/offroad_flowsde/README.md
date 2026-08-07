# Off-road FlowSDE benchmark

The v2 dataset is a 100 Hz MuJoCo benchmark with hidden spatial friction and
speed-dependent stochastic disturbances. Observations are
`[p_x, p_y, theta, v_x, v_y, omega, omega_fl, omega_fr, omega_rl, omega_rr]`.
Controls are `[throttle_brake, steer]`, with steering limited to ±0.6 rad by
default. Friction and the instantaneous slip mode are never observations.

Generate a small validation dataset:

```bash
python -m experiments.offroad_flowsde.dataset \
  --output /tmp/offroad-smoke.h5 \
  --trajectories 10 --split-trajectories 7 1 2 --steps 60
```

Generate the training corpus (the defaults encode the complete specification):

```bash
python -m experiments.offroad_flowsde.dataset
```

This produces `data/offroad_flowsde/offroad_v2.h5` and a JSON manifest. Its
2,000 trajectories contain 1,000 transitions (10 seconds) each. Train and
validation use fixed four-component friction Map A with independent process and
policy seeds. The OOD test split uses unseen Map B with narrower Gaussian
kernels and hence sharper friction transitions. Each split independently uses
40% OU, 20% chirp, and 40% racing-controller trajectories.

The HDF5 file stores `state`, `action`, and `next_state` transition arrays plus
integer `split`, `policy`, and `trajectory_id` arrays. `window_start` contains
the starts of all H=50, stride=5 windows; slicing through `start + H + 1`
returns the 51 state/action pairs in the requested window without duplicating
overlapping data. Split names, policy names, ordering, dt, horizon, and stride
are recorded as file attributes and in the manifest.
