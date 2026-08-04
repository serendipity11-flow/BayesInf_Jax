# JAX FlowSDE-MPPI

This package implements the six-module interactive-pushing blueprint with
MuJoCo for the physical/oracle plant and JAX/Equinox for learned dynamics,
training, and vectorized MPPI rollouts.

```bash
python -m experiments.flowsde_mppi_jax.dataset --model path/to/pusher.xml \
  --output data/flowsde_mppi_jax/pushing.h5 --episodes 1000
python -m experiments.flowsde_mppi_jax.train \
  --data data/flowsde_mppi_jax/pushing.h5 \
  --output results/flowsde_mppi_jax
python -m experiments.flowsde_mppi_jax.evaluate --model path/to/pusher.xml \
  --config results/flowsde_mppi_jax/config.json \
  --output results/flowsde_mppi_jax/evaluation --samples 1000 --horizon 30
```

The transition table uses LeRobot field names (`observation.state`, `action`,
and `next_observation.state`) and episode/frame indices. The 24-D local state
and 78-D history input are documented in `state.py`.
