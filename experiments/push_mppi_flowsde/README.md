# Push MPPI FlowSDE

This package implements the six-module interactive-pushing blueprint with
MuJoCo for the physical/oracle plant and JAX/Equinox for learned dynamics,
training, and vectorized MPPI rollouts.

```bash
python -m experiments.push_mppi_flowsde.dataset --model path/to/pusher.xml \
  --output data/push_mppi_flowsde/pushing.h5 --episodes 1000
python -m experiments.push_mppi_flowsde.train \
  --data data/push_mppi_flowsde/pushing.h5 \
  --output results/push_mppi_flowsde
python -m experiments.push_mppi_flowsde.evaluate --model path/to/pusher.xml \
  --config results/push_mppi_flowsde/config.json \
  --output results/push_mppi_flowsde/evaluation --samples 1000 --horizon 30
```

The transition table uses LeRobot field names (`observation.state`, `action`,
and `next_observation.state`) and episode/frame indices. The 24-D local state
and 78-D history input are documented in `state.py`.
