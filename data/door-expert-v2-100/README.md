# Adroit Door expert-100 dataset

This is an episode-disjoint conversion of the first 100 genuine trajectories
from Minari `D4RL/door/expert-v2`. The seeded split contains 80 training
episodes (16,000 transitions) and 20 validation episodes (4,000 transitions).
Observations are standardized using training-only statistics and actions are
scaled to `[-1, 1]`.

Recreate and validate it from the repository root:

```bash
/home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.data \
  --dataset D4RL/door/expert-v2 \
  --output data/door-expert-v2-100 \
  --max-episodes 100 --validation-fraction 0.2 --seed 0
/home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.validate_data \
  data/door-expert-v2-100 --obs-dim 39 --action-dim 28 --horizon 4
```
