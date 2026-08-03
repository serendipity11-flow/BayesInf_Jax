# Door human-v2 DPPO dataset

Episode-disjoint train/validation conversion of Minari
`D4RL/door/human-v2` for the conditional stochastic-interpolant versus
diffusion-policy comparison. Observations are standardized with training-set
statistics and actions are scaled to `[-1, 1]`.

Recreate it from the repository root:

```bash
python -m experiments.cond_stoch_interpolant_dppo.data \
  --dataset D4RL/door/human-v2 \
  --output data/dppo/door-human-v2 \
  --validation-fraction 0.2 --seed 0
python -m experiments.cond_stoch_interpolant_dppo.validate_data \
  data/dppo/door-human-v2 --obs-dim 39 --action-dim 28 --horizon 4
```

The split has 20 training episodes (5,444 transitions) and five validation
episodes (1,285 transitions). Exact artifact checksums are in
`provenance.yaml`.
