# Kitchen Complete DPPO split

Episode-disjoint split of the official DPPO-preprocessed
`kitchen-complete-v0` trajectories. The converter preserves the official
full-dataset min/max preprocessing instead of normalizing the split twice.

First acquire the official snapshot into
`data/dppo/official/kitchen-complete-v0`, then recreate this split:

```bash
python -m experiments.cond_stoch_interpolant_dppo.split_dppo_data \
  --input data/dppo/official/kitchen-complete-v0/train.npz \
  --normalization data/dppo/official/kitchen-complete-v0/normalization.npz \
  --output data/dppo/kitchen-complete-v0 \
  --validation-fraction 0.2 --seed 0 \
  --source official-dppo-kitchen-complete-v0 --preserve-preprocessing
```

The resulting split has 15 training and four validation episodes. Checksums
are recorded in `provenance.yaml`.
