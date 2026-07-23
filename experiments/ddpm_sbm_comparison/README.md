# Matched DDPM/SBM comparison

This experiment compares a discrete DDPM sampler with a continuous reverse-SDE
sampler for mass-spring-damper parameter identification. Both methods share the
same conditional epsilon-prediction architecture, objective, data stream,
normalization, initialization, and linear VP schedule. DDPM trains at discrete
VP times while SBM trains at continuous times. The DDPM transition variances are obtained
by exactly integrating that VP schedule over each discrete interval.

## Run

From the repository root:

```bash
python -m experiments.ddpm_sbm_comparison.run --cases basic linear multimodal
```

The default study uses five seeds and reverse grids from 25 to 1000 steps. It is
intended for a GPU and can take substantial time. Validate the installation with:

```bash
python -m experiments.ddpm_sbm_comparison.run --smoke --output /tmp/ddpm_sbm_smoke
```

Useful smaller run:

```bash
python -m experiments.ddpm_sbm_comparison.run \
  --cases basic linear multimodal --seeds 0 \
  --epochs 500 --train-size 2048 --posterior-samples 250 \
  --sample-steps 50 100 250
```

Outputs include an Equinox checkpoint per case/seed, posterior plots,
`metrics.csv`, and the complete `config.json`. Metrics cover point accuracy,
energy score, credible-interval calibration, DDPM/SBM sliced Wasserstein distance,
the linear-case constraint, and multimodal symmetry/mode balance.

This is the controlled comparison. The repository's original DDPM and SBM also
differ in architecture, condition dropout, classifier-free guidance, schedule,
and sampling, so their existing checkpoints should be treated as a separate
"native implementation" ablation rather than evidence about discretization.

## Theoretical convergence test

After training, evaluate both reverse samplers on each *identical, frozen* score
network. This removes independent-training variation and isolates the discrete
DDPM to reverse-SDE limit:

```bash
python -m experiments.ddpm_sbm_comparison.evaluate_convergence \
  --input experiments/ddpm_sbm_comparison/results/basic_seed0
```

The output `convergence_metrics.csv` records the same-network sliced Wasserstein
distance and posterior metrics, while `theoretical_convergence.png` plots that
distance against reverse step count. The L40S array for every case and seed is
`slurm_convergence_l40s.sh`.
