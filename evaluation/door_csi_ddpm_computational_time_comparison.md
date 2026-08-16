# CSI-DPPO versus DDPM-DPPO computational-time comparison

Both policies use 20 neural function evaluations (NFE) to generate each
four-action chunk. Despite the matched NFE, DDPM-DPPO is substantially faster
in the evaluated implementation.

| Stage | Policy | Latency per chunk (ms) | Latency per executed action (ms) | Evaluation wall time (s) |
|---|---|---:|---:|---:|
| Pretraining | CSI-DPPO | 3.406 +/- 0.008 | 0.851 +/- 0.002 | 27.46 +/- 0.40 |
| Pretraining | DDPM-DPPO | 0.607 +/- 0.001 | 0.152 +/- 0.000 | 16.10 +/- 0.02 |
| Fine-tuned | CSI-DPPO | 3.422 +/- 0.005 | 0.855 +/- 0.001 | 27.68 +/- 0.38 |
| Fine-tuned | DDPM-DPPO | 0.866 +/- 0.002 | 0.216 +/- 0.000 | 16.70 +/- 0.22 |

The values are means and sample standard deviations across three seeds. Each
evaluation contains 50 episodes. Because the controller executes four actions
from every generated chunk, the per-action latency is one quarter of the
per-chunk latency.

## Main comparison

For the fine-tuned policies, DDPM-DPPO is approximately 3.95 times faster in
policy inference:

```text
3.422 / 0.866 = 3.95
```

CSI-DPPO adds about 2.56 ms per generated chunk, or 0.64 ms per executed
action. The difference in complete evaluation time is smaller: CSI-DPPO is
approximately 1.66 times slower end to end (`27.68 / 16.70`). This ratio is
smaller because environment simulation, observation processing,
synchronization, and metric collection take time independently of policy
inference.

Both policies remain fast in absolute terms. Fine-tuned CSI-DPPO generates
approximately 292 action chunks per second, whereas DDPM-DPPO generates about
1,155 chunks per second. With four executed actions per chunk, this corresponds
to approximately 1,169 and 4,621 actions per second, respectively.

## Why CSI-DPPO is slower

The latency difference is not caused by the number of generative steps because
both policies use 20 NFE. It primarily reflects the computation performed at
each step.

The evaluated CSI policy contains 1,305,144 parameters:

- encoder: 71,168 parameters;
- velocity model: 616,988 parameters;
- score model: 616,988 parameters.

The DDPM baseline uses a considerably smaller residual MLP actor, documented
in the experiment description as containing approximately 211,904 parameters.
The evaluated CSI policy therefore has roughly six times as many parameters.
It also evaluates a learned-SDE construction involving velocity and score
computations, whereas DDPM uses a compact noise-prediction network with
precomputed diffusion-schedule coefficients.

The measured fourfold inference difference should consequently not be treated
as an intrinsic cost of all stochastic-interpolant policies. It is a comparison
of these specific implementations and architectures. A capacity-matched CSI
model, velocity-only policy, shared velocity-score backbone, or optimized
sampler could reduce the difference.

## Effect of fine-tuning on latency

CSI-DPPO latency changes little after fine-tuning, increasing from 3.406 to
3.422 ms per chunk. Its parameter count and sampling structure remain
essentially unchanged.

DDPM-DPPO latency increases from 0.607 to 0.866 ms per chunk, an increase of
approximately 43%. The fine-tuned DDPM sampler is hybrid: the frozen behavior-
cloning policy controls the early reverse-diffusion transitions, and the
fine-tuned actor controls the final transitions. The compiled sampling loop
computes quantities associated with both policies before selecting the
appropriate transition, increasing its inference cost.

## Performance-computation trade-off

At the final fine-tuning checkpoint, CSI-DPPO provides higher and more stable
task performance at greater computational cost:

| Metric | CSI-DPPO | DDPM-DPPO |
|---|---:|---:|
| Success rate | 0.947 +/- 0.012 | 0.887 +/- 0.070 |
| Episode return | 4,733.9 +/- 71.9 | 4,347.8 +/- 308.3 |
| Action jerk | 0.136 +/- 0.010 | 0.318 +/- 0.008 |
| Latency per chunk (ms) | 3.422 +/- 0.005 | 0.866 +/- 0.002 |

CSI-DPPO consequently improves success by approximately 6 percentage points
and return by about 386 points while reducing action jerk by roughly 57%.
DDPM-DPPO, by contrast, provides substantially lower inference latency and a
smaller actor.

## Paper-ready description

> Both CSI-DPPO and DDPM-DPPO use 20 generative function evaluations per
> action chunk, but their per-step computational costs differ substantially.
> After fine-tuning, CSI-DPPO requires 3.422 +/- 0.005 ms per four-action chunk,
> compared with 0.866 +/- 0.002 ms for DDPM-DPPO, making DDPM-DPPO approximately
> four times faster in policy inference. The end-to-end evaluation-time
> difference is smaller--27.68 versus 16.70 seconds, or approximately 1.66
> times--because environment simulation and evaluation overhead are shared by
> both methods. CSI-DPPO's higher latency is largely attributable to its larger
> velocity-and-score architecture rather than to additional integration steps.
> The results therefore reveal a trade-off: DDPM-DPPO provides greater
> computational efficiency, whereas CSI-DPPO provides higher and more stable
> fine-tuning performance and substantially smoother actions.

## Scope

These measurements describe policy-inference latency and complete evaluation
wall time. They do not compare behavior-cloning or DPPO training time. A valid
training-cost comparison would require wall-clock measurements collected under
matched hardware, software, batch-size, checkpointing, and job conditions.

## Data sources

- `results/results_optimal_control/door_dppo/csi_state_vs_ddpm_curve_evaluation/final_results.csv`
- `results/results_optimal_control/door_dppo/csi_state_vs_ddpm_curve_evaluation/checkpoints.csv`
- `results/results_optimal_control/door_dppo/csi_state_equivalent_v1_20step/seed0/pretrain/checkpoint/parameter_count.json`
