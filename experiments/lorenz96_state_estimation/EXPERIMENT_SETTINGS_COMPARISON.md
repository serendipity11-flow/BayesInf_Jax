# Lorenz-96 State-Estimation Experiment Settings

This document compares the dataset, observation-noise models, training
hyperparameters, and evaluation protocols used by the Lorenz-96 state-estimation
experiments in this directory. It reflects the current Python entry points,
Slurm launchers, and saved checkpoints under
`results/results_state_estimation/lorenz96_N12`.

## Shared dataset foundation

Most experiments generate data through `utils/dataset.py` and select/split it
through `dataset_selection.py`.

| Setting | Common value |
|---|---:|
| Lorenz-96 dimension | `N=12` |
| Forcing | `F=8.0` |
| Integration | RK4, `dt=0.02` |
| Spin-up | 500 steps |
| Observed coordinates | `0, 2, 4, 6, 8, 10` |
| Observation dimension | 6 |
| Observation scale | `0.5` |
| Dataset windows | 2,000 |
| Split | 80% train / 10% validation / 10% test |
| Split counts | 1,600 / 200 / 200 |
| Seed | 0 |
| Control | One all-zero channel |
| Initial state | Known `x0`; FlowSDE and LatentSSMFlow request clean initial states |

### Observation-noise definitions

- Gaussian: `epsilon ~ N(0, 0.5^2)`.
- Student-t: scale `0.5`, normally `df=5`.
- GMM:
  - Weights: `(0.70, 0.15, 0.15)`.
  - Offsets in units of `obs_std`: `(0, +2.5, -2.5)`.
  - Standard deviations in units of `obs_std`: `(0.5, 1.5, 1.5)`.
  - With `obs_std=0.5`, the component means are `0, +1.25, -1.25`
    and component standard deviations are `0.25, 0.75, 0.75`.

## Saved learned-model experiments

| Family | Saved data/noise | Effective trajectory setup | Architecture |
|---|---|---|---|
| AmortizedStateFlow | Gaussian, std `0.5` | Horizon 100, segments `5/10/20/25` | Transformer encoder; RealNVP and flow-matching variants |
| PointwiseStateFlow | Gaussian, std `0.5` | Horizon 100, causal windows `5/10/20/25` | GRU encoder; mostly RealNVP, plus FM |
| LatentSSMFlow | Student-t, df `5`, scale `0.5` | Horizon 100; segments `5/10/20/25/50` and full 100 | Local previous-only GRU, fixed linear SSM, RealNVP |
| FlowSDE | GMM, std `0.5` | Old runs use 101 states; current `uw=0.4` run uses horizon 100 | BiGRU context, Lorenz-96 physics drift, RealNVP diffusion |
| Stochastic Interpolant | GMM, std `0.5` | Primarily horizon 101; segments 10 or 100 | Transformer velocity/score networks |
| EKF/UKF/PF | Gaussian, Student-t, or GMM depending on benchmark | Horizon 100; no training | Exact Lorenz-96 RK4 transition |

## Training hyperparameters

| Setting | Amortized | Pointwise | Latent SSM | FlowSDE | Stochastic interpolant |
|---|---:|---:|---:|---:|---:|
| Batch size | 32 | 64 | 32 | 64 | 64 |
| Maximum epochs | 10,000 | 10,000 | 10,000 | 500 | 500 |
| Learning rate | `5e-4` | `1e-4` | `5e-4` | `3e-4` | `3e-4` |
| Gradient clipping | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| Weight decay | `1e-4` | `1e-4` | `1e-4` | `1e-4` | `1e-4` |
| Early-stop patience | 10 | 30 | 30 | 40 | 40 |
| Learning-rate schedule | Constant | Constant | Warm-up + cosine | Warm-up + cosine | Warm-up + cosine |
| Normalization | Dataset/model path | Dataset/model path | Train statistics | Train statistics | Train statistics |

### AmortizedStateFlow

Source: `train_amortized_state_flow.py`.

- Transformer encoder with `d_enc=64` and conditioning output size 96.
- Eight coupling blocks, coupling width 128, and depth 2.
- Default flow is RealNVP; saved results also include flow matching.
- Default prior is IID.
- Segments are non-overlapping unless `segment_stride` is supplied.
- The saved RealNVP/FM segment-comparison results are evaluated separately by
  `evaluation/compare_amortized_state_flow_segments.py`, using 50 test windows
  and 50 posterior draws by default. The aggregate results are stored in
  `evaluation/amortized_state_flow_segment_comparison_metrics.csv`.

### PointwiseStateFlow

Source: `train_pointwise_state_flow.py`.

- GRU encoder with hidden size 64.
- Conditioning dimension 96.
- Eight flow couplings with hidden width 128 and depth 2.
- Each estimate receives a causal observation history rather than a full
  smoothing context.

### LatentSSMFlow

Sources: `train_latent_ssm_flow.py` and `evaluate_latent_ssm_flow.py`.

- Saved models use RealNVP.
- GRU with local previous-only context.
- Fixed SSM with process standard deviation 1.0.
- Maximum transition spectral radius 0.8.
- Entropy regularization 0.01.
- Eight coupling blocks with hidden width 128 and depth 2.
- Dedicated evaluation defaults to 100 sequences, 100 samples, and a 90%
  credible interval.

### Current FlowSDE `uw=0.4`

Source: `slurm_train_lorenz96_flowsde_uw04.sh`.

- Context dimension `d_c=128`.
- BiGRU hidden size 128.
- Drift width 256 and depth 3.
- RealNVP diffusion width 64 with two coupling layers.
- Physics-informed drift: normalized Lorenz-96 RHS plus a learned residual.
- Stochastic Heun solver.
- Segment-start perturbation standard deviation 0.01.
- Path reconstruction plus:
  - Energy-score rollout weight 1.
  - Five rollout particles.
  - Energy uncertainty weight 0.4.
  - Rollout evaluated every five optimizer steps.
  - Rollout window 100.
  - Exact transition-NLL weight 1.
  - Continuity weight 0.1.
- The initial-state inference network has been removed; starts are provided
  states.
- Observation context is refreshed every state step (`observation_stride=1`).

### Stochastic interpolant

Primary source: `slurm_train_si_inferred_sigma_sweep.sh`.

- Context dimension 128.
- Hidden width 256.
- Four attention heads and depth 4.
- Linear interpolant.
- Inferred source distribution.
- `sigma_0` sweep: `0, 0.1, 0.25, 0.5, 0.75, 1.0`.
- `g0=0.5`.
- Score weight 0.25, except `sigma_0=0`, which uses zero score weight.
- Lifter/source loss weight 0.1.
- Multiple shooting with segment length 10.
- A separate experiment uses segment length 100.

## Evaluation settings

| Family | Test sequences | Posterior samples | Interval | Evaluation form |
|---|---:|---:|---:|---|
| Amortized RealNVP/FM comparison | 50 | 50 | 90% | Segment-conditioned comparison |
| Latent dedicated | 100 | 100 | 90% | Segmented or full-horizon option |
| FlowSDE dedicated | 100 | 200 | 90% | Continuous Heun rollout |
| SI large evaluation | 100 | 200 | 90% | 10 flow-time steps, segment 10 |
| UKF launcher | 200 | Analytic Gaussian | 90% | Continuous filter |
| Particle filter | 100 by default | 1,000 by default | 90% | Continuous filter |

The learned-model evaluators generally report posterior-mean RMSE,
observed/unobserved RMSE, MAE, and interval coverage. FlowSDE additionally
records mean credible-interval width. LatentSSMFlow also calculates NLL and
calibration diagnostics.

## Comparability limitations

The current results are not an apples-to-apples benchmark.

1. **Different observation distributions.** Amortized and pointwise flows use
   Gaussian noise, LatentSSMFlow uses Student-t noise, and FlowSDE and SI use
   GMM noise.
2. **Different horizon conventions.** Most flow models use 100 predicted
   states. Several SI and older FlowSDE runs use `h101`; the current FlowSDE
   checkpoint is explicitly `h100`.
3. **Different Monte Carlo sample counts.** Dedicated learned-model evaluations
   use 50-200 samples, while the default PF uses 1,000.
4. **Different access to boundary states.** Segmented models may restart from
   true segment boundaries. Continuous FlowSDE and filters normally receive
   only the initial `x0`. This can materially favor segmented evaluation.
5. **Different conditioning information.** BiGRU and Transformer models can
   use future observations, PointwiseStateFlow is causal-window based, and
   filters are strictly sequential.

## Recommended common comparison protocol

For a fair headline comparison, use:

- GMM observation noise with the common parameters above.
- Horizon 100 with one consistent indexing convention.
- The exact same seed-generated test split.
- 100 test sequences.
- 200 posterior samples per sequence.
- 90% credible intervals.
- Known initial `x0` only, with no true intermediate segment resets.
- Metrics: RMSE, MAE, observed/unobserved RMSE, coverage, mean interval width,
  NLL where available, and runtime.
