# CondStochInterpolant vs DPPO

This directory is a reproducible, **result-free** benchmark scaffold.  It does
not contain invented scores.  Raw per-seed CSVs are the source of truth; tables
and plots are generated only from those CSVs.

## Door experiment scope

The Door benchmark is **Adroit-only**: `AdroitHandDoor-v2`, a 39-dimensional
state observation, and a 28-dimensional hand action. It uses 100 trajectories
from `D4RL/door/expert-v2`. Alternate robot embodiments and end-effector control
interfaces are outside this experiment's scope.

## What the official baseline does (pinned audit)

The reference code is `irom-princeton/dppo` commit
`cc7234ad7ff39a8f32de3af903606723a16f0648` (v0.8.0).

* DPPO does **not** contain an Adroit or Door config, dataset URL, or checkpoint.
  Door is therefore a new environment adaptation, not a published DPPO
  reproduction.  Its closest state-based manipulation configuration is Kitchen.
* DPPO does contain `kitchen-{complete,partial,mixed}-v0`.  The paper-facing
  complete config has observation dimension 60, action dimension 9, action
  horizon 4, DDPM K=20, 10 fine-tuned steps, 40 environments, 70 outer rollout
  steps, PPO batch 5600, 10 update epochs, clip 0.01, actor LR 1e-4, critic LR
  1e-3, and 301 iterations.  `config/kitchen_complete.yaml` transcribes these.
* The official Kitchen checkpoint and preprocessed data URLs live in
  `third_party/dppo/script/download_url.py`.  There is no Door branch there.
* Official Kitchen uses legacy D4RL/Gym.  It must remain isolated from the
  Gymnasium-Robotics Door environment, as DPPO's own extras warn.

## Policy definition and matching

The SI uses

`I_t = (1-t)x0 + t*x1 + sigma*t*(1-t)*z`

and regresses its analytic time derivative. For Door, `x0` is sampled from the
fixed distribution N(0,I).
Sampling is a K-step Euler--Maruyama chain.  Its Gaussian transitions give
tractable log likelihoods for PPO.

`MatchedMLP` copies the official state-policy family: `[128,32]` condition
encoder, 16-D sinusoidal time embedding, and 3x256 residual trunk.  Always
record `parameter_count.json`; the informative base adds parameters, while the
velocity/diffusion backbones are capacity matched.

The native JAX policy reuses transformer classes from state estimation. Its
fixed-Gaussian-source configuration is width 52, depth 2, four heads, and
context dimension 32. The encoder and velocity head total 144,048 parameters,
versus 211,904 in the official Door-sized DPPO MLP actor. Parameter counts must
be reported with results.

CSI behavior cloning trains velocity matching only. Its source is fixed
`N(0,I)` and has no learned parameters. The marginal score is recovered from
the velocity and schedule identity documented in `cond_stoch_interpolant.py`;
there is no score head or score loss. Use manifest sampler `"ode"` for
deterministic velocity transport and `"sde"` for analytic-score
Euler--Maruyama sampling. Door uses 50 integration steps. The submission chain
evaluates BC first and launches DPPO only if mean CSI-ODE success is at least
50 percent.

The `si_analytic_path_v5` BC ablation supplements velocity matching with a
two-sample energy score on differentiable 10-step action-chunk endpoints. Its
weight is 0.1. This directly penalizes endpoint mismatch while the pairwise
term discourages mode collapse. Evaluation compares Euler and Heun ODE/SDE
sampling and sweeps SDE eta without changing checkpoint weights. Heun is an
inference ablation only; PPO continues to use Euler transitions because their
Gaussian likelihood is explicit.

## Door setup and verified interface

Create a dedicated environment (do not reuse it for Kitchen):

```bash
python -m venv --system-site-packages .venv-door
.venv-door/bin/pip install -r experiments/cond_stoch_interpolant_dppo/requirements-door.txt
MUJOCO_GL=egl .venv-door/bin/python \
  -m experiments.cond_stoch_interpolant_dppo.check_env
```

Before training, verify that the prepared demonstrations and the installed
online environment satisfy the same runtime contract:

```bash
MUJOCO_GL=egl .venv-door/bin/python \
  -m experiments.cond_stoch_interpolant_dppo.check_door_compatibility
```

The command exits nonzero on shape, action-bound, episode-horizon,
reset-observation-support, reward, or `info["success"]` incompatibilities.

### Matched Door pipeline

For the four-way, DDPM-controlled comparison requested here, use:

```bash
bash experiments/cond_stoch_interpolant_dppo/submit_door_matched.sh
```

This submits FM, learned-score CSI, and SBM pretraining followed by fine-tuning;
the already completed DDPM runs remain the baseline. All four use the same Door
data, three seeds, one observation-history step, action horizon 4, 100,000 BC
updates, and 20 generative integration steps. Fine-tuning uses the baseline
DPPO settings (301 iterations, 40 environments, 70 rollout chunks, and 10 PPO
epochs). The model roles are deliberately disjoint:

* FM has an MLP encoder and velocity model, with no score model. Fixed Gaussian
  transition noise is used only as PPO exploration; deterministic evaluation
  integrates the velocity ODE.
* CSI has separate learned velocity and score networks. Its encoder+velocity
  side has 95,804 parameters and its score side has 117,216 (213,020 total).
* SBM has one MLP-conditioned score/noise model and no velocity model.
* DDPM is the 211,904-parameter baseline.

The corresponding FM and SBM counts are 202,428 and 219,312. Thus each total
is within 5% of DDPM, while the two CSI sides are approximately half of the
baseline. Every saved checkpoint contains `parameter_count.json`, including
component counts and the integration-step count.

Submit the complete three-seed comparison as one dependency chain:

```bash
bash experiments/cond_stoch_interpolant_dppo/submit_door_pipeline.sh
```

The pretraining array runs both the stochastic-interpolant and diffusion
policies for 100,000 matched gradient steps on seeds 0--2. A dependent CSI-ODE
BC job then evaluates all three seeds on the same 50 resets. It exits nonzero
when mean success is below 50%, so the DPPO dependency is never released for a
failed behavior-cloning policy. After a passing gate, the fine-tuning array
applies the same DPPO settings to both policies for exactly 840,000 primitive
environment steps per seed. The final job evaluates every
pretraining and fine-tuning checkpoint on the same 50 reset seeds and writes
`episodes.csv`, `checkpoints.csv`, `final_results.csv`, `flags.csv`, plots, and
`results.md` under
`results/results_optimal_control/door/generative_policy_evaluation`.

Verified with the lock file: fixed dense `AdroitHandDoor-v2`, observation
`(39,)`, action `(28,)`, an explicit 300-step wrapper matching the offline
episodes, and binary `info["success"]`. Gymnasium-Robotics 1.2.1--1.4.2 changed
the dense v1 reward without an ID bump. The requirements pin the exact upstream
revision containing v2 because a 1.4.3 wheel has not yet been published. This
avoids the broken v1 contract while retaining Python 3.12 and NumPy 2 support.

The official DPPO data list has no Door artifact.  Convert the Minari
D4RL-derived human demonstrations:

```bash
.venv-door/bin/python -m experiments.cond_stoch_interpolant_dppo.data \
  --dataset D4RL/door/expert-v2 --output data/door-expert-v2-100 \
  --max-episodes 100
```

The converter preserves episode boundaries and writes DPPO-compatible
`states`, `actions`, `rewards`, `terminals`, and `traj_lengths`, plus the
normalization file used online.

## Runs

Clone the official baseline at the audited revision:

```bash
git clone https://github.com/irom-princeton/dppo third_party/dppo
git -C third_party/dppo checkout cc7234ad7ff39a8f32de3af903606723a16f0648
```

Run official Kitchen BC and PPO unchanged first (its data/checkpoint downloader
is automatic):

```bash
conda create -n dppo-kitchen python=3.8 -y
conda activate dppo-kitchen
pip install -e 'third_party/dppo[kitchen]'
export DPPO_DATA_DIR=$PWD/data/dppo
export DPPO_LOG_DIR=$PWD/results/cond_stoch_interpolant_dppo
export DPPO_WANDB_ENTITY=YOUR_ENTITY
python third_party/dppo/script/run.py --config-name=pre_diffusion_mlp \
  --config-dir=third_party/dppo/cfg/gym/pretrain/kitchen-complete-v0
python third_party/dppo/script/run.py --config-name=ft_ppo_diffusion_mlp \
  --config-dir=third_party/dppo/cfg/gym/finetune/kitchen-complete-v0 seed=0
```

Before the comparison, run a Door DPPO pilot by copying the Kitchen MLP configs
and changing only `env_name`, dimensions, normalization path, and episode
horizon to values in `config/door.yaml`.  Do not tune on SI outcomes.  Freeze
the success threshold based on this pilot, then launch seeds 0--4.

SI BC is explicitly gradient-step matched:

```bash
python -m experiments.cond_stoch_interpolant_dppo.pretrain \
  --data data/door-expert-v2-100/train.npz \
  --output results/cond_stoch_interpolant_dppo/door/si_informative/seed0/pretrain \
  --method si_informative --obs-dim 39 --action-dim 28 --seed 0
```

Fine-tune through the same two-layer objective:

```bash
python -m experiments.cond_stoch_interpolant_dppo.finetune_si \
  --checkpoint results/cond_stoch_interpolant_dppo/door/si_informative/seed0/pretrain/state.pt \
  --normalization data/door-expert-v2-100/normalization.npz \
  --output results/cond_stoch_interpolant_dppo/door/si_informative/seed0/finetune \
  --method si_informative --seed 0
```

Repeat with `--method si_gaussian` for the Door-only ablation.  Kitchen uses
the same SI commands with dimensions 60/9, its official DPPO NPZ, and its
isolated legacy environment.

## Native JAX DPPO adaptation

The maintained implementation now uses
`bayesinf.applications.optimal_control.CondStochInterpolantPolicy` and
`bayesinf.applications.optimal_control.dppo_jax`.  Formula provenance is the
vendored official files:

* `third_party/dppo/model/diffusion/diffusion_ppo.py`
* `third_party/dppo/agent/finetune/train_ppo_diffusion_agent.py`
* `third_party/dppo/cfg/gym/finetune/kitchen-complete-v0/ft_ppo_diffusion_mlp.yaml`

Pretrain:

```bash
JAX_PLATFORMS=cpu /home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
  --data data/door-expert-v2-100 --output results/dppo_jax/door/seed0 \
  --obs-dim 39 --action-dim 28 --seed 0
```

Fine-tune with the official Kitchen MLP PPO defaults (40 environments, 70
outer steps, 10 update epochs, gamma/GAE 0.99/0.95, clip 0.01, actor/critic
learning rates 1e-4/1e-3):

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl /home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.train_jax finetune \
  --data data/door-expert-v2-100 \
  --checkpoint results/dppo_jax/door/seed0/checkpoint \
  --output results/dppo_jax/door/seed0-finetune \
  --env AdroitHandDoor-v2 --obs-dim 39 --action-dim 28 --seed 0
```

Deterministic evaluation:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl /home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.train_jax evaluate \
  --data data/door-expert-v2-100 \
  --checkpoint results/dppo_jax/door/seed0-finetune/checkpoint \
  --output results/dppo_jax/door/seed0-eval \
  --env AdroitHandDoor-v2 --obs-dim 39 --action-dim 28 --eval-episodes 50
```

Prediction and execution horizons are separate.  Pass, for example,
`--horizon 16 --execution-horizon 8 --cond-steps 2` to replan after the first
eight actions of every 16-action chunk.  Existing Door runs retain their
matched four-action setting.  The evaluator also accepts per-method
`"execution_horizon"`, `"inference_steps"`, and `"sampler": "ddim"` manifest
keys.  SI checkpoints can be evaluated on a different Euler grid because the
grid has no learned parameters; diffusion checkpoints use deterministic,
strided DDIM and therefore require `inference_steps` not to exceed the trained
noise schedule.

Evaluation reports mean squared action jerk (the second finite difference),
mean primitive steps among successful episodes, and latency per generated
chunk.  It retains the legacy `action_smoothness` episode column as an alias
for downstream compatibility, but that value now has the documented jerk
semantics.

The JAX update uses the full collected inner-transition batch per compiled
update instead of PyTorch's shuffled flattened-inner-step minibatches.  This is
the same PPO objective and expectation, but a different minibatch schedule.
The informative source encoder and source head are frozen during online PPO;
only `SITransformerNet` is updated, keeping the initial inner state
non-policy-controlled as in DPPO.  These choices must remain identical for the
informative- and Gaussian-base SI runs.

## Measurement protocol

* Seeds: 0--4 for every cell; 50 evaluation episodes at each checkpoint.
* Evaluation uses deterministic generative transitions for both methods.
* Report environment steps and gradient steps.  Select the sample-efficiency
  threshold once from the baseline pilot and freeze it.
* Latency: batch size one, 20 warmups, 100 synchronized repeats.  NFE is actual
  drift/noise network calls, not nominal training steps.
* Reset CUDA peak-memory counters immediately before each separately timed
  pretraining/fine-tuning run.  Do not combine phases.
* Record GPU, CPU, CUDA, PyTorch, MuJoCo, environment version and git commit in
  every run directory.  Timing comparisons are invalid across different hosts.
* Kitchen overall success means all four target elements complete.  Also log
  `microwave`, `kettle`, `light_switch`, and `slide_cabinet` completion
  separately from the environment's task bookkeeping.

Aggregate and plot only after all seeds exist:

```bash
python -m experiments.cond_stoch_interpolant_dppo.metrics raw_metrics.csv \
  --output results_table.csv
python -m experiments.cond_stoch_interpolant_dppo.plot raw_metrics.csv \
  --output learning_curves.png
```

For checkpoint evaluation across every configured dataset, method, seed, and
training stage, use the manifest-driven harness:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl /home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.evaluate_across_datasets \
  --manifest experiments/cond_stoch_interpolant_dppo/config/evaluation_across_datasets.json
```

The manifest is the extension point for another experiment: add a task with
its environment registration module, normalization file, success semantics,
fixed reset seeds, and optional subtask keys. Checkpoint patterns remain
method-level and may use `{task}` and `{method}` placeholders. The evaluator
writes per-episode and per-checkpoint CSV files, final seed aggregates, flags
for missing stages or insufficient seeds, success curves, a Markdown table,
and optional GIFs. A one-episode pipeline check can be run with `--episodes 1`;
use `--record-episodes 1` only when the environment supports RGB rendering.

## Remaining gate before scientific claims

The local machine exposes no NVIDIA device and the full MuJoCo datasets were
not downloaded, so no five-seed learning curves or comparative result table
were generated here.  The experiment is not complete until official DPPO
Kitchen is reproduced, the new Door baseline passes its pilot, all raw runs
share hardware and budgets, and Kitchen per-element completion is added to the
legacy wrapper.  Any report before those gates should say “pending,” not infer
performance from BC loss or toy tests.
