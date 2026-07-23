# CondStochInterpolant vs DPPO

This directory is a reproducible, **result-free** benchmark scaffold.  It does
not contain invented scores.  Raw per-seed CSVs are the source of truth; tables
and plots are generated only from those CSVs.

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

`I_t = (1-t)x0 + t*x1 + sigma*sqrt(2t(1-t))*z`

and regresses its analytic time derivative.  `x0` is sampled either from a
state-conditioned diagonal Gaussian or, for the Door ablation, N(0,I).
Sampling is a K-step Euler--Maruyama chain.  Its Gaussian transitions give
tractable log likelihoods for PPO.

`MatchedMLP` copies the official state-policy family: `[128,32]` condition
encoder, 16-D sinusoidal time embedding, and 3x256 residual trunk.  Always
record `parameter_count.json`; the informative base adds parameters, while the
velocity/diffusion backbones are capacity matched.

The native JAX policy reuses transformer classes from state estimation.  Its
matched configuration is width 64, depth 2, four heads, and context dimension
32.  On Door this has 226,548 parameters total (105,120 encoder + 12,408
informative base + 109,020 velocity), versus 211,904 in the official
Door-sized DPPO MLP actor.  Do not use the old 256x3 transformer configuration:
at 4,891,764 parameters it is 23.1 times the DPPO actor and invalidates a
compute-efficiency comparison.

The state-conditioned base is learned during BC and frozen during PPO.  This is
deliberate: DPPO treats its initial Gaussian draw as an inner-MDP state, not an
inner action.  Freezing `q` keeps the number of policy actions and the PPO
credit assignment identical.  It remains stochastic during collection.

## Door setup and verified interface

Create a dedicated environment (do not reuse it for Kitchen):

```bash
python -m venv --system-site-packages .venv-door
.venv-door/bin/pip install -r experiments/cond_stoch_interpolant_dppo/requirements-door.txt
MUJOCO_GL=egl .venv-door/bin/python \
  -m experiments.cond_stoch_interpolant_dppo.check_env
```

Verified with the lock file: dense `AdroitHandDoor-v1`, observation `(39,)`,
action `(28,)`, episode horizon 200, and binary `info["success"]`.  The sparse
ID is `AdroitHandDoorSparse-v1`; it is not used.  Gymnasium-Robotics warns that
the dense v1 reward changed without an ID bump, hence the strict package lock.

The official DPPO data list has no Door artifact.  Convert the Minari
D4RL-derived human demonstrations:

```bash
.venv-door/bin/python -m experiments.cond_stoch_interpolant_dppo.data \
  --dataset D4RL/door/human-v2 --output data/dppo/door-human-v2
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
  --data data/dppo/door-human-v2/train.npz \
  --output results/cond_stoch_interpolant_dppo/door/si_informative/seed0/pretrain \
  --method si_informative --obs-dim 39 --action-dim 28 --seed 0
```

Fine-tune through the same two-layer objective:

```bash
python -m experiments.cond_stoch_interpolant_dppo.finetune_si \
  --checkpoint results/cond_stoch_interpolant_dppo/door/si_informative/seed0/pretrain/state.pt \
  --normalization data/dppo/door-human-v2/normalization.npz \
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
  --data data/dppo/door-human-v2 --output results/dppo_jax/door/seed0 \
  --obs-dim 39 --action-dim 28 --seed 0
```

Fine-tune with the official Kitchen MLP PPO defaults (40 environments, 70
outer steps, 10 update epochs, gamma/GAE 0.99/0.95, clip 0.01, actor/critic
learning rates 1e-4/1e-3):

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl /home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.train_jax finetune \
  --data data/dppo/door-human-v2 \
  --checkpoint results/dppo_jax/door/seed0/checkpoint \
  --output results/dppo_jax/door/seed0-finetune \
  --env AdroitHandDoor-v1 --obs-dim 39 --action-dim 28 --seed 0
```

Deterministic evaluation:

```bash
JAX_PLATFORMS=cpu MUJOCO_GL=egl /home/zhangyi/myenv_312/bin/python -m \
  experiments.cond_stoch_interpolant_dppo.train_jax evaluate \
  --data data/dppo/door-human-v2 \
  --checkpoint results/dppo_jax/door/seed0-finetune/checkpoint \
  --output results/dppo_jax/door/seed0-eval \
  --env AdroitHandDoor-v1 --obs-dim 39 --action-dim 28 --eval-episodes 50
```

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

## Remaining gate before scientific claims

The local machine exposes no NVIDIA device and the full MuJoCo datasets were
not downloaded, so no five-seed learning curves or comparative result table
were generated here.  The experiment is not complete until official DPPO
Kitchen is reproduced, the new Door baseline passes its pilot, all raw runs
share hardware and budgets, and Kitchen per-element completion is added to the
legacy wrapper.  Any report before those gates should say “pending,” not infer
performance from BC loss or toy tests.
