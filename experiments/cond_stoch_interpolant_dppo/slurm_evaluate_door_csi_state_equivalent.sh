#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_state_eval
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-state-eval-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-state-eval-%A_%a.out

set -euo pipefail

ROOT=/home/zhangyi/bayesinf_jax
TRAIN_SEED=${SLURM_ARRAY_TASK_ID}
EVAL_STEPS=${EVAL_STEPS:-100}
PYTHON=${PYTHON:-}

if [[ -z "$PYTHON" ]]; then
  for candidate in \
    /home/zhangyi/myenv_312/bin/python \
    /usr/bin/python3.12 \
    /usr/local/bin/python3.12
  do
    if [[ -x "$candidate" ]] && "$candidate" -c \
      'import jax, equinox, gymnasium_robotics' >/dev/null 2>&1
    then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "No compatible Door/JAX Python found on $(hostname)" >&2
  exit 127
fi

BASE="$ROOT/results/results_optimal_control/door/csi_state_equivalent_v1/seed$TRAIN_SEED/pretrain"
CHECKPOINT="$BASE/checkpoint"
if [[ "$EVAL_STEPS" -eq 100 ]]; then
  EVAL_DIR=csi_state_equivalent_v1_evaluation
else
  EVAL_DIR="csi_state_equivalent_v1_${EVAL_STEPS}step_evaluation"
fi
OUT="$ROOT/results/results_optimal_control/door/$EVAL_DIR/seed$TRAIN_SEED"
test -f "$CHECKPOINT/policy.eqx"
mkdir -p "$OUT" "$ROOT/results/results_optimal_control/slurm"
cd "$ROOT"

export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "host=$(hostname) python=$PYTHON method=csi_state_equivalent train_seed=$TRAIN_SEED nfe=$EVAL_STEPS"

# train_jax.evaluate JIT-compiles the requested sampler once, then
# reuses it for every action chunk. Resume support avoids repeating completed
# episodes if a wall-time or node interruption occurs.
"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax evaluate \
  --policy interpolant --score-mode learned \
  --data data/door-expert-v2-100 \
  --checkpoint "$CHECKPOINT" --output "$OUT" \
  --env AdroitHandDoor-v2 --max-episode-steps 300 \
  --obs-dim 39 --action-dim 28 --horizon 4 --execution-horizon 4 --cond-steps 1 \
  --encoder-type mlp --integration-steps 100 \
  --inference-steps "$EVAL_STEPS" \
  --sigma-0 1.0 --score-sigma-floor 0.05 \
  --sde-eta 0.25 --sampling-std 0.001 \
  --score-loss-weight 0.25 \
  --hidden-size 128 --score-head-width 128 --depth 3 \
  --context-dim 64 --time-dim 32 --n-heads 4 \
  --seed 1000 --eval-episodes 50 --stochastic-eval --resume-eval
