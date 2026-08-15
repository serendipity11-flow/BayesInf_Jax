#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_state_bc
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-2%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-state-bc-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-state-bc-%A_%a.out

set -euo pipefail

ROOT=/home/zhangyi/bayesinf_jax
SEED=${SLURM_ARRAY_TASK_ID}
PYTHON=${PYTHON:-}

if [[ -z "$PYTHON" ]]; then
  for candidate in \
    /home/zhangyi/myenv_312/bin/python \
    /usr/bin/python3.12 \
    /usr/local/bin/python3.12
  do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import jax, equinox' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "No Python 3.12 interpreter with JAX/Equinox found on $(hostname)" >&2
  exit 127
fi

OUT="$ROOT/results/results_optimal_control/door/csi_state_equivalent_v1/seed$SEED/pretrain"
mkdir -p "$OUT" "$ROOT/results/results_optimal_control/slurm"
cd "$ROOT"

export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"

echo "host=$(hostname) python=$PYTHON method=csi_state_equivalent seed=$SEED"

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
  --policy interpolant --score-mode learned \
  --data data/door-expert-v2-100 --output "$OUT" \
  --env AdroitHandDoor-v2 --max-episode-steps 300 \
  --obs-dim 39 --action-dim 28 --horizon 4 --cond-steps 1 \
  --encoder-type mlp --integration-steps 100 \
  --sigma-0 1.0 --score-sigma-floor 0.05 \
  --sde-eta 0.25 --sampling-std 0.001 \
  --score-loss-weight 0.25 \
  --hidden-size 128 --score-head-width 128 --depth 3 \
  --context-dim 64 --time-dim 32 --n-heads 4 \
  --seed "$SEED" --gradient-steps 100000 --batch-size 128 \
  --learning-rate 1e-3 --weight-decay 1e-6 --log-every 1000
