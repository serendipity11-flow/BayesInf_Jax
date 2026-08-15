#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_matched_bc
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-8%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-matched-bc-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-matched-bc-%A_%a.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
SEEDS=3
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / SEEDS))
SEED=$((SLURM_ARRAY_TASK_ID % SEEDS))

case "$METHOD_INDEX" in
  0) METHOD=fm; POLICY=interpolant
     MODEL_ARGS=(--score-mode flow --hidden-size 84 --depth 2 --context-dim 32) ;;
  1) METHOD=csi; POLICY=interpolant
     MODEL_ARGS=(--score-mode learned --hidden-size 56 --depth 2 --context-dim 32
                 --score-head-width 208 --score-loss-weight 0.1 --sde-eta 0.1) ;;
  2) METHOD=sbm; POLICY=sbm
     MODEL_ARGS=(--hidden-size 256 --depth 2 --context-dim 64
                 --sbm-context-width 128) ;;
  *) exit 2 ;;
esac

PYTHON=${PYTHON:-}
if [[ -z "$PYTHON" ]]; then
  for candidate in \
    "$HOME/myenv_312/bin/python" \
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
OUT="$ROOT/results/results_optimal_control/door/matched_20step/$METHOD/seed$SEED/pretrain"
mkdir -p "$OUT" "$ROOT/results/results_optimal_control/slurm"
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}" JAX_PLATFORMS=cuda MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
echo "host=$(hostname) python=$PYTHON method=$METHOD seed=$SEED"

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
  --policy "$POLICY" --data data/door-expert-v2-100 --output "$OUT" \
  --env AdroitHandDoor-v2 --max-episode-steps 300 \
  --obs-dim 39 --action-dim 28 --horizon 4 --cond-steps 1 \
  --encoder-type mlp --integration-steps 20 --sampling-std 0.1 \
  --seed "$SEED" --gradient-steps 100000 --batch-size 128 \
  --learning-rate 1e-3 --weight-decay 1e-6 --log-every 1000 \
  "${MODEL_ARGS[@]}"
