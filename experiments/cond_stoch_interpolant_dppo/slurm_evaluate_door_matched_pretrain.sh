#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_matched_eval
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --array=0-11%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-matched-eval-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-matched-eval-%A_%a.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
SEEDS=3
METHOD_INDEX=$((SLURM_ARRAY_TASK_ID / SEEDS))
TRAIN_SEED=$((SLURM_ARRAY_TASK_ID % SEEDS))

case "$METHOD_INDEX" in
  0) METHOD=fm; POLICY=interpolant; ROOT_DIR=matched_20step/fm
     MODEL_ARGS=(--score-mode flow --hidden-size 84 --depth 2 --context-dim 32)
     if [[ "$TRAIN_SEED" -eq 2 ]]; then MODEL_ARGS+=(--resume-eval); fi ;;
  1) METHOD=csi; POLICY=interpolant; ROOT_DIR=matched_20step/csi
     MODEL_ARGS=(--score-mode learned --hidden-size 56 --depth 2 --context-dim 32
                 --score-head-width 208 --score-loss-weight 0.1 --sde-eta 0.1
                 --stochastic-eval) ;;
  2) METHOD=sbm; POLICY=sbm; ROOT_DIR=matched_20step/sbm
     MODEL_ARGS=(--hidden-size 256 --depth 2 --context-dim 64
                 --sbm-context-width 128 --stochastic-eval --resume-eval) ;;
  3) METHOD=ddpm; POLICY=dppo; ROOT_DIR=diffusion_dppo_v2
     MODEL_ARGS=(--denoising-steps 20 --stochastic-eval) ;;
  *) exit 2 ;;
esac

PYTHON=${PYTHON:-}
if [[ -z "$PYTHON" ]]; then
  for candidate in "$HOME/myenv_312/bin/python" /usr/bin/python3.12 /usr/local/bin/python3.12; do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import jax, equinox, gymnasium_robotics' >/dev/null 2>&1; then
      PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$PYTHON" ]]; then
  echo "No compatible Door/JAX Python found on $(hostname)" >&2
  exit 127
fi

BASE="$ROOT/results/results_optimal_control/door/$ROOT_DIR/seed$TRAIN_SEED/pretrain"
CHECKPOINT="$BASE/checkpoint"
OUT="$ROOT/results/results_optimal_control/door/matched_20step_evaluation/$METHOD/seed$TRAIN_SEED"
test -f "$CHECKPOINT/policy.eqx"
mkdir -p "$OUT" "$ROOT/results/results_optimal_control/slurm"
cd "$ROOT"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}" JAX_PLATFORMS=cpu MUJOCO_GL=egl
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
echo "host=$(hostname) python=$PYTHON method=$METHOD train_seed=$TRAIN_SEED"

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax evaluate \
  --policy "$POLICY" --data data/door-expert-v2-100 \
  --checkpoint "$CHECKPOINT" --output "$OUT" \
  --env AdroitHandDoor-v2 --max-episode-steps 300 \
  --obs-dim 39 --action-dim 28 --horizon 4 --execution-horizon 4 --cond-steps 1 \
  --encoder-type mlp --integration-steps 20 --sampling-std 0.1 \
  --seed 1000 --eval-episodes 50 "${MODEL_ARGS[@]}"
