#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_match
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-8%3
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-match-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-match-%A_%a.out
set -euo pipefail

ROOT=/home/zhangyi/bayesinf_jax
SEED=$((SLURM_ARRAY_TASK_ID % 3))
VARIANT_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
case "$VARIANT_INDEX" in
  0) VARIANT=flow; SCORE_ARGS=(--score-mode flow --score-head-width 0 --score-loss-weight 0) ;;
  1) VARIANT=analytic; SCORE_ARGS=(--score-mode analytic --score-head-width 0 --score-loss-weight 0) ;;
  2) VARIANT=learned; SCORE_ARGS=(--score-mode learned --score-head-width 16 --score-loss-weight .05) ;;
  *) exit 2 ;;
esac
DATA="$ROOT/data/door-expert-v2-100"
OUT="$ROOT/results/results_optimal_control/door/csi_ddpm_matched/$VARIANT/seed$SEED/pretrain"
cd "$ROOT"
mkdir -p "$OUT" "$ROOT/results/results_optimal_control/slurm"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda MUJOCO_GL=egl
export PYTHONPATH="$ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON="$HOME/myenv_312/bin/python"

"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.check_door_compatibility
"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
  --policy csi_sde --data "$DATA" --output "$OUT" --env AdroitHandDoor-v2 \
  --max-episode-steps 300 --obs-dim 39 --action-dim 28 --horizon 4 \
  --execution-horizon 4 --cond-steps 1 --encoder-type mlp \
  --seed "$SEED" --gradient-steps 100000 --batch-size 128 \
  --learning-rate 1e-3 --weight-decay 1e-6 --log-every 1000 \
  --integration-steps 20 --sigma-0 .1 --sampling-std .1 \
  --sde-eta-schedule ddpm --score-sigma-floor .02 --path-loss-weight 0 \
  --hidden-size 72 --depth 3 --context-dim 32 --time-dim 16 --n-heads 4 \
  "${SCORE_ARGS[@]}"
