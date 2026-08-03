#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=pusher_flowsde
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-3%4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/mujoco_pusher/logs/train_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/mujoco_pusher/logs/train_%A_%a.err

set -euo pipefail
PROJECT=/home/zhangyi/bayesinf_jax
DATA="$PROJECT/data/mujoco_pusher/pseudo_process.parquet"
NAMES=(baseline geometry contact both)
NAME="${NAMES[$SLURM_ARRAY_TASK_ID]}"
OUTPUT="$PROJECT/results/results_optimal_control/mujoco_pusher/flow_sde/$NAME"
CONFIG="$OUTPUT/aloha_flow_sde_heun_seg5_noise0p01_seed0_config.json"
ABLATION_ARGS=(--separate-dynamics-heads)
case "$NAME" in
  geometry) ABLATION_ARGS+=(--use-contact-geometry) ;;
  contact) ABLATION_ARGS+=(--contact-force-supervision-weight 1.0 --object-trajectory-supervision-weight 1.0) ;;
  both) ABLATION_ARGS+=(--use-contact-geometry --contact-force-supervision-weight 1.0 --object-trajectory-supervision-weight 1.0) ;;
esac

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false
mkdir -p "$MPLCONFIGDIR" "$OUTPUT" "$(dirname "$CONFIG")" \
  "$PROJECT/results/results_optimal_control/mujoco_pusher/logs"

PYTHON="$HOME/myenv_312/bin/python"
if [[ ! -x "$PYTHON" ]] || ! "$PYTHON" -c 'import jax, equinox, optax, pyarrow' >/dev/null 2>&1; then
  PYTHON="$(command -v python3.12)"
fi
cd "$PROJECT"
"$PYTHON" -c 'import jax; print("JAX devices:", jax.devices())'

"$PYTHON" -m bayesinf.applications.optimal_control.train_aloha_flow_sde \
  --data "$DATA" --output "$OUTPUT" \
  --epochs 300 --batch-size 16 --lr 1e-3 \
  --lr-decay-epoch 200 --lr-after-decay 3e-4 --dt 0.02 \
  --width 128 --depth 3 --diffusion-width 128 --flow-layers 4 \
  --time-embedding 16 --realnvp-alpha 2 \
  --window-steps 250 --segment-length 5 --stride 1 \
  --samples-per-epoch 100 --train-particles 4 --solver heun \
  --path-weight 0.1 --uncertainty-weight 0.5 --continuity-weight 1.0 \
  --contact-weight 1.0 --contact-center 0.5 --contact-width 0.15 \
  --contact-anneal-epochs 25 \
  --state-weight-epoch 200 --object-state-weight 1.0 \
  --state-noise-std 0.01 --control-noise-std 0.01 \
  --validation-windows 16 --seed 0 \
  --wandb-project bayesinf-jax \
  --wandb-run-name "mujoco-pusher-flowsde-$NAME" \
  "${ABLATION_ARGS[@]}"

"$PYTHON" -m bayesinf.applications.optimal_control.eval_aloha_flow_sde \
  --data "$DATA" --config "$CONFIG" --output "$OUTPUT/evaluation" \
  --particles 128 --eval-seed 2026

"$PYTHON" -m bayesinf.applications.optimal_control.eval_aloha_flow_sde \
  --data "$DATA" --config "$CONFIG" --output "$OUTPUT/evaluation_recondition50" \
  --particles 128 --eval-seed 2026 --recondition-every 50
