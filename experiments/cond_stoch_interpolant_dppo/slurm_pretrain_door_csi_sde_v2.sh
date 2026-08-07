#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_sde_v2_bc
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --array=0-2%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-sde-v2-bc-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/slurm/door-csi-sde-v2-bc-%A_%a.out
set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax; SEED="$SLURM_ARRAY_TASK_ID"
DATA="$ROOT/data/door-expert-v2-100"; OUT="$ROOT/results/results_optimal_control/door/csi_sde_v2/seed$SEED/pretrain"
cd "$ROOT"; mkdir -p "$OUT" "$ROOT/results/results_optimal_control/slurm"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda MUJOCO_GL=egl MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_ARRAY_JOB_ID}-${SEED}"
mkdir -p "$MPLCONFIGDIR"; PYTHON="$HOME/myenv_312/bin/python"
if [[ ! -x "$PYTHON" ]]; then PYTHON="$(command -v python3.12)"; fi
"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.train_jax pretrain \
 --policy csi_sde --data "$DATA" --output "$OUT" --env AdroitHandDoor-v2 \
 --max-episode-steps 300 --obs-dim 39 --action-dim 28 --horizon 4 --cond-steps 1 \
 --seed "$SEED" --gradient-steps 100000 --batch-size 128 --learning-rate 1e-3 \
 --weight-decay 1e-6 --log-every 1000 --integration-steps 20 --sigma-0 .1 \
 --sde-eta .005 --score-sigma-floor .02 --path-loss-weight .2 --path-loss-steps 20 \
 --score-head-width 16 --score-loss-weight .05 --score-correction-scale .1 \
 --hidden-size 52 --depth 3 --context-dim 64 --time-dim 16 --n-heads 4
