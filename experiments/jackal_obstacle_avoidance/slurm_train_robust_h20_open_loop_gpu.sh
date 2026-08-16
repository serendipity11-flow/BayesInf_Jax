#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=jackal_lssmf_robust
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_obstacle/robust_h20/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_obstacle/robust_h20/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/usr/bin/python3
RUN="$PROJECT_ROOT/results/results_optimal_control/jackal_obstacle/robust_h20"
TRAIN="$RUN/training"
EVALUATION="$RUN/open_loop"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-jackal-robust-${SLURM_JOB_ID}"

mkdir -p "$TRAIN" "$EVALUATION" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices(), flush=True)'

"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.train \
    --output "$TRAIN" --history 5 --rollout-steps 20 \
    --epochs 40 --batch-size 128 --lr 3e-4 \
    --hidden-dim 64 --width 64 --hidden-layers 2 --blocks 4 \
    --max-rho .9 --synthetic-episodes 256 --synthetic-steps 100 \
    --plant robust --seed 0 \
    --rollout-loss-weight 20 --rollout-discount .98

"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.evaluate_ood \
    --checkpoint "$TRAIN/jackal_latent_ssm_flow.eqx" \
    --output "$EVALUATION" --history 5 \
    --hidden-dim 64 --width 64 --hidden-layers 2 --blocks 4 --max-rho .9 \
    --open-loop-episodes 32 --open-loop-steps 80 --open-loop-particles 64 \
    --open-loop-only
