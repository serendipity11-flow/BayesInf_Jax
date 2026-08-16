#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=jackal_lssmf_h20
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_obstacle/slurm-train-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_obstacle/slurm-train-%j.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/usr/bin/python3
RUN="$PROJECT_ROOT/results/results_optimal_control/jackal_obstacle"
TRAIN="$RUN/training"
EVALUATION="$RUN/evaluation"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-jackal-h20-${SLURM_JOB_ID}"

mkdir -p "$TRAIN" "$EVALUATION" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices(), flush=True)'

"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.train \
    --output "$TRAIN" --history 5 --rollout-steps 20 \
    --epochs 30 --batch-size 128 --lr 3e-4 \
    --hidden-dim 64 --width 64 --hidden-layers 2 --blocks 4 \
    --max-rho .9 --synthetic-episodes 128 --synthetic-steps 100 \
    --plant high_fidelity --seed 0

"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.evaluate \
    --checkpoint "$TRAIN/jackal_latent_ssm_flow.eqx" \
    --output "$EVALUATION" --history 5 --closed-loop-episodes 10 \
    --closed-loop-steps 100 --horizon 20 --candidates 256 \
    --particles 8 --cem-iterations 5 --elite-fraction .1 \
    --initial-std .4 --risk-weight 1 --smoothness-weight .05 \
    --open-loop-episodes 32 --open-loop-steps 100 --open-loop-particles 64
