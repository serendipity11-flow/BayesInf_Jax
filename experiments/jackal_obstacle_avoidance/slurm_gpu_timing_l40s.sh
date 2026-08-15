#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=jackal_lssmf_gpu
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_ood/gpu_timing/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_ood/gpu_timing/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
# The Python 3.12 venv contains conflicting CUDA-12/CUDA-13 cuDNN wheels.
# The established user-site Python has the CUDA-12 JAX stack compatible with
# the cluster's 570-series L40S driver.
PYTHON_BIN=/usr/bin/python3
BASE="$PROJECT_ROOT/results/results_optimal_control/jackal_ood"
RUN="$BASE/gpu_timing"
IN_DOMAIN="$RUN/in_domain"
OOD="$RUN/ood"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-jackal-${SLURM_JOB_ID}"

mkdir -p "$IN_DOMAIN" "$OOD" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices(), flush=True)'

"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.evaluate \
    --checkpoint "$BASE/in_domain/jackal_latent_ssm_flow.eqx" \
    --output "$IN_DOMAIN" --closed-loop-episodes 4 --closed-loop-steps 100 \
    --horizon 20 --candidates 128 --particles 8 --cem-iterations 4 \
    --skip-open-loop

"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.evaluate_ood \
    --checkpoint "$BASE/jackal_latent_ssm_flow.eqx" \
    --output "$OOD" --closed-loop-episodes 4 --closed-loop-steps 120 \
    --open-loop-episodes 4 --open-loop-steps 80 --open-loop-particles 64 \
    --horizon 15 --candidates 64 --particles 4 --cem-iterations 3

"$PYTHON_BIN" -m evaluation.update_jackal_gpu_figures \
    --in-domain "$IN_DOMAIN" --ood "$OOD" \
    --cpu-in-domain "$BASE/in_domain/cem_comparison" \
    --cpu-ood "$BASE/ood_cem_comparison" \
    --output "$PROJECT_ROOT/evaluation"
