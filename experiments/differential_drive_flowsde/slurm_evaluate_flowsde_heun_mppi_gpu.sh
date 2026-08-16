#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_flow_mppi_gpu
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_gpu_aligned_64x64/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_gpu_aligned_64x64/slurm-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
RESULT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/joint_stage"
OUTPUT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_gpu_aligned_64x64"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-dd-heun-mppi-gpu-${SLURM_JOB_ID}"

mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_closed_loop \
    --result "$RESULT" --output "$OUTPUT" --task-set complex --tasks 3 \
    --max-steps 6000 --mpc-horizon 100 --candidates 64 --stochastic-samples 64 --elites 8 \
    --cem-iterations 3 --optimizer mppi --temperature .35 \
    --goal-tolerance .35 --report-every 50 --save-samples --seed 47
