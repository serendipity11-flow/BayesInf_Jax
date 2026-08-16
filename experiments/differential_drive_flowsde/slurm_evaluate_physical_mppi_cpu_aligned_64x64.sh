#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_phys_mppi_64_cpu
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex_aligned_64x16_cpu/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex_aligned_64x16_cpu/slurm-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
MODEL="$PROJECT_ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex/extended_physical_model.npz"
OUTPUT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_extended_complex_aligned_64x16_cpu"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-dd-physical-mppi-cpu-${SLURM_JOB_ID}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_nonlinear_physical_mppi \
    --output "$OUTPUT" --extended-model "$MODEL" --task-set complex --tasks 3 \
    --max-steps 6000 --horizon 100 --samples 64 --stochastic-samples 16 --iterations 3 \
    --temperature .35 --sigma-v .75 --sigma-omega 1.4 --goal-tolerance .35 \
    --report-every 50 --save-samples --seed 47
