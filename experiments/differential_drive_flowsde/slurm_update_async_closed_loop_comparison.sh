#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_async_compare
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/evaluation/slurm-async-comparison-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/evaluation/slurm-async-comparison-%j.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
PYTHON=/home/zhangyi/myenv_312/bin/python
BASE="$ROOT/results/results_optimal_control/differential_drive"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-dd-async-compare-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
cd "$ROOT"
"$PYTHON" -m experiments.differential_drive_flowsde.plot_closed_loop_comparison \
  --flowsde "$BASE/structured_extended_postfix_seed0/complex_navigation_mppi_gpu_aligned_64x16/closed_loop_metrics.json" \
  --physical "$BASE/nonlinear_physical_mppi_extended_complex_aligned_64x16_cpu/nonlinear_physical_mppi_metrics.json" \
  --async-flowsde "$BASE/structured_extended_postfix_seed0/complex_navigation_mppi_async_gpu_64x16/closed_loop_metrics.json" \
  --async-physical "$BASE/nonlinear_physical_mppi_async_cpu_64x16/nonlinear_physical_mppi_metrics.json" \
  --output "$BASE/structured_extended_postfix_seed0/evaluation" \
  --flowsde-hardware "A30 GPU" --physical-hardware "CPU (4 cores)"
