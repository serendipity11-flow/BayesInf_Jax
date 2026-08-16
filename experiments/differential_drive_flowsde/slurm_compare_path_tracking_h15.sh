#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_path_h12_compare
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/evaluation/path_tracking_h12/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/evaluation/path_tracking_h12/slurm-%j.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
PYTHON=/home/zhangyi/myenv_312/bin/python
BASE="$ROOT/results/results_optimal_control/differential_drive"
OUTPUT="$BASE/evaluation/path_tracking_h12"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-dd-path-compare-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$ROOT"
"$PYTHON" -m experiments.differential_drive_flowsde.plot_closed_loop_comparison \
  --flowsde "$BASE/flowsde_mppi/path_tracking_mppi_gpu_h12_64x16/closed_loop_metrics.json" \
  --physical "$BASE/nonlinear_physical_mppi/physical_path_tracking_mppi_cpu_h12_64x16/nonlinear_physical_mppi_metrics.json" \
  --output "$OUTPUT" --flowsde-hardware "A30 GPU" --physical-hardware "CPU (4 cores)"
