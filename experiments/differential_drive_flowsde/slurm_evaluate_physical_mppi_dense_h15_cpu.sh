#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_phys_path_h12
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi/physical_path_tracking_mppi_cpu_h12_64x16/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi/physical_path_tracking_mppi_cpu_h12_64x16/slurm-%j.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
PYTHON=/home/zhangyi/myenv_312/bin/python
OUTPUT="$ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi/physical_path_tracking_mppi_cpu_h12_64x16"
MODEL="$ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi/nonlinear_physical_mppi_extended_complex/extended_physical_model.npz"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}" JAX_PLATFORMS=cpu
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-dd-phys-dense-h15-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$ROOT"
"$PYTHON" -m experiments.differential_drive_flowsde.evaluate_nonlinear_physical_mppi \
  --output "$OUTPUT" --extended-model "$MODEL" --task-set complex_dense --tasks 3 \
  --max-steps 10000 --horizon 12 --samples 64 --stochastic-samples 16 --iterations 3 \
  --temperature .35 --sigma-v .75 --sigma-omega 1.4 --goal-tolerance .35 \
  --path-lookahead 1.8 --path-tolerance .55 --heading-weight 1.0 \
  --report-every 50 --seed 47
