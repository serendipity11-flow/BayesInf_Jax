#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_mppi_complex
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_complex/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_complex/slurm-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
OUTPUT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_complex"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/matplotlib-dd-mppi-complex-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_nonlinear_physical_mppi \
  --output "$OUTPUT" --task-set complex --tasks 3 --max-steps 6000 \
  --horizon 100 --samples 256 --iterations 3 --temperature .35 \
  --sigma-v .7 --sigma-omega 1.5 --goal-tolerance .35 --report-every 50 \
  --gain-v .225 --gain-omega .36 --save-samples --seed 47
