#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_nonlinear_mppi_postfix
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_diagnostic_postfix/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_diagnostic_postfix/slurm-%j.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
OUTPUT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/nonlinear_physical_mppi_diagnostic_postfix"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/matplotlib-dd-nonlinear-postfix-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_nonlinear_physical_mppi \
  --output "$OUTPUT" --tasks 3 --max-steps 2000 --horizon 100 --samples 64 \
  --stochastic-samples 64 \
  --iterations 3 --temperature .35 --sigma-v .75 --sigma-omega 1.4 \
  --goal-tolerance .35 --report-every 25 --gain-v .225 --gain-omega .36 \
  --save-samples --seed 31
