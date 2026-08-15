#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_mppi_samples
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_mppi_candidate_sweep/slurm-%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/nonlinear_mppi_candidate_sweep/slurm-%A_%a.out

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
COUNTS=(64 128 256 512)
SAMPLES=${COUNTS[$SLURM_ARRAY_TASK_ID]}
OUTPUT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/nonlinear_mppi_candidate_sweep/samples_${SAMPLES}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="/tmp/matplotlib-dd-mppi-sweep-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_nonlinear_physical_mppi \
  --output "$OUTPUT" --task-set direct --tasks 3 --max-steps 2000 \
  --horizon 100 --samples "$SAMPLES" --iterations 3 --temperature .35 \
  --sigma-v .7 --sigma-omega 1.5 --goal-tolerance .35 --report-every 100 \
  --gain-v .225 --gain-omega .36 --seed 31
