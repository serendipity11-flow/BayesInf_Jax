#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_flow_async_gpu
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_async_gpu_64x16/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_async_gpu_64x16/slurm-%j.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
PYTHON=/home/zhangyi/myenv_312/bin/python
OUTPUT="$ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/complex_navigation_mppi_async_gpu_64x16"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-dd-flow-async-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$ROOT"
"$PYTHON" -m experiments.differential_drive_flowsde.evaluate_closed_loop \
  --result "$ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/joint_stage" \
  --output "$OUTPUT" --task-set complex --tasks 3 --max-steps 6000 \
  --mpc-horizon 100 --candidates 64 --stochastic-samples 16 --cem-iterations 3 \
  --optimizer mppi --temperature .35 --goal-tolerance .35 --report-every 50 \
  --async-control --command-dt .02 --seed 47
