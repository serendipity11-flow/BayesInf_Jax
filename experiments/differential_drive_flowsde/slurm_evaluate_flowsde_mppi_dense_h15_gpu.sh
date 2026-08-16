#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_flow_path_h12
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/flowsde_mppi/path_tracking_mppi_gpu_h12_64x16/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/flowsde_mppi/path_tracking_mppi_gpu_h12_64x16/slurm-%j.out

set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
PYTHON=/home/zhangyi/myenv_312/bin/python
OUTPUT="$ROOT/results/results_optimal_control/differential_drive/flowsde_mppi/path_tracking_mppi_gpu_h12_64x16"
export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/mpl-dd-flow-dense-h15-${SLURM_JOB_ID}"
mkdir -p "$OUTPUT" "$MPLCONFIGDIR"
cd "$ROOT"
"$PYTHON" -m experiments.differential_drive_flowsde.evaluate_closed_loop \
  --result "$ROOT/results/results_optimal_control/differential_drive/flowsde_mppi/joint_stage" \
  --output "$OUTPUT" --task-set complex_dense --tasks 3 --max-steps 10000 \
  --mpc-horizon 12 --candidates 64 --stochastic-samples 16 --cem-iterations 3 \
  --optimizer mppi --temperature .35 --goal-tolerance .35 --path-lookahead 1.8 \
  --path-tolerance .55 --heading-weight 1.0 --report-every 50 --seed 47
