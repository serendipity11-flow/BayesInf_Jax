#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_heads_summary
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
/usr/bin/python3 -m experiments.MEMS_end_of_line.summarize_shared_encoder_heads
