#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_notebook_bn
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/notebook_lstm_bn/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/notebook_lstm_bn/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
export BAYESFLOW_JAX_DATA_DIR=/home/zhangyi/data/MEMS/MEMS_simulated
export BAYESFLOW_JAX_OUTPUT_DIR="$PROJECT_ROOT/results/results_parameter_identification/MEMS_end_of_line/notebook_lstm_bn"
export BAYESFLOW_JAX_EPOCHS=10000
export BAYESFLOW_JAX_PATIENCE=1000
export BAYESFLOW_JAX_POSTERIOR_SAMPLES=200
export BAYESFLOW_JAX_PLOT_INTERVAL=100
export BAYESFLOW_JAX_RUN_NAME=mems_notebook_lstm_bn_seed42
export BAYESFLOW_JAX_REQUIRE_GPU=1
mkdir -p "$MPLCONFIGDIR" "$BAYESFLOW_JAX_OUTPUT_DIR/logs"

PYTHON=/usr/bin/python3
echo "Python: $PYTHON ($($PYTHON --version))"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON" -m experiments.MEMS_end_of_line.BayesFlow_MEMS_JAX_Equinox
