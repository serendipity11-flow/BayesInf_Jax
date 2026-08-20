#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_encoder_flows_eval
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"
mkdir -p results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison/logs
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

/usr/bin/python3 -m experiments.MEMS_end_of_line.evaluate_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison \
    --models lstm_bn_flow lstm_flow transformer_flow \
    --seed 0

/usr/bin/python3 -m experiments.MEMS_end_of_line.plot_encoder_pca \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir results/results_parameter_identification/MEMS_end_of_line/encoder_feature_raw_comparison \
    --seed 0
