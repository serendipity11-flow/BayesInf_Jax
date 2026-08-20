#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_bn_heads
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads/logs/%x_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
RESULT_ROOT="$PROJECT_ROOT/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads"
ENCODER_DIR="$PROJECT_ROOT/results/results_parameter_identification/MEMS_end_of_line/lstm_bn_encoder_width_sweep/width_64"
MODELS=(bnn vae nsf fm ddpm sbm)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
RESULTS_DIR="$RESULT_ROOT/$MODEL"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$RESULTS_DIR"

PYTHON=/usr/bin/python3
echo "Inference head: $MODEL"
echo "Frozen encoder: $ENCODER_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$PYTHON" -m experiments.MEMS_end_of_line.train_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --output-dir "$RESULTS_DIR" \
    --models "$MODEL" \
    --pretrained-lstm-bn-encoder-dir "$ENCODER_DIR" \
    --epochs 10000 \
    --patience 1500 \
    --batch-size 200 \
    --learning-rate 1e-4 \
    --lr-scheduler \
    --min-learning-rate-ratio 0.01 \
    --flow-dropout 0.1 \
    --width 64 \
    --encoder-layers 2 \
    --l2-coefficient 1e-5 \
    --diffusion-steps 100 \
    --flow-steps 100 \
    --seed 0 \
    --no-signal-standardization \
    --log-every 25 \
    --wandb-mode disabled \
    --wandb-project MEMS_end_of_line \
    --wandb-group "mems-frozen-lstm-bn64-$MODEL"

"$PYTHON" -m experiments.MEMS_end_of_line.evaluate_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir "$RESULTS_DIR" \
    --models "$MODEL" \
    --posterior-samples 500 \
    --seed 0
