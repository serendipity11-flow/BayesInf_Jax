#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_sbm_nsf_lr
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads/logs/%x_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
RESULT_ROOT="$PROJECT_ROOT/results/results_parameter_identification/MEMS_end_of_line/lstm_bn64_inference_heads/lr_scheduler_tuning"
ENCODER_DIR="$PROJECT_ROOT/results/results_parameter_identification/MEMS_end_of_line/lstm_bn_encoder_width_sweep/width_64"
MODELS=(nsf nsf sbm sbm)
LEARNING_RATES=(1e-3 5e-4 1e-3 5e-4)
DECAY_EPOCHS=(3000 3000 1000 1000)
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
LEARNING_RATE="${LEARNING_RATES[$SLURM_ARRAY_TASK_ID]}"
DECAY_EPOCH="${DECAY_EPOCHS[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="${MODEL}_lr${LEARNING_RATE}_cos${DECAY_EPOCH}"
RESULTS_DIR="$RESULT_ROOT/$RUN_NAME"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$RESULTS_DIR"

PYTHON=/usr/bin/python3
echo "Inference head: $MODEL"
echo "Learning rate: $LEARNING_RATE"
echo "Warmup/cosine epochs: 50/$DECAY_EPOCH"
echo "Results: $RESULTS_DIR"

"$PYTHON" -m experiments.MEMS_end_of_line.train_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --output-dir "$RESULTS_DIR" \
    --models "$MODEL" \
    --pretrained-lstm-bn-encoder-dir "$ENCODER_DIR" \
    --epochs 10000 \
    --patience 1500 \
    --batch-size 200 \
    --learning-rate "$LEARNING_RATE" \
    --lr-scheduler \
    --lr-warmup-epochs 50 \
    --lr-decay-epochs "$DECAY_EPOCH" \
    --min-learning-rate-ratio 0.001 \
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
    --wandb-group "mems-frozen-lstm-bn64-${RUN_NAME}"

"$PYTHON" -m experiments.MEMS_end_of_line.evaluate_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir "$RESULTS_DIR" \
    --models "$MODEL" \
    --posterior-samples 500 \
    --seed 0
