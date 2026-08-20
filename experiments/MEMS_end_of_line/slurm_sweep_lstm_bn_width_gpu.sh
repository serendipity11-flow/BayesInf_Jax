#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_bn_width
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/encoder_width_sweep/logs/%x_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_parameter_identification/MEMS_end_of_line/encoder_width_sweep/logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
SWEEP_ROOT="$PROJECT_ROOT/results/results_parameter_identification/MEMS_end_of_line/encoder_width_sweep"
WIDTHS=(8 16 32 64)
WIDTH="${WIDTHS[$SLURM_ARRAY_TASK_ID]}"
RESULTS_DIR="$SWEEP_ROOT/width_$WIDTH"
cd "$PROJECT_ROOT"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$RESULTS_DIR"

PYTHON=/usr/bin/python3
echo "Width: $WIDTH"
echo "Results: $RESULTS_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

"$PYTHON" -m experiments.MEMS_end_of_line.train_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --output-dir "$RESULTS_DIR" \
    --models lstm_bn_flow \
    --epochs 20000 \
    --patience 1500 \
    --batch-size 200 \
    --learning-rate 1e-4 \
    --lr-hold-epochs 500 \
    --min-learning-rate-ratio 0.01 \
    --flow-dropout 0.1 \
    --width "$WIDTH" \
    --encoder-layers 2 \
    --transformer-heads 4 \
    --l2-coefficient 1e-5 \
    --seed 0 \
    --no-signal-standardization \
    --log-every 25 \
    --wandb-mode disabled \
    --wandb-project MEMS_end_of_line \
    --wandb-group "mems-lstm-bn-width-$WIDTH"

"$PYTHON" -m experiments.MEMS_end_of_line.evaluate_inference \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir "$RESULTS_DIR" \
    --models lstm_bn_flow \
    --posterior-samples 500 \
    --seed 0

"$PYTHON" -m experiments.MEMS_end_of_line.plot_encoder_pca \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir "$RESULTS_DIR" \
    --models lstm_bn_flow \
    --seed 0
