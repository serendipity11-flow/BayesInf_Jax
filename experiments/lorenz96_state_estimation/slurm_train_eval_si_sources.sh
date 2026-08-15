#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=si_sources
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/experiments/lorenz96_state_estimation/logs/si_sources_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/experiments/lorenz96_state_estimation/logs/si_sources_%A_%a.err

set -euo pipefail

REPO=/home/zhangyi/bayesinf_jax
ROOT="$REPO/results/results_state_estimation/lorenz96_N12/si_sources"
SOURCES=(inferred gaussian)
TAGS=(inf_y gaussian)
SOURCE="${SOURCES[$SLURM_ARRAY_TASK_ID]}"
TAG="${TAGS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$ROOT/$TAG"
PLOT_DIR="$RUN_DIR/plots"

mkdir -p "$RUN_DIR" "$PLOT_DIR"
cd "$REPO"

export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cuda
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
VENV_SITE=/home/zhangyi/myenv_312/lib/python3.12/site-packages
export PYTHONPATH="$REPO:$VENV_SITE:${PYTHONPATH:-}"
PYTHON=/home/zhangyi/myenv_312/bin/python

"$PYTHON" -m experiments.lorenz96_state_estimation.train_stochastic_interpolant \
    --train --N 12 --horizon 101 --obs_interval 2 --obs_std 0.5 \
    --obs_noise_type gmm --d_c 128 --d_hidden 256 --n_heads 4 --depth 4 \
    --sigma_0 1.0 --g0 0.5 --score_weight 0.25 \
    --lifter_loss_weight 0.1 --score_sigma_floor 0.05 \
    --interpolant_type linear --source_type "$SOURCE" \
    --base_mean 0.0 --base_std 1.0 --ms --seg_len 10 \
    --batch_size 64 --max_epochs 500 --learning_rate 3e-4 \
    --grad_norm 1.0 --weight_decay 1e-4 --patience 40 --seed 0 \
    --experiment_tag "$TAG" --results_dir "$RUN_DIR"

CKPT=$(find "$RUN_DIR" -maxdepth 1 -name '*.pt' -print -quit)
test -n "$CKPT"

"$PYTHON" -m experiments.lorenz96_state_estimation.evaluate_stochastic_interpolant \
    --ckpt_path "$CKPT" --output_dir "$PLOT_DIR" --out_tag "$TAG" \
    --N 12 --horizon 101 --obs_interval 2 --obs_std 0.5 \
    --obs_noise_type gmm --d_c 128 --d_hidden 256 --n_heads 4 --depth 4 \
    --sigma_0 1.0 --g0 0.5 --score_weight 0.25 \
    --lifter_loss_weight 0.1 --score_sigma_floor 0.05 \
    --interpolant_type linear --source_type "$SOURCE" \
    --base_mean 0.0 --base_std 1.0 --ms --seg_len 10 \
    --sampler segmented --seg_len_eval 10 --n_eval 20 \
    --n_particles 20 --n_steps 10 \
    --metrics_path "$RUN_DIR/metrics.csv"
