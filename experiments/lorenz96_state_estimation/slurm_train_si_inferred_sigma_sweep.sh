#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=si_inf_sigma
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/experiments/lorenz96_state_estimation/logs/si_inf_sigma_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/experiments/lorenz96_state_estimation/logs/si_inf_sigma_%A_%a.err

set -euo pipefail
REPO=/home/zhangyi/bayesinf_jax
ROOT="$REPO/results/results_state_estimation/lorenz96_N12/si_inferred_sigma_sweep"
SIGMAS=(0 0.1 0.25 0.5 0.75 1.0)
FLOORS=(0.001 0.005 0.0125 0.025 0.0375 0.05)
SIGMA="${SIGMAS[$SLURM_ARRAY_TASK_ID]}"
FLOOR="${FLOORS[$SLURM_ARRAY_TASK_ID]}"
TAG="sig${SIGMA}"
RUN_DIR="$ROOT/$TAG"
SCORE_WEIGHT=0.25
if [[ "$SIGMA" == "0" ]]; then SCORE_WEIGHT=0; fi

mkdir -p "$RUN_DIR"
cd "$REPO"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda
export PYTHONPATH="$REPO:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
PYTHON="$HOME/myenv_312/bin/python"

"$PYTHON" -m experiments.lorenz96_state_estimation.train_stochastic_interpolant \
  --train --skip_post_train_eval --N 12 --horizon 101 --obs_interval 2 \
  --obs_std 0.5 --obs_noise_type gmm --d_c 128 --d_hidden 256 \
  --n_heads 4 --depth 4 --sigma_0 "$SIGMA" --g0 0.5 \
  --score_weight "$SCORE_WEIGHT" --lifter_loss_weight 0.1 \
  --score_sigma_floor "$FLOOR" --interpolant_type linear \
  --source_type inferred --ms --seg_len 10 --batch_size 64 \
  --max_epochs 500 --learning_rate 3e-4 --grad_norm 1.0 \
  --weight_decay 1e-4 --patience 40 --seed 0 \
  --experiment_tag "$TAG" --results_dir "$RUN_DIR"
