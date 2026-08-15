#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=lorenz96_ukf
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_state_estimation/lorenz96_N12/ukf_logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_state_estimation/lorenz96_N12/ukf_logs/%x_%j.err

set -euo pipefail

REPO=/home/zhangyi/bayesinf_jax
RESULTS="$REPO/results/results_state_estimation/lorenz96_N12"
cd "$REPO"

export PYTHONNOUSERSITE=1
export JAX_PLATFORMS=cpu
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
VENV_SITE=/home/zhangyi/myenv_312/lib/python3.12/site-packages
export PYTHONPATH="$REPO:$VENV_SITE:${PYTHONPATH:-}"
if /home/zhangyi/myenv_312/bin/python -c 'import sys' >/dev/null 2>&1; then
    PYTHON=/home/zhangyi/myenv_312/bin/python
else
    PYTHON=python3
fi

"$PYTHON" -m experiments.lorenz96_state_estimation.benchmark_filters \
    --use-case amortized_state_flow \
    --method ukf \
    --noise-type gaussian \
    --N 12 \
    --horizon 100 \
    --obs-interval 2 \
    --obs-std 0.5 \
    --process-std 0.05 \
    --initial-std 0.0 \
    --n-eval 200 \
    --ci 90 \
    --segment-length 5 \
    --seed 0 \
    --out "$RESULTS/UnscentedKalmanFilter_lorenz96_N12_seg5_obs2_noise_0.5_eval"
