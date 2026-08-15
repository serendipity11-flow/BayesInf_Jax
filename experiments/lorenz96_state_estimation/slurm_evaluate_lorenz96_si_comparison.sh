#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=si_compare_eval
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_state_estimation/lorenz96_N12/si_comparison/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_state_estimation/lorenz96_N12/si_comparison/logs/%x_%j.err

set -euo pipefail

REPO=/home/zhangyi/bayesinf_jax
RESULTS="$REPO/results/results_state_estimation/lorenz96_N12"
OUTPUT="$RESULTS/si_comparison"
TAG=SI_N12_h101_obs2_gmm3c_std0.5_dc128_dh256_nh4_dep4_sig1.0_g0.5_sw0.25_lw0.1_sf0.05_lin_ow1_uw1.5_on0.03_us1.2_og0.9_ug1.25_ms100
CKPT="$RESULTS/${TAG}.pt"

mkdir -p "$OUTPUT/logs"
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

"$PYTHON" -m experiments.lorenz96_state_estimation.evaluate_stochastic_interpolant \
    --ckpt_path "$CKPT" \
    --output_dir "$OUTPUT" \
    --out_tag comparison \
    --legacy_lifter \
    --horizon 101 \
    --sigma_0 1.0 \
    --g0 0.5 \
    --score_weight 0.25 \
    --lifter_loss_weight 0.1 \
    --score_sigma_floor 0.05 \
    --interpolant_type linear \
    --ms \
    --seg_len 100 \
    --sampler segmented \
    --seg_len_eval 100 \
    --n_eval 20 \
    --n_particles 20 \
    --n_steps 10 \
    --metrics_path "$OUTPUT/conditional_stochastic_interpolant_metrics.csv"
