#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_same_sampler_plot
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/door/csi_analytic_finetune_same_sampler_evaluation/plot-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/door/csi_analytic_finetune_same_sampler_evaluation/plot-%j.out
set -euo pipefail

ROOT=/home/zhangyi/bayesinf_jax
OUT="$ROOT/results/results_optimal_control/door/csi_analytic_finetune_same_sampler_evaluation"
cd "$ROOT"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLBACKEND=Agg MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR" "$OUT"
"$HOME/myenv_312/bin/python" -m experiments.cond_stoch_interpolant_dppo.plot_same_sampler_finetune \
 --csi results/results_optimal_control/door/csi_analytic_finetune_evaluation/checkpoints.csv \
 --ddpm "$OUT/checkpoints.csv" \
 --output "$OUT"
