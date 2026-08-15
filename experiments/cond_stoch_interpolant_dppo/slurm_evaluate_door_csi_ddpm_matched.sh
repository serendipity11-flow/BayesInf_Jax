#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_csi_match_eval
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/door/csi_ddpm_matched_evaluation/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/door/csi_ddpm_matched_evaluation/slurm-%j.out
set -euo pipefail
ROOT=/home/zhangyi/bayesinf_jax
OUTPUT="$ROOT/results/results_optimal_control/door/csi_ddpm_matched_evaluation"
cd "$ROOT"
mkdir -p "$OUTPUT"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda MUJOCO_GL=egl
export PYTHONPATH="$ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"
PYTHON="$HOME/myenv_312/bin/python"
"$PYTHON" -m experiments.cond_stoch_interpolant_dppo.evaluate_across_datasets \
  --manifest experiments/cond_stoch_interpolant_dppo/config/evaluation_door_csi_ddpm_matched.json
"$PYTHON" experiments/cond_stoch_interpolant_dppo/plot_score_ablation.py \
  "$OUTPUT/final_results.csv" "$OUTPUT/door_csi_ddpm_matched.png" --title Door
