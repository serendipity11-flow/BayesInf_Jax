#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=si_inf_sig_eval
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --array=0-5%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/experiments/lorenz96_state_estimation/logs/si_inf_sig_eval_%A_%a.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/experiments/lorenz96_state_estimation/logs/si_inf_sig_eval_%A_%a.err

set -euo pipefail
REPO=/home/zhangyi/bayesinf_jax
ROOT="$REPO/results/results_state_estimation/lorenz96_N12/si_inferred_sigma_sweep"
SIGMAS=(0 0.1 0.25 0.5 0.75 1.0)
FLOORS=(0.001 0.005 0.0125 0.025 0.0375 0.05)
SIGMA="${SIGMAS[$SLURM_ARRAY_TASK_ID]}"
FLOOR="${FLOORS[$SLURM_ARRAY_TASK_ID]}"
RUN_DIR="$ROOT/sig${SIGMA}"
OUTPUT="$RUN_DIR/eval_100x200"
CKPT=$(find "$RUN_DIR" -maxdepth 1 -name '*.pt' -print -quit)
SAMPLER=segmented_parallel
if [[ "$SIGMA" == "0" ]]; then SAMPLER=ode_segmented_parallel; fi

mkdir -p "$OUTPUT"
cd "$REPO"
export PYTHONNOUSERSITE=1 JAX_PLATFORMS=cuda
export PYTHONPATH="$REPO:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-${SLURM_JOB_ID}"
PYTHON="$HOME/myenv_312/bin/python"

"$PYTHON" -m experiments.lorenz96_state_estimation.evaluate_stochastic_interpolant \
  --ckpt_path "$CKPT" --output_dir "$OUTPUT" --out_tag "sig${SIGMA}_100x200" \
  --N 12 --horizon 101 --obs_interval 2 --obs_std 0.5 \
  --obs_noise_type gmm --d_c 128 --d_hidden 256 --n_heads 4 --depth 4 \
  --sigma_0 "$SIGMA" --g0 0.5 --score_weight 0.25 \
  --lifter_loss_weight 0.1 --score_sigma_floor "$FLOOR" \
  --interpolant_type linear --source_type inferred --ms --seg_len 10 \
  --sampler "$SAMPLER" --seg_len_eval 10 --n_eval 100 \
  --n_particles 200 --n_steps 10 --eval_chunk_size 20 \
  --metrics_path "$OUTPUT/metrics.csv"
