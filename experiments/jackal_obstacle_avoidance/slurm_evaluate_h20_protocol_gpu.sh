#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=jackal_h20_eval
#SBATCH --partition=l40s
#SBATCH --qos=l40s-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_obstacle/slurm-eval-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_obstacle/slurm-eval-%j.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/usr/bin/python3
RUN="$PROJECT_ROOT/results/results_optimal_control/jackal_obstacle"
CHECKPOINT="$RUN/training/jackal_latent_ssm_flow.eqx"

export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-jackal-h20-eval-${SLURM_JOB_ID}"

mkdir -p "$RUN/in_domain" "$RUN/ood" "$MPLCONFIGDIR"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices(), flush=True)'

# Eight closed-loop trajectories on the in-domain plant.
"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.evaluate \
    --checkpoint "$CHECKPOINT" --output "$RUN/in_domain" --history 5 \
    --closed-loop-episodes 8 --closed-loop-steps 100 \
    --horizon 20 --candidates 256 --particles 8 --cem-iterations 5 \
    --elite-fraction .1 --initial-std .4 --risk-weight 1 \
    --smoothness-weight .05 --skip-open-loop

# Fifty closed-loop trajectories for every held-out OOD condition.
"$PYTHON_BIN" -m experiments.jackal_obstacle_avoidance.evaluate_ood \
    --checkpoint "$CHECKPOINT" --output "$RUN/ood" --history 5 \
    --hidden-dim 64 --width 64 --hidden-layers 2 --blocks 4 --max-rho .9 \
    --closed-loop-episodes 50 --closed-loop-steps 100 \
    --horizon 20 --candidates 256 --particles 8 --cem-iterations 5 \
    --risk-weight 1 --closed-loop-only
