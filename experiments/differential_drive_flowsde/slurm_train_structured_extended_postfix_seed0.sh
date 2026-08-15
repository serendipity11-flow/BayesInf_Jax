#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=dd_struct_ext_s0
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/slurm-%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0/slurm-%j.out

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
PYTHON_BIN=/home/zhangyi/myenv_312/bin/python
DATA="$PROJECT_ROOT/data/differential_drive/smooth_terrain_navigation_extended_state_postfix.h5"
RESULT="$PROJECT_ROOT/results/results_optimal_control/differential_drive/structured_extended_postfix_seed0"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib-dd-structured-${SLURM_JOB_ID}"
export WANDB_MODE=disabled

mkdir -p "$MPLCONFIGDIR" "$RESULT"
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.train_multishoot \
    --data "$DATA" \
    --output "$RESULT" \
    --structured-kinematics \
    --history-steps 1 \
    --segment-length 100 \
    --shooting-interval 5 \
    --stride 10 \
    --particles 4 \
    --segments-per-epoch 2048 \
    --validation-segments 256 \
    --batch-size 32 \
    --terrain-bins 4 \
    --drift-epochs 120 \
    --joint-epochs 180 \
    --drift-lr 3e-4 \
    --joint-lr 5e-5 \
    --warmup-epochs 5 \
    --final-lr-ratio 0.05 \
    --drift-weight 1 \
    --joint-drift-weight 2 \
    --transition-weight 1 \
    --continuity-weight 5 \
    --energy-weight 5 \
    --width 192 \
    --depth 3 \
    --condition-dim 96 \
    --flow-layers 6 \
    --flow-scale-limit 0.25 \
    --seed 0 \
    --wandb-project ""

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.plot_multishoot_training \
    --result "$RESULT" \
    --output "$RESULT/evaluation"

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate \
    --data "$DATA" \
    --result "$RESULT/joint_stage" \
    --output "$RESULT/evaluation" \
    --particles 100 \
    --calibration-transitions 2000 \
    --rollout-episodes 10 \
    --horizon 100 \
    --seed 17

"$PYTHON_BIN" -m experiments.differential_drive_flowsde.evaluate_closed_loop \
    --result "$RESULT/joint_stage" \
    --output "$RESULT/closed_loop" \
    --tasks 6 \
    --max-steps 2000 \
    --mpc-horizon 100 \
    --candidates 128 \
    --elites 16 \
    --cem-iterations 3 \
    --goal-tolerance 0.35 \
    --seed 23
