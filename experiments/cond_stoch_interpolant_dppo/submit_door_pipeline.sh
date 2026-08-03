#!/usr/bin/env bash
# Submit matched Door pretraining, DPPO fine-tuning, and final evaluation.
set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"
mkdir -p \
    results/results_optimal_control/slurm \
    results/results_optimal_control/door/generative_policy_evaluation

PRETRAIN_JOB=$(sbatch --parsable \
    experiments/cond_stoch_interpolant_dppo/slurm_pretrain_door.sh)
FINETUNE_JOB=$(sbatch --parsable --dependency="afterok:$PRETRAIN_JOB" \
    experiments/cond_stoch_interpolant_dppo/slurm_finetune_door.sh)
EVALUATION_JOB=$(sbatch --parsable --dependency="afterok:$FINETUNE_JOB" \
    experiments/cond_stoch_interpolant_dppo/slurm_evaluate_across_datasets.sh)

echo "pretrain_job=$PRETRAIN_JOB"
echo "finetune_job=$FINETUNE_JOB"
echo "evaluation_job=$EVALUATION_JOB"
