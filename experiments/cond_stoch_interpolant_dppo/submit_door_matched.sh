#!/usr/bin/env bash
set -euo pipefail
PRETRAIN=$(sbatch --parsable experiments/cond_stoch_interpolant_dppo/slurm_pretrain_door_matched.sh)
FINETUNE=$(sbatch --parsable --dependency="afterok:$PRETRAIN" experiments/cond_stoch_interpolant_dppo/slurm_finetune_door_matched.sh)
echo "pretrain=$PRETRAIN finetune=$FINETUNE"
