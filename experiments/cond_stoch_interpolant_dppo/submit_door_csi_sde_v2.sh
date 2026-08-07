#!/usr/bin/env bash
set -euo pipefail
cd /home/zhangyi/bayesinf_jax
PRETRAIN=$(sbatch --parsable experiments/cond_stoch_interpolant_dppo/slurm_pretrain_door_csi_sde_v2.sh)
FINETUNE=$(sbatch --parsable --dependency="afterok:$PRETRAIN" experiments/cond_stoch_interpolant_dppo/slurm_finetune_door_csi_sde_v2.sh)
EVALUATE=$(sbatch --parsable --dependency="afterok:$FINETUNE" experiments/cond_stoch_interpolant_dppo/slurm_evaluate_door_csi_sde_v2.sh)
echo "pretrain_job=$PRETRAIN"
echo "finetune_job=$FINETUNE"
echo "evaluation_job=$EVALUATE"
