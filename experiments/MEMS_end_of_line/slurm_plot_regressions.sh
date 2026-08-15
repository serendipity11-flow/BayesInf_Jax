#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=mems_eol_plots
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/experiments/MEMS_end_of_line/logs/%x_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/experiments/MEMS_end_of_line/logs/%x_%j.err

set -euo pipefail
PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

/usr/bin/python3 -m experiments.MEMS_end_of_line.plot_regressions \
    --data-dir /home/zhangyi/data/MEMS/MEMS_simulated \
    --results-dir results/results_parameter_identification/MEMS_end_of_line/slurm_seed0 \
    --models bnn vae realnvp nsf fm ddpm sbm \
    --seed 0
