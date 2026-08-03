#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=jackal_open_loop
#SBATCH --partition=a30
#SBATCH --qos=a30-cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --output=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_ood/jackal_open_loop_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/results/results_optimal_control/jackal_ood/jackal_open_loop_%j.err

set -euo pipefail

PROJECT_ROOT=/home/zhangyi/bayesinf_jax
cd "$PROJECT_ROOT"

export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/matplotlib-${USER}-${SLURM_JOB_ID}"
mkdir -p "$MPLCONFIGDIR"

if "$HOME/myenv_312/bin/python" -c 'import jax, equinox, matplotlib' >/dev/null 2>&1; then
    PYTHON="$HOME/myenv_312/bin/python"
else
    PYTHON="$(command -v python3.12)"
fi

"$PYTHON" -m bayesinf.applications.optimal_control.evaluate_jackal_ood_cem \
    --checkpoint "$PROJECT_ROOT/results/results_optimal_control/jackal_ood/jackal_latent_ssm_flow.eqx" \
    --output "$PROJECT_ROOT/results/results_optimal_control/jackal_ood/open_loop_reconstruction" \
    --open-loop-only
