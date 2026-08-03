#!/usr/bin/env bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=door_eval
#SBATCH --partition=a30
#SBATCH --qos=a30-gpu
#SBATCH --gres=gpu:a30:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=results/results_optimal_control/door/generative_policy_evaluation/slurm-%j.out

set -euo pipefail

cd /home/zhangyi/bayesinf_jax
mkdir -p results/results_optimal_control/door/generative_policy_evaluation

export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/zhangyi/bayesinf_jax:$HOME/myenv_312/lib/python3.12/site-packages:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export MUJOCO_GL=egl
export MPLCONFIGDIR=/tmp/matplotlib-cache-"${SLURM_JOB_ID}"

PYTHON=""
for candidate in \
    "$HOME/myenv_312/bin/python" \
    /usr/bin/python3.12 \
    /usr/local/bin/python3.12
do
    if [[ -x "$candidate" ]] && "$candidate" -c 'import sys' >/dev/null 2>&1; then
        PYTHON="$candidate"
        break
    fi
done
if [[ -z "$PYTHON" ]]; then
    echo "No working Python 3.12 interpreter found on $(hostname)" >&2
    exit 127
fi

echo "job_id=$SLURM_JOB_ID host=$(hostname) python=$PYTHON"
nvidia-smi || true

"$PYTHON" -m \
  experiments.cond_stoch_interpolant_dppo.evaluate_across_datasets \
  --manifest \
  experiments/cond_stoch_interpolant_dppo/config/evaluation_door.json
