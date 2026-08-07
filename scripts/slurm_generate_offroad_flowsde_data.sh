#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=offroad_data_v2
#SBATCH --partition=l40s
#SBATCH --qos=l40s-cpu
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/zhangyi/bayesinf_jax/data/offroad_flowsde/logs/generate_v2_%j.out
#SBATCH --error=/home/zhangyi/bayesinf_jax/data/offroad_flowsde/logs/generate_v2_%j.err

set -euo pipefail

PROJECT=/home/zhangyi/bayesinf_jax
PYTHON=/home/zhangyi/myenv_312/bin/python
OUTPUT="$PROJECT/data/offroad_flowsde/offroad_v2.h5"
DEPS="$PROJECT/data/offroad_flowsde/.slurm_python_packages"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1

cd "$PROJECT"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3.12 || command -v python3)"
fi
mkdir -p "$DEPS"
export PYTHONPATH="$PROJECT:$DEPS:${PYTHONPATH:-}"
if ! "$PYTHON" -c 'import h5py,mujoco,numpy' 2>/dev/null; then
  "$PYTHON" -m pip install --target "$DEPS" \
    h5py==3.16.0 mujoco==3.10.0 numpy==2.4.4
fi
"$PYTHON" -c 'import h5py,mujoco,numpy; print("generator environment ready")'
"$PYTHON" -m experiments.offroad_flowsde.dataset --output "$OUTPUT"
"$PYTHON" -c 'import h5py,numpy as np; p="data/offroad_flowsde/offroad_v2.h5"; f=h5py.File(p); assert f["state"].shape==(2000000,10); assert f["action"].shape==(2000000,2); assert f["window_start"].shape==(380000,); assert np.bincount(f["split"][:],minlength=3).tolist()==[1400000,300000,300000]; assert np.isfinite(f["state"][:]).all(); print("validated",p); f.close()'
