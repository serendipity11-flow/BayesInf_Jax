#!/bin/bash
set -euo pipefail

REMOTE="${REMOTE:-zhangyi@licca-li-01.rz.uni-augsburg.de}"
REMOTE_DIR="${REMOTE_DIR:-bayesinf_jax}"
LOCAL_DIR="${LOCAL_DIR:-/home/zhangyi/bayesinf_jax}"
INCLUDE_RESULTS="${INCLUDE_RESULTS:-0}"

EXCLUDES=(
  "--exclude=.git/"
  "--exclude=__pycache__/"
  "--exclude=.pytest_cache/"
  "--exclude=.mypy_cache/"
  "--exclude=.ipynb_checkpoints/"
)

if [[ "$INCLUDE_RESULTS" != "1" ]]; then
  EXCLUDES+=("--exclude=results/")
fi

ssh "$REMOTE" "mkdir -p '$REMOTE_DIR'"

rsync -az --info=progress2 "${EXCLUDES[@]}" \
  "$LOCAL_DIR"/ \
  "$REMOTE":"$REMOTE_DIR"/

cat <<EOF

Synced project to:
  $REMOTE:$REMOTE_DIR

On the server, rebuild/activate the environment with:

  cd $REMOTE_DIR
  /usr/bin/python3.12 -m venv ~/myenv_312
  source ~/myenv_312/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r requirements_myenv_312.txt

Then submit jobs, for example:

  sbatch slurm_flow_sde.sh

EOF
