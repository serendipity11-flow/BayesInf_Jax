#!/bin/bash
#SBATCH --account=Mitarbeiter
#SBATCH --job-name=eval_si
#SBATCH --partition=a30
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=yi.zhang@uni-a.de
#SBATCH --time=36:00:00

# Request memory per CPU
#SBATCH --mem-per-cpu=32G
# Number of GPUs, in the format "gpu:[1-4]" or "gpu:a30:[1-4]":
#SBATCH --gres=gpu:a30:1
# Request n CPUs for your task.
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --qos=a30-gpu

export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=${TMPDIR:-/tmp}/matplotlib-${USER}
mkdir -p "$MPLCONFIGDIR"
VENV_SITE=$HOME/myenv_312/lib/python3.12/site-packages
export PYTHONPATH=/home/zhangyi/bayesinf_jax:$VENV_SITE:$PYTHONPATH
if "$HOME/myenv_312/bin/python" -c 'import sys' >/dev/null 2>&1; then
    PYTHON=$HOME/myenv_312/bin/python
else
    PYTHON=$(command -v python3.12 || ls /usr/bin/python3.12 /usr/local/bin/python3.12 2>/dev/null | head -1)
fi
echo "Using Python: $PYTHON  ($($PYTHON --version))"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"

# ---------------------------------------------------------------------------
# Improved SI: masked source distribution + ỹ-conditioned velocity/score nets
#   Observed dims  : z_0 = ỹ (tight, deterministic source)
#   Unobserved dims: z_0 ~ N(0,1) per particle (diverse Gaussian prior)
#   vel_net / score_net both see ỹ explicitly as conditioning signal
# ---------------------------------------------------------------------------
DATA_ARGS="
    --N 12 --obs_interval 2 --obs_std 0.5
    --horizon 101 --dt 0.02 --F_lorenz 8.0
    --spinup_steps 500 --n_samples 2000"

NOISE_ARGS="
    --obs_noise_type gmm
    --gmm_weights 0.70,0.15,0.15
    --gmm_offsets 0.0,2.5,-2.5
    --gmm_stds 0.50,1.50,1.50"

ARCH_ARGS="
    --d_c 128 --d_hidden_enc 128
    --d_hidden 256 --n_heads 4 --depth 4
    --time_emb_dim 32"

EVAL_ARGS="
    $DATA_ARGS $NOISE_ARGS
    --d_c 128 --d_hidden 256 --n_heads 4 --depth 4
    --time_emb_dim 32
    --sigma_0 1.0 --g0 0.5 --score_weight 1.0 --s_min 0.02
    --interpolant_type linear
    --seed 0
    --n_eval 100 --n_plot_seqs 6
    --n_particles 200 --n_steps 100
    --ci 90.0 --t_hist 0.5
    --sampler segmented"

for SEG in 5 10 20 25; do
    echo "========================================"
    echo "  SI  seg_len=${SEG}  linear interpolant  masked-source  ỹ-conditioned"
    echo "========================================"

    $PYTHON -m bayesinf.applications.state_estimation.eval_si_lorenz96 \
        $EVAL_ARGS \
        --ms --seg_len ${SEG} \
        --seg_len_eval ${SEG}
done
