#!/bin/bash
# Slurm job running run_train.sh. Submit from the repository root:
#
#   mkdir -p logs && sbatch Fedprotoid.sh
#
# Adapt the resources below (and add --partition / --account) to your cluster.
# Environment variables understood by this script:
#   RUN_NAME   name of the experiment (default: FEDPROTOID_S_M)
#   CONDA_ENV  conda environment to activate (default: none)
# plus the ones documented in run_train.sh (METHOD, ROUNDS, ...).

#SBATCH --job-name=fedprotoid
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --ntasks=1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if [ -n "${CONDA_ENV:-}" ]; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
fi

export GPUS=${GPUS:-${SLURM_GPUS:-2}}
bash run_train.sh "${RUN_NAME:-FEDPROTOID_S_M}" \
    --epochs_s 1 --iters_s 0 --epochs_t 1 --iters_t 0 --times_itc 10
