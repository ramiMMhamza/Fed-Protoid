#!/usr/bin/env bash
# Train Fed-Protoid: one initial round, then ROUNDS further communication rounds.
#
# Usage:
#   bash run_train.sh RUN_NAME [training options]
#
# Example:
#   bash run_train.sh msmt2market --epochs_s 1 --iters_s 0 --epochs_t 1 --times_itc 10
#
# Environment variables:
#   METHOD     FEDPROTO (ResNet-50, default) or FEDPROTO++ (ViT)
#   GPUS       number of GPUs / processes (default: 2)
#   ROUNDS     communication rounds run after the initial one (default: 800)
#   CONFIG     train config (default: OpenUnReID/tools/$METHOD/config.yaml)
#   PYTHON     python interpreter (default: python3)
#   SKIP_INIT  set to 1 to resume a run whose initial round is already done
#
# Paths (datasets, logs, weights) come from the config files and can be
# overridden with --data-root, --logs-root, --backbone-path and
# --source-pretrained. Relative paths are resolved from the repository root.
set -euo pipefail

if [ $# -lt 1 ]; then
    sed -n '2,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
fi
RUN_NAME=$1
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
METHOD=${METHOD:-FEDPROTO}
GPUS=${GPUS:-2}
ROUNDS=${ROUNDS:-800}
PYTHON=${PYTHON:-python3}
TOOLS="$REPO_ROOT/OpenUnReID/tools/$METHOD"
CONFIG=${CONFIG:-$TOOLS/config.yaml}

if [ ! -d "$TOOLS" ]; then
    echo "Unknown METHOD '$METHOD': expected FEDPROTO or FEDPROTO++" >&2
    exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/OpenUnReID${PYTHONPATH:+:$PYTHONPATH}"
# required by cuBLAS when deterministic algorithms are enabled
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:16:8}

# pick an unused tcp port for the process group
PORT=$("$PYTHON" -c "import socket; s = socket.socket(); s.bind(('', 0)); print(s.getsockname()[1])")

launch() {
    "$PYTHON" -m torch.distributed.run --nproc_per_node="$GPUS" --master_port="$PORT" \
        "$TOOLS/$1" "$CONFIG" --work-dir="$RUN_NAME" \
        --launcher=pytorch --tcp-port="$PORT" "${@:2}"
}

# Round 1: source-pretrained model -> first local trainings -> first aggregation.
if [ "${SKIP_INIT:-0}" != "1" ]; then
    launch main_fedl_init.py "$@"
fi

# Each further round runs in its own process and resumes from the aggregated
# checkpoint written by the previous one. Stops at the first failing round.
for round in $(seq 1 "$ROUNDS"); do
    echo "==> communication round $((round + 1))"
    launch main_fedl.py "$@"
done
