#!/bin/bash
# Llama-2-52B 双机预训练 (node109 + node110, 共16卡)
# 并行: TP=8, PP=2; 每机 8 卡; 使用 mpirun 启动两机
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-/public/home/baidu_test/hygon_2030/py310}"
CONFIG_FILE="${CONFIG_FILE:-configs/pt/train_52b.yaml}"
LOG_DIR="${LOG_DIR:-logs/pt/52b}"
LOG_FILE="${LOG_FILE:-logs/pt/52b.log}"
DIST_LOG_DIR="${DIST_LOG_DIR:-logs/pt/52b.dist}"
HOSTS="${HOSTS:-node109:1,node110:1}"
NNODES="${NNODES:-2}"
NUM_PROCS="${NUM_PROCS:-$NNODES}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-29500}"
GPUS_PER_NODE="${GPUS_PER_NODE:-0,1,2,3,4,5,6,7}"
PYTHONPATH_OVERRIDE="${PYTHONPATH_OVERRIDE:-$PROJECT_ROOT/../Paddle/build/python}"
TRAIN_ARGS="${TRAIN_ARGS:-}"

cd "$PROJECT_ROOT"

[ -f "$VENV_DIR/bin/activate" ] || { echo "Virtualenv activate script not found: $VENV_DIR/bin/activate" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Training config not found: $CONFIG_FILE" >&2; exit 1; }
[ -f "$PROJECT_ROOT/models/Llama-2-52B/config.json" ] || {
  echo "52B model config not found: $PROJECT_ROOT/models/Llama-2-52B/config.json" >&2
  echo "Run scripts/pt/prepare_52b_model.sh first." >&2
  exit 1
}
command -v mpirun >/dev/null 2>&1 || { echo "mpirun not found" >&2; exit 1; }

source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found in current environment" >&2; exit 1; }

if [ -z "$MASTER_ADDR" ]; then
  if command -v hostname >/dev/null 2>&1; then
    MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
fi
[ -n "$MASTER_ADDR" ] || { echo "Failed to resolve MASTER_ADDR automatically; please export MASTER_ADDR=<master_ip>." >&2; exit 1; }

mkdir -p "$LOG_DIR" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VENV_DIR=$VENV_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "LOG_FILE=$LOG_FILE"
echo "PADDLEFORMERS_DIST_LOG=$PADDLEFORMERS_DIST_LOG"
echo "HOSTS=$HOSTS"
echo "NNODES=$NNODES"
echo "NUM_PROCS=$NUM_PROCS"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES=$GPUS_PER_NODE"
echo "PYTHONPATH=$PYTHONPATH_OVERRIDE"
echo "TRAIN_ARGS=$TRAIN_ARGS"

mpirun \
  -np "$NUM_PROCS" \
  -H "$HOSTS" \
  --bind-to none \
  --map-by ppr:1:node \
  -x PYTHONUNBUFFERED=1 \
  -x FLAGS_deterministic_rng=1 \
  -x CUDA_VISIBLE_DEVICES="$GPUS_PER_NODE" \
  -x PYTHONPATH="$PYTHONPATH_OVERRIDE" \
  -x NNODES="$NNODES" \
  -x MASTER_ADDR="$MASTER_ADDR" \
  -x MASTER_PORT="$MASTER_PORT" \
  -x TRAIN_ARGS="$TRAIN_ARGS" \
  -x PADDLEFORMERS_DIST_LOG="$PADDLEFORMERS_DIST_LOG" \
  -x PATH \
  -x LD_LIBRARY_PATH \
  bash -lc "export RANK=\${OMPI_COMM_WORLD_RANK}; source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && echo NODE=\$(hostname) RANK=\$RANK NNODES=\$NNODES MASTER_ADDR=\$MASTER_ADDR MASTER_PORT=\$MASTER_PORT CUDA_VISIBLE_DEVICES=\$CUDA_VISIBLE_DEVICES PYTHONPATH=\$PYTHONPATH TRAIN_ARGS=\$TRAIN_ARGS && paddleformers-cli train '$CONFIG_FILE' \$TRAIN_ARGS" \
  > "$LOG_FILE" 2>&1

echo "Llama-2-52B 2-node pretrain completed."
