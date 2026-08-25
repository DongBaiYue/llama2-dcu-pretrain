#!/bin/bash
set -euo pipefail

module load compiler/dtk/25.04.4

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/../py310"

# 用法: run.sh [dcu|gpu]  (默认自动检测)
DEVICE="${1:-auto}"

if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DEVICE="gpu"
    elif command -v rocm-smi &>/dev/null; then
        DEVICE="dcu"
    else
        echo "ERROR: 无法自动检测设备类型，请手动指定: run.sh [dcu|gpu]" >&2
        exit 1
    fi
fi

case "$DEVICE" in
    dcu)
        CONFIG_FILE="configs/sft/train_llama3_8b.yaml"
        LOG_FILE="logs/sft/llama3-8b.log"
        DIST_LOG_DIR="logs/sft/llama3-8b.dist"
        ;;
    gpu)
        CONFIG_FILE="configs/sft/train_llama3_8b.yaml"
        LOG_FILE="logs/sft/llama3-8b_gpu.log"
        DIST_LOG_DIR="logs/sft/llama3-8b_gpu.dist"
        ;;
    *)
        echo "ERROR: 未知设备类型 '$DEVICE', 请使用 dcu 或 gpu" >&2
        exit 1
        ;;
esac

cd "$PROJECT_ROOT"

[ -f "$VENV_DIR/bin/activate" ] || { echo "Virtualenv activate script not found: $VENV_DIR/bin/activate" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Training config not found: $CONFIG_FILE" >&2; exit 1; }

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found in current environment" >&2; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/../Paddle/build/python:${PYTHONPATH:-}"

# GPU/DCU 配置 (4卡)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONUNBUFFERED=1

# 分布式配置
mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "DEVICE=$DEVICE"
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VENV_DIR=$VENV_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "LOG_FILE=$LOG_FILE"
echo "PADDLEFORMERS_DIST_LOG=$PADDLEFORMERS_DIST_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTHONPATH=$PYTHONPATH"

export FLAGS_deterministic_rng=1
paddleformers-cli train "$CONFIG_FILE" > "$LOG_FILE" 2>&1
