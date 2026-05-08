#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/../py310"

# 用法: run_elastic.sh <scale> [dcu|gpu]
#   scale: 4card | 8card
#   示例:
#     run_elastic.sh 4card gpu    # 阶段1: 4卡训练
#     run_elastic.sh 8card gpu    # 阶段2: 扩容到8卡，自动从checkpoint恢复

SCALE="${1:?用法: run_elastic.sh <4card|8card> [dcu|gpu]}"
DEVICE="${2:-auto}"

if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DEVICE="gpu"
    elif command -v rocm-smi &>/dev/null; then
        DEVICE="dcu"
    else
        echo "ERROR: 无法自动检测设备类型，请手动指定: run_elastic.sh <4card|8card> [dcu|gpu]" >&2
        exit 1
    fi
fi

CONFIG_FILE="configs/pt/train_elastic.yaml"

case "$SCALE" in
    4card)
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        LOG_FILE="logs/pt/13b_elastic_phase1.log"
        DIST_LOG_DIR="logs/pt/13b_elastic_phase1.dist"
        # 4卡: TP=2, PP=2, DP=1
        ;;
    8card)
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
        LOG_FILE="logs/pt/13b_elastic_phase2.log"
        DIST_LOG_DIR="logs/pt/13b_elastic_phase2.dist"
        # 8卡: TP=2, PP=2, DP=2
        # 自动设置 resume_from_checkpoint
        LATEST=$(ls -d checkpoints/pt/13b_elastic/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
        if [ -n "$LATEST" ]; then
            echo "检测到已有checkpoint: $LATEST"
            sed -i "s|^#* *resume_from_checkpoint:.*|resume_from_checkpoint: $LATEST|" "$CONFIG_FILE"
            echo "已自动修改YAML: resume_from_checkpoint=$LATEST"
        else
            echo "WARNING: 未检测到checkpoint，将从初始权重开始训练"
        fi
        ;;
    *)
        echo "ERROR: 未知规模 '$SCALE', 请使用 4card 或 8card" >&2
        exit 1
        ;;
esac

cd "$PROJECT_ROOT"

[ -f "$VENV_DIR/bin/activate" ] || { echo "Virtualenv activate script not found: $VENV_DIR/bin/activate" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Training config not found: $CONFIG_FILE" >&2; exit 1; }

source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found in current environment" >&2; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/../Paddle/build/python:${PYTHONPATH:-}"

export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "========== 弹性扩缩容训练 =========="
echo "DEVICE=$DEVICE"
echo "SCALE=$SCALE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "OUTPUT_DIR=./checkpoints/pt/13b_elastic"
echo "======================================"

paddleformers-cli train "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
