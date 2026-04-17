#!/bin/bash
# Llama-2-26B 预训练 (2倍扩张: 13B -> 26B)
# 架构: 80层, 5120 hidden_size (深度翻倍)
# 并行: TP=4, PP=2, 共8卡
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="/public/home/baidu_test/hygon_2030/py310"
CONFIG_FILE="configs/pt/train_26b.yaml"
LOG_FILE="logs/pt/26b.log"
DIST_LOG_DIR="logs/pt/26b.dist"

cd "$PROJECT_ROOT"

[ -f "$VENV_DIR/bin/activate" ] || { echo "Virtualenv activate script not found: $VENV_DIR/bin/activate" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Training config not found: $CONFIG_FILE" >&2; exit 1; }

source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found in current environment" >&2; exit 1; }

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT/../Paddle/build/python:${PYTHONPATH:-}"

# 8卡配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONUNBUFFERED=1

# 启动训练
mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "VENV_DIR=$VENV_DIR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "LOG_FILE=$LOG_FILE"
echo "PADDLEFORMERS_DIST_LOG=$PADDLEFORMERS_DIST_LOG"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PYTHONPATH=$PYTHONPATH"

paddleformers-cli train "$CONFIG_FILE" > "$LOG_FILE" 2>&1

echo "Llama-2-26B pretrain completed."
