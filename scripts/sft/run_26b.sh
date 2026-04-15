#!/bin/bash
# Llama-2-26B SFT (随机初始化, 8卡)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="/public/home/baidu_test/hygon_2030/py310"
cd "$PROJECT_ROOT"

source "$VENV_DIR/bin/activate"

# 8卡配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONUNBUFFERED=1

# 启动训练
mkdir -p logs/sft
paddleformers-cli train configs/sft/train_26b.yaml > logs/sft/26b_train.log 2>&1

echo "Llama-2-26B SFT completed."
