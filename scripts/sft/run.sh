#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="/public/home/baidu_test/hygon_2030/py310"
cd "$PROJECT_ROOT"

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

# GPU/DCU 配置 (4卡)
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export PYTHONUNBUFFERED=1

# 分布式配置
mkdir -p logs/sft
paddleformers-cli train configs/sft/train.yaml > logs/sft/train.log 2>&1
