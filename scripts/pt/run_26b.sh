#!/bin/bash
# Llama-2-26B 预训练 (2倍扩张: 13B -> 26B)
# 架构: 80层, 5120 hidden_size (深度翻倍)
# 并行: TP=4, PP=2, 共8卡

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="/public/home/baidu_test/hygon_2030/py310"
cd "$PROJECT_ROOT"

source "$VENV_DIR/bin/activate"

# 设置环境变量
export PYTHONPATH="$PROJECT_ROOT/PaddleFormers:$PYTHONPATH"

# 8卡配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export PYTHONUNBUFFERED=1

# 启动训练
mkdir -p logs/pt
paddleformers-cli train configs/pt/train_26b.yaml > logs/pt/26b_train.log 2>&1

echo "Llama-2-26B pretrain completed."
