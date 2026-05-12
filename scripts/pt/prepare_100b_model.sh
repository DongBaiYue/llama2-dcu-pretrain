#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODEL_DIR="$PROJECT_ROOT/models/Llama-3-100b"
SRC_DIR="$PROJECT_ROOT/models/Llama-3-70b"

if [ ! -d "$SRC_DIR" ]; then
    echo "ERROR: 源模型目录不存在: $SRC_DIR" >&2
    echo "请先下载 Llama-3-70B 模型" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

# 复制 tokenizer 相关文件
cp "$SRC_DIR/tokenizer.model" "$MODEL_DIR/"
cp "$SRC_DIR/tokenizer_config.json" "$MODEL_DIR/"
cp "$SRC_DIR/special_tokens_map.json" "$MODEL_DIR/"
[ -f "$SRC_DIR/added_tokens.json" ] && cp "$SRC_DIR/added_tokens.json" "$MODEL_DIR/"
[ -f "$SRC_DIR/generation_config.json" ] && cp "$SRC_DIR/generation_config.json" "$MODEL_DIR/"

# 基于 70B config 生成 100B config，深度 80→120
python3 -c "
import json
with open('$SRC_DIR/config.json') as f:
    cfg = json.load(f)
cfg['num_hidden_layers'] = 120
with open('$MODEL_DIR/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)
"

echo "已生成 $MODEL_DIR/config.json"
echo "num_hidden_layers: 80 → 120 (≈104B 参数)"
