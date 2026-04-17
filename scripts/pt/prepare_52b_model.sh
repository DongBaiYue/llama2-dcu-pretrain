#!/bin/bash
# 基于 13B 配置生成 52B 模型目录（仅复制 tokenizer / config，不复制权重索引）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="${VENV_DIR:-/public/home/baidu_test/hygon_2030/py310}"
BASE_MODEL_DIR="$PROJECT_ROOT/models/Llama-2-13b"
TARGET_MODEL_DIR="$PROJECT_ROOT/models/Llama-2-52B"
PYTHON_BIN="${PYTHON_BIN:-}"

cd "$PROJECT_ROOT"

[ -d "$BASE_MODEL_DIR" ] || { echo "Base model dir not found: $BASE_MODEL_DIR" >&2; exit 1; }
[ -f "$BASE_MODEL_DIR/config.json" ] || { echo "Base config not found: $BASE_MODEL_DIR/config.json" >&2; exit 1; }

if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "No usable python found (checked $VENV_DIR/bin/python, python3, python)" >&2
    exit 1
  fi
fi

mkdir -p "$TARGET_MODEL_DIR"

for file in added_tokens.json generation_config.json special_tokens_map.json tokenizer_config.json tokenizer.model; do
  [ -f "$BASE_MODEL_DIR/$file" ] || { echo "Required model asset not found: $BASE_MODEL_DIR/$file" >&2; exit 1; }
  cp -f "$BASE_MODEL_DIR/$file" "$TARGET_MODEL_DIR/"
done

BASE_MODEL_DIR="$BASE_MODEL_DIR" TARGET_MODEL_DIR="$TARGET_MODEL_DIR" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

base_model_dir = Path(os.environ["BASE_MODEL_DIR"])
target_model_dir = Path(os.environ["TARGET_MODEL_DIR"])

config = json.loads((base_model_dir / "config.json").read_text(encoding="utf-8"))
config["_name_or_path"] = "./models/Llama-2-52B"
config["hidden_size"] = 7168
config["intermediate_size"] = 19456
config["num_hidden_layers"] = 80
config["num_attention_heads"] = 56
config["num_key_value_heads"] = 56

(target_model_dir / "config.json").write_text(
    json.dumps(config, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

echo "Prepared $TARGET_MODEL_DIR"
