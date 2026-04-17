#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_SCRIPTS=(
  "pt/run.sh"
  "pt/run_26b.sh"
  "sft/run.sh"
  "sft/run_26b.sh"
  "lora/run.sh"
  "lora/run_26b.sh"
)

for relative_script in "${TRAIN_SCRIPTS[@]}"; do
  script_path="$SCRIPT_DIR/$relative_script"
  [ -f "$script_path" ] || { echo "Training script not found: $script_path" >&2; exit 1; }

  echo "============================================================"
  echo "Running $relative_script"
  echo "============================================================"

  bash "$script_path"
done

echo "All training scripts completed."
