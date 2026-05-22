#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG_FILE="configs/pt/train_llama3_8b_mini.yaml"
LOG_FILE="logs/pt/llama3-8b_mini_gpu.log"
DIST_LOG_DIR="logs/pt/llama3-8b_mini_gpu.dist"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/../py312}"
PADDLEFORMERS_DIR="${PADDLEFORMERS_DIR:-$PROJECT_ROOT/../PaddleFormers}"
HOSTFILE="${HOSTFILE:-/root/paddlejob/workspace/hostfile}"
ORTERUN="${ORTERUN:-/usr/local/openmpi-4.1.5/bin/orterun}"
MASTER_PORT="${MASTER_PORT:-38888}"

cd "$PROJECT_ROOT"
source "$VENV_DIR/bin/activate"
export PYTHONPATH="$PADDLEFORMERS_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPUS" PYTHONUNBUFFERED=1 PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR" FLAGS_deterministic_rng=1

mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"

sed -i '/^resume_from_checkpoint:/d' "$CONFIG_FILE"
sed -i 's/^max_steps:.*/max_steps: 20/' "$CONFIG_FILE"
sed -i 's/^gradient_accumulation_steps:.*/gradient_accumulation_steps: 4/' "$CONFIG_FILE"
sed -i 's/^save_checkpoint_format:.*/save_checkpoint_format: flex_checkpoint/' "$CONFIG_FILE"

HOSTS="${HOSTS:-$(awk '!/^#/ && NF {printf "%s%s", sep, $1; sep=","}' "$HOSTFILE")}"
LOCAL_IDS="$(hostname 2>/dev/null || true) $(hostname -s 2>/dev/null || true) $(hostname -f 2>/dev/null || true) $(hostname -I 2>/dev/null || true) localhost 127.0.0.1"

is_local() {
    [[ " $LOCAL_IDS " == *" $1 "* ]]
}

is_idle() {
    local host="$1" cmd="nvidia-smi | grep -q 'No running processes found'"
    if is_local "$host"; then
        bash -lc "$cmd"
    else
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$host" "$cmd"
    fi
}

SELECTED_HOST=""
for pass in local remote; do
    IFS=',' read -ra HOST_ITEMS <<< "$HOSTS"
    for item in "${HOST_ITEMS[@]}"; do
        host="${item%%:*}"
        [ -n "$host" ] || continue
        if { [ "$pass" = local ] && is_local "$host"; } || { [ "$pass" = remote ] && ! is_local "$host"; }; then
            echo "Checking host idle: $host" >&2
            if is_idle "$host"; then
                SELECTED_HOST="$host"
                break 2
            fi
        fi
    done
done

[ -n "$SELECTED_HOST" ] || { echo "No idle GPU host found" >&2; exit 1; }

{
    echo "SELECTED_HOST=$SELECTED_HOST"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    echo "CONFIG_FILE=$CONFIG_FILE"
    echo "LOG_FILE=$LOG_FILE"
} | tee "$LOG_FILE"

"$ORTERUN" --allow-run-as-root -bind-to none -tag-output -H "$SELECTED_HOST:1" \
    -x NNODES=1 \
    -x RANK=0 \
    -x MASTER_ADDR="$SELECTED_HOST" \
    -x MASTER_PORT="$MASTER_PORT" \
    -x PYTHONUNBUFFERED=1 \
    -x FLAGS_deterministic_rng=1 \
    -x CUDA_VISIBLE_DEVICES="$GPUS" \
    -x PYTHONPATH \
    bash -lc "export PADDLEFORMERS_DIST_LOG='$DIST_LOG_DIR/node_0' && source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && paddleformers-cli train '$CONFIG_FILE'" \
    >> "$LOG_FILE" 2>&1
