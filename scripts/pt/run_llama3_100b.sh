#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 用法: run_llama3_100b.sh [dcu|gpu]
# 4 节点 32 卡: TP=8, PP=4, DP=1

DEVICE="${1:-auto}"

if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DEVICE="gpu"
    elif command -v rocm-smi &>/dev/null; then
        DEVICE="dcu"
    else
        echo "ERROR: 无法自动检测设备类型，请手动指定: run_llama3_100b.sh [dcu|gpu]" >&2
        exit 1
    fi
fi

case "$DEVICE" in
    dcu)
        CONFIG_FILE="configs/pt/train_llama3_100b.yaml"
        LOG_FILE="logs/pt/llama3-100b.log"
        DIST_LOG_DIR="logs/pt/llama3-100b.dist"
        VENV_DIR="$PROJECT_ROOT/../py310"
        ;;
    gpu)
        CONFIG_FILE="configs/pt/train_llama3_100b.yaml"
        LOG_FILE="logs/pt/llama3-100b_gpu.log"
        DIST_LOG_DIR="logs/pt/llama3-100b_gpu.dist"
        VENV_DIR="$PROJECT_ROOT/../py312"
        ;;
    *)
        echo "ERROR: 未知设备类型 '$DEVICE', 请使用 dcu 或 gpu" >&2
        exit 1
        ;;
esac

cd "$PROJECT_ROOT"

[ -f "$VENV_DIR/bin/activate" ] || { echo "Virtualenv not found: $VENV_DIR/bin/activate" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }

# 停掉正在运行的训练进程
if [ "$DEVICE" = "dcu" ]; then
    STOP_HOSTS="${HOSTS:-f09r4n17,f09r4n18,f09r4n19,f09r4n20}"
else
    STOP_HOSTFILE="${HOSTFILE:-/root/paddlejob/workspace/hostfile}"
    STOP_HOSTS="${HOSTS:-$(awk '{printf "%s%s", sep, $1; sep=","}' "$STOP_HOSTFILE")}"
fi
for h in $(echo "$STOP_HOSTS" | tr ',' ' '); do
    ssh "$h" "pkill -f paddleformers.cli.launcher.*llama3_100b" 2>/dev/null || true
done
sleep 5

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found" >&2; exit 1; }

# 分布式配置
GPUS="0,1,2,3,4,5,6,7"
if [ "$DEVICE" = "dcu" ]; then
    HOSTS="${HOSTS:-f09r4n17,f09r4n18,f09r4n19,f09r4n20}"
else
    HOSTFILE="${HOSTFILE:-/root/paddlejob/workspace/hostfile}"
    HOSTS="${HOSTS:-$(awk '{printf "%s%s", sep, $1; sep=","}' "$HOSTFILE")}"
fi
NNODES=$(echo "$HOSTS" | tr ',' '\n' | wc -l)
MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"

export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "========== Llama-3-100B PT =========="
echo "DEVICE=$DEVICE"
echo "HOSTS=$HOSTS"
echo "NNODES=$NNODES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "LOG_FILE=$LOG_FILE"
echo "======================================="

if [ "$DEVICE" = "dcu" ]; then
    mpirun -H "$HOSTS" \
        -x NNODES="$NNODES" \
        -x MASTER_ADDR="$MASTER_ADDR" \
        -x NCCL_SOCKET_IFNAME=ib0 \
        -x NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1 \
        -x NCCL_IB_DISABLE=0 \
        -x NCCL_NET_PLUGIN=shca \
        -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
        -x PYTHONUNBUFFERED=1 \
        -x FLAGS_deterministic_rng=1 \
        -x CUDA_VISIBLE_DEVICES="$GPUS" \
        -x PYTHONPATH \
        bash -lc "export RANK=\${OMPI_COMM_WORLD_RANK} && source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && paddleformers-cli train '$CONFIG_FILE' tensorwise_offload_optimizer=true" \
        >"$LOG_FILE" 2>&1
else
    # GPU: 仿照 DCU 方式，使用 orterun -H
    /usr/local/openmpi-4.1.5/bin/orterun --allow-run-as-root -bind-to none -tag-output -H "${HOSTS//,/:1,}:1" \
        -x NNODES="$NNODES" \
        -x MASTER_ADDR="$MASTER_ADDR" \
        -x MASTER_PORT=38888 \
        -x PYTHONUNBUFFERED=1 \
        -x FLAGS_deterministic_rng=1 \
        -x CUDA_VISIBLE_DEVICES="$GPUS" \
        -x PYTHONPATH \
        bash -c "export RANK=\${OMPI_COMM_WORLD_RANK} && export PADDLEFORMERS_DIST_LOG=$DIST_LOG_DIR/node_\${RANK} && source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && paddleformers-cli train '$CONFIG_FILE' tensorwise_offload_optimizer=false" \
        >"$LOG_FILE" 2>&1
fi
