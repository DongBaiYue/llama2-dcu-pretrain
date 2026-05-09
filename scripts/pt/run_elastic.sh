#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
VENV_DIR="$PROJECT_ROOT/../py310"

# 用法: run_elastic.sh <scale> [dcu|gpu]
#   scale: 4card | 8card | 2node
#   扩容: 4card → 8card → 2node
#   缩容: 2node → 8card → 4card

SCALE="${1:?用法: run_elastic.sh <4card|8card|2node> [dcu|gpu]}"
DEVICE="${2:-auto}"

if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DEVICE="gpu"
    elif command -v rocm-smi &>/dev/null; then
        DEVICE="dcu"
    else
        echo "ERROR: 无法自动检测设备类型，请手动指定" >&2; exit 1
    fi
fi

CONFIG_FILE="configs/pt/train_elastic.yaml"
GPUS="0,1,2,3,4,5,6,7"

case "$SCALE" in
    4card)
        export CUDA_VISIBLE_DEVICES="0,1,2,3"
        SHARDING_SIZE=1
        ;;
    8card)
        export CUDA_VISIBLE_DEVICES="$GPUS"
        SHARDING_SIZE=2
        ;;
    2node)
        SHARDING_SIZE=2
        ;;
    *)
        echo "ERROR: 未知规模 '$SCALE', 请使用 4card、8card 或 2node" >&2
        exit 1
        ;;
esac

# 停掉正在运行的训练进程
if [ "$SCALE" = "2node" ]; then
    mpirun -H f09r2n15:1,f09r2n16:1 pkill -f paddleformers.cli.launcher.*train_elastic || true
    echo "已停止训练进程"
else
    pkill -f "paddleformers.cli.launcher.*train_elastic" 2>/dev/null && echo "已停止训练进程" || true
fi
sleep 5

# 自动修改YAML: sharding_parallel_size
sed -i "s|sharding_parallel_size: [0-9]*|sharding_parallel_size: $SHARDING_SIZE|" "$CONFIG_FILE"
echo "已设置 sharding_parallel_size=$SHARDING_SIZE"

# 自动设置 resume_from_checkpoint
LATEST=$(ls -d checkpoints/pt/13b_elastic/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -n "$LATEST" ]; then
    echo "检测到已有checkpoint: $LATEST"
    sed -i "/^resume_from_checkpoint:/d" "$CONFIG_FILE"
    sed -i "/^# checkpoint$/a resume_from_checkpoint: $LATEST" "$CONFIG_FILE"
    echo "已设置 resume_from_checkpoint=$LATEST"
else
    echo "WARNING: 未检测到checkpoint，将从初始权重开始训练"
fi

cd "$PROJECT_ROOT"

[ -f "$VENV_DIR/bin/activate" ] || { echo "Virtualenv not found: $VENV_DIR/bin/activate" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }

source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found" >&2; exit 1; }
export PYTHONPATH="$PROJECT_ROOT/../Paddle/build/python:${PYTHONPATH:-}"

# 日志按规模+阶段区分（用checkpoint step，首次无checkpoint时为0）
STEP=$(echo "$LATEST" | grep -oP '\d+$' 2>/dev/null || true)
STEP=${STEP:-0}
LOG_FILE="logs/pt/13b_elastic_${SCALE}_step${STEP}.log"
DIST_LOG_DIR="logs/pt/13b_elastic_${SCALE}_step${STEP}.dist"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "========== 弹性扩缩容训练 =========="
echo "DEVICE=$DEVICE  SCALE=$SCALE  CONFIG=$CONFIG_FILE"
echo "======================================"

if [ "$SCALE" = "2node" ]; then
    MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
    mpirun -H f09r2n15,f09r2n16 \
        -x NNODES=2 \
        -x MASTER_ADDR="$MASTER_ADDR" \
        -x NCCL_SOCKET_IFNAME=ib0 \
        -x NCCL_IB_HCA=shca_0 \
        -x NCCL_IB_DISABLE=0 \
        -x HSA_FORCE_FINE_GRAIN_PCIE=1 \
        -x PYTHONUNBUFFERED=1 \
        -x CUDA_VISIBLE_DEVICES="$GPUS" \
        -x PYTHONPATH \
        bash -lc "export RANK=\${OMPI_COMM_WORLD_RANK} && source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && paddleformers-cli train '$CONFIG_FILE'" \
        2>&1 | tee "$LOG_FILE"
else
    paddleformers-cli train "$CONFIG_FILE" 2>&1 | tee "$LOG_FILE"
fi
