#!/bin/bash
set -euo pipefail

module load compiler/dtk/25.04.4
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 用法: run_llama3_8b_elastic.sh <2node|4node|8node> [dcu|gpu]
#   扩容: 2node → 4node → 8node (DP 1→2→4)
#   缩容: 8node → 4node → 2node
# 节点获取与 run_llama3_100b.sh 对齐: DCU 用 srun 动态获取, GPU 从 hostfile 取前 NNODES 个

SCALE="${1:?用法: run_llama3_8b_elastic.sh <2node|4node|8node> [dcu|gpu]}"
DEVICE="${2:-auto}"

if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DEVICE="gpu"
    elif command -v rocm-smi &>/dev/null; then
        DEVICE="dcu"
    else
        echo "ERROR: 无法自动检测设备类型，请手动指定: run_llama3_8b_elastic.sh <2node|4node|8node> [dcu|gpu]" >&2
        exit 1
    fi
fi

GPUS="0,1,2,3,4,5,6,7"

# case 块结构与 run_llama3_100b.sh 对齐: 按设备设置 CONFIG_FILE / VENV_DIR
case "$DEVICE" in
    dcu)
        CONFIG_FILE="configs/pt/train_llama3_8b_elastic.yaml"
        VENV_DIR="$PROJECT_ROOT/../py310"
        ;;
    gpu)
        CONFIG_FILE="configs/pt/train_llama3_8b_elastic.yaml"
        VENV_DIR="$PROJECT_ROOT/../py312"
        ;;
    *)
        echo "ERROR: 未知设备类型 '$DEVICE', 请使用 dcu 或 gpu" >&2
        exit 1
        ;;
esac

SHARDING_SIZE=4
SAVE_CHECKPOINT_FORMAT="sharding_io"

case "$SCALE" in
    2node)
        NNODES=2
        ;;
    4node)
        NNODES=4
        ;;
    8node)
        NNODES=8
        ;;
    *)
        echo "ERROR: 未知规模 '$SCALE', 请使用 2node、4node 或 8node" >&2
        exit 1
        ;;
esac

# 停掉正在运行的训练进程 (与 run_llama3_100b.sh 对齐)
if [ "$DEVICE" = "dcu" ]; then
    srun pkill -9 -f paddleformers.cli.launcher.*llama3_8b_elastic 2>/dev/null || true
else
    mpirun pkill -9 -f paddleformers.cli.launcher.*llama3_8b_elastic 2>/dev/null || true
fi
sleep 5

# 自动设置 resume_from_checkpoint
LATEST=$(ls -d checkpoints/pt/llama3-8b_elastic/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
STEP=$(echo "$LATEST" | grep -oP '\d+$' 2>/dev/null || true)
STEP=${STEP:-0}
if [ "$SCALE" = "2node" ] && [ "$STEP" -ge 60 ]; then
    SAVE_CHECKPOINT_FORMAT="flex_checkpoint"
fi

# 自动修改YAML: sharding_parallel_size / save_checkpoint_format
sed -i "s|sharding_parallel_size: [0-9]*|sharding_parallel_size: $SHARDING_SIZE|" "$CONFIG_FILE"
if grep -q "^save_checkpoint_format:" "$CONFIG_FILE"; then
    sed -i "s|^save_checkpoint_format:.*|save_checkpoint_format: $SAVE_CHECKPOINT_FORMAT|" "$CONFIG_FILE"
else
    sed -i "/^# checkpoint$/a save_checkpoint_format: $SAVE_CHECKPOINT_FORMAT" "$CONFIG_FILE"
fi
echo "已设置 sharding_parallel_size=$SHARDING_SIZE"
echo "已设置 save_checkpoint_format=$SAVE_CHECKPOINT_FORMAT"

if [ -n "$LATEST" ]; then
    echo "检测到已有checkpoint: $LATEST"
    sed -i "/^resume_from_checkpoint:/d" "$CONFIG_FILE"
    sed -i "/^# checkpoint$/a resume_from_checkpoint: $LATEST" "$CONFIG_FILE"
    echo "已设置 resume_from_checkpoint=$LATEST"
else
    echo "WARNING: 未检测到checkpoint，将从初始权重开始训练"
    # 删除 YAML 中残留的 resume_from_checkpoint, 避免误加载旧 checkpoint
    sed -i "/^resume_from_checkpoint:/d" "$CONFIG_FILE"
fi

cd "$PROJECT_ROOT"

[ -f "$VENV_DIR/bin/activate" ] || { echo "Virtualenv not found: $VENV_DIR/bin/activate" >&2; exit 1; }
[ -f "$CONFIG_FILE" ] || { echo "Config not found: $CONFIG_FILE" >&2; exit 1; }

source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found" >&2; exit 1; }

# 分布式配置: 按 NNODES 解析节点列表 (与 run_llama3_100b.sh 对齐)
MASTER_ADDR="$(hostname -I 2>/dev/null | awk '{print $1}')"
if [ "$DEVICE" = "dcu" ]; then
    HOSTS="$(srun -N "$NNODES" -n "$NNODES" hostname | paste -sd, -)"
else
    # GPU: 从 hostfile 取前 NNODES 个节点
    HOSTFILE="${HOSTFILE:-/root/paddlejob/workspace/hostfile}"
    [ -f "$HOSTFILE" ] || { echo "Hostfile not found: $HOSTFILE" >&2; exit 1; }
    HOSTS="$(awk '{printf "%s%s", sep, $1; sep=","}' "$HOSTFILE" | cut -d, -f1-"$NNODES")"
    HOST_COUNT="$(echo "$HOSTS" | tr ',' '\n' | wc -l)"
    if [ "$HOST_COUNT" -ne "$NNODES" ]; then
        echo "ERROR: hostfile 节点数不足 (需要 $NNODES, 实际 $HOST_COUNT)" >&2
        exit 1
    fi
fi

# 日志按规模+阶段区分
LOG_FILE="logs/pt/llama3-8b_elastic_${SCALE}_step${STEP}_${DEVICE}.log"
DIST_LOG_DIR="logs/pt/llama3-8b_elastic_${SCALE}_step${STEP}_${DEVICE}.dist"
export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "========== Llama-3-8B 弹性扩缩容 =========="
echo "DEVICE=$DEVICE  SCALE=$SCALE  SHARDING=$SHARDING_SIZE"
echo "HOSTS=$HOSTS  NNODES=$NNODES"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "LOG_FILE=$LOG_FILE"
echo "============================================="

if [ "$DEVICE" = "dcu" ]; then
    # srun 模式 (参考 run_llama3_100b.sh): 每节点 1 任务, RANK=SLURM_PROCID
    export NNODES MASTER_ADDR
    export NCCL_SOCKET_IFNAME=ib0
    export NCCL_IB_HCA=shca_0:1,shca_1:1,shca_2:1,shca_3:1
    export NCCL_IB_DISABLE=0
    export NCCL_NET_PLUGIN=shca
    export HSA_FORCE_FINE_GRAIN_PCIE=1
    export PYTHONUNBUFFERED=1
    export FLAGS_deterministic_rng=1
    export CUDA_VISIBLE_DEVICES="$GPUS"
    srun -N "$NNODES" \
        bash -lc "export RANK=\${SLURM_PROCID} && export PADDLEFORMERS_DIST_LOG=$DIST_LOG_DIR/node_\${RANK} && module load compiler/dtk/25.04.4 && source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && paddleformers-cli train '$CONFIG_FILE'" \
        >"$LOG_FILE" 2>&1
else
    # GPU: 使用 orterun -H
    /usr/local/openmpi-4.1.5/bin/orterun --allow-run-as-root -bind-to none -tag-output -H "${HOSTS//,/:1,}:1" \
        -x NNODES="$NNODES" \
        -x MASTER_ADDR="$MASTER_ADDR" \
        -x MASTER_PORT=38888 \
        -x PYTHONUNBUFFERED=1 \
        -x FLAGS_deterministic_rng=1 \
        -x CUDA_VISIBLE_DEVICES="$GPUS" \
        -x PYTHONPATH \
        bash -c "export RANK=\${OMPI_COMM_WORLD_RANK} && export PADDLEFORMERS_DIST_LOG=$DIST_LOG_DIR/node_\${RANK} && source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && paddleformers-cli train '$CONFIG_FILE'" \
        >"$LOG_FILE" 2>&1
fi
