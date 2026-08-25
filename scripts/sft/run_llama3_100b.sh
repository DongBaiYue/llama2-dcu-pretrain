#!/bin/bash
set -euo pipefail

module load compiler/dtk/25.04.4
module load mpi/hpcx/2.18.0/gcc-8.5.0/shca

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# 用法: run_llama3_100b.sh [NNODES] [dcu|gpu]
#   NNODES: 可选, 节点数, 默认 8; 兼容 "16" 或 "16node" 两种格式; 环境变量 NNODES 可覆盖 (命令行优先)
# 并行策略: TP=8, PP=8 固定 (至少 64 卡 = 8 节点), 更多节点扩剩余维度 (sharding/DP 由框架分配):
#   8   节点 -> TP=8 PP=8    (64 卡)
#   16  节点 -> TP=8 PP=8    (128 卡)
#   128 节点 -> TP=8 PP=8    (1024 卡)
# 约束: NNODES >= 8 且为 8 的倍数

# 参数解析: $1=NNODES(默认8, 兼容 "16"/"16node"), $2=设备(默认auto)
NNODES="${1:-${NNODES:-8}}"
NNODES="${NNODES%node}"   # 去掉 "node" 后缀, 兼容 "128node" 格式
DEVICE="${2:-auto}"

# 校验节点数: >=8 且为 8 的倍数 (TP=8, PP=8 至少 64 卡)
if ! [[ "$NNODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: 无效的节点数 '$NNODES', 请使用正整数, 如: run_llama3_100b.sh 16node dcu" >&2
    exit 1
fi
if [ "$NNODES" -lt 8 ]; then
    echo "ERROR: NNODES=$NNODES 过小; TP=8, PP=8 至少需要 8 节点 (64 卡)" >&2
    exit 1
fi
if [ $((NNODES % 8)) -ne 0 ]; then
    echo "ERROR: NNODES=$NNODES 不是 8 的倍数; TP=8, PP=8 下节点数须为 8 的倍数" >&2
    exit 1
fi

if [ "$DEVICE" = "auto" ]; then
    if command -v nvidia-smi &>/dev/null; then
        DEVICE="gpu"
    elif command -v rocm-smi &>/dev/null; then
        DEVICE="dcu"
    else
        echo "ERROR: 无法自动检测设备类型，请手动指定: run_llama3_100b.sh [NNODES] [dcu|gpu]" >&2
        exit 1
    fi
fi

case "$DEVICE" in
    dcu)
        CONFIG_FILE="configs/sft/train_llama3_100b.yaml"
        LOG_FILE="logs/sft/llama3-100b_${NNODES}node.log"
        DIST_LOG_DIR="logs/sft/llama3-100b_${NNODES}node.dist"
        VENV_DIR="$PROJECT_ROOT/../py310"
        ;;
    gpu)
        CONFIG_FILE="configs/sft/train_llama3_100b.yaml"
        LOG_FILE="logs/sft/llama3-100b_${NNODES}node_gpu.log"
        DIST_LOG_DIR="logs/sft/llama3-100b_${NNODES}node_gpu.dist"
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
    srun pkill -9 -f paddleformers.cli.launcher.*llama3_100b 2>/dev/null || true
else
    mpirun pkill -9 -f paddleformers.cli.launcher.*llama3_100b 2>/dev/null || true
fi
sleep 5

# 激活虚拟环境
source "$VENV_DIR/bin/activate"
command -v paddleformers-cli >/dev/null 2>&1 || { echo "paddleformers-cli not found" >&2; exit 1; }

GPUS="0,1,2,3,4,5,6,7"
TP_SIZE=8
PP_SIZE=8
DP_SIZE=$((NNODES / 8))
TOTAL_GPUS=$((NNODES * 8))

# 分布式配置: 按 NNODES 解析节点列表
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

export PYTHONUNBUFFERED=1
mkdir -p "$(dirname "$LOG_FILE")" "$DIST_LOG_DIR"
export PADDLEFORMERS_DIST_LOG="$DIST_LOG_DIR"

echo "========== Llama-3-100B SFT =========="
echo "DEVICE=$DEVICE   NNODES=$NNODES"
echo "TP=$TP_SIZE  PP=$PP_SIZE  DP=$DP_SIZE  (共 $TOTAL_GPUS 卡)"
echo "HOSTS=$HOSTS"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "CONFIG_FILE=$CONFIG_FILE"
echo "LOG_FILE=$LOG_FILE"
echo "======================================="

if [ "$DEVICE" = "dcu" ]; then
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
        bash -lc "export RANK=\${SLURM_PROCID} && export PADDLEFORMERS_DIST_LOG=$DIST_LOG_DIR/node_\${RANK} && module load compiler/dtk/25.04.4 && source '$VENV_DIR/bin/activate' && cd '$PROJECT_ROOT' && paddleformers-cli train '$CONFIG_FILE' tensorwise_offload_optimizer=false" \
        >"$LOG_FILE" 2>&1
else
    mpirun \
        -H "$HOSTS" \
        -n "$NNODES" \
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
