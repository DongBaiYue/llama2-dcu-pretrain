#!/bin/bash
set -euo pipefail

# 用法: run_llama3_8b_cold.sh [node5] [node2] [node3] ...
# 指定节点名（来自 nodes.conf）则多机冷启动；无参数则只跑本机单机 8 卡
# 冷启动会移除 resume_from_checkpoint，固定 max_steps=20，并使用 flex_checkpoint 保存
# 本容器=rank0, 远程容器通过 relay + docker exec -d 启动

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="/workspace/llama2-dcu-pretrain"
CONF="$SCRIPT_DIR/nodes.conf"
RELAY_HOST="${RELAY_HOST:-10.213.179.22:6161}"
MASTER_PORT=38888
CONFIG_FILE="configs/pt/train_llama3_8b_mini.yaml"

# 读 nodes.conf 到关联数组
declare -A NODE_SSH NODE_CTR
while IFS= read -r line; do
    [[ "$line" =~ ^# || -z "$line" ]] && continue
    name=$(echo "$line" | awk '{print $1}')
    NODE_SSH[$name]=$(echo "$line" | awk '{print $2}')
    NODE_CTR[$name]=$(echo "$line" | awk '{print $3}')
done < <(grep -v '^\s*#' "$CONF" | grep -v '^\s*$')

# 解析参数：指定要用的节点名
SELECTED=()
for arg in "$@"; do
    [[ -v NODE_SSH[$arg] ]] || { echo "ERROR: 节点 '$arg' 不在 $CONF 中" >&2; exit 1; }
    SELECTED+=("$arg")
done

NNODES=$(( ${#SELECTED[@]} + 1 ))
MASTER_ADDR="$(hostname -I | awk '{print $1}')"
SCALE="${SELECTED[*]:-local}"
SCALE="${SCALE// /_}"

cd "$PROJECT_ROOT"

# 冷启动：移除 checkpoint 恢复配置，固定短跑 20 step，避免 sharding_io 最终保存触发 TP merge 不支持路径
# 4 层 mini 配置中，单机 8 卡 accumulation=4；两机 16 卡为了对齐 global batch 使用 accumulation=2。
GRAD_ACC=$([[ "$NNODES" -eq 1 ]] && echo 4 || echo 2)
sed -i '/^resume_from_checkpoint:/d' "$CONFIG_FILE"
sed -i 's/^max_steps:.*/max_steps: 20/' "$CONFIG_FILE"
sed -i "s/^gradient_accumulation_steps:.*/gradient_accumulation_steps: $GRAD_ACC/" "$CONFIG_FILE"
if grep -q "^save_checkpoint_format:" "$CONFIG_FILE"; then
    sed -i 's|^save_checkpoint_format:.*|save_checkpoint_format: flex_checkpoint|' "$CONFIG_FILE"
else
    sed -i '/^# checkpoint$/a save_checkpoint_format: flex_checkpoint' "$CONFIG_FILE"
fi
echo "已设置冷启动配置: 删除 resume_from_checkpoint, max_steps=20, gradient_accumulation_steps=$GRAD_ACC, save_checkpoint_format=flex_checkpoint"

echo "=== Llama3-8B 冷启动节点: 本机 ${SELECTED[*]} (NNODES=$NNODES, MASTER=$MASTER_ADDR) ==="

relay_run() {
    local target="$1" cmd="$2" encoded
    encoded=$(echo -n "$cmd" | base64 -w0)
    curl -s -X POST "http://$RELAY_HOST/run" \
        --data-urlencode "target=$target" \
        --data-urlencode "cmd=$encoded" \
        --data-urlencode "b64=1"
}

# 1) 停旧进程
pkill -f paddleformers 2>/dev/null || true
for n in "${SELECTED[@]}"; do
    relay_run "${NODE_SSH[$n]}" "docker exec ${NODE_CTR[$n]} pkill -f paddleformers" 2>/dev/null || true
done
sleep 3

# 2) 同步冷启动 yaml 到远程节点
for n in "${SELECTED[@]}"; do
    content=$(base64 -w0 "$CONFIG_FILE")
    docs/k100/dnode.sh exec "$n" "cd $PROJECT_ROOT && echo $content | base64 -d > $CONFIG_FILE" >/dev/null
done

# 3) 启动本容器 (rank=0)
LOG0="logs/pt/llama3-8b_cold_${SCALE}_dcu_r0.log"
source /workspace/py310/bin/activate
RANK=0 NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 \
    nohup paddleformers-cli train "$CONFIG_FILE" > "$LOG0" 2>&1 &
echo "本容器 rank=0 PID=$! LOG=$LOG0"

# 4) 启动远程容器
rank=1
for n in "${SELECTED[@]}"; do
    ssh="${NODE_SSH[$n]}"
    ctr="${NODE_CTR[$n]}"

    cmd="cd $PROJECT_ROOT && source /workspace/py310/bin/activate && \
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
export RANK=$rank NNODES=$NNODES MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTHONUNBUFFERED=1 && \
nohup paddleformers-cli train $CONFIG_FILE > logs/pt/llama3-8b_cold_${SCALE}_dcu_r${rank}.log 2>&1"

    encoded=$(echo -n "$cmd" | base64 -w0)
    relay_run "$ssh" "docker exec -d $ctr bash -c 'echo $encoded | base64 -d | bash'" >/dev/null
    echo "$n rank=$rank"
    rank=$(( rank + 1 ))
done

echo "=== 已启动 ==="
echo "本机: tail -f $LOG0"
rank=1
for n in "${SELECTED[@]}"; do
    echo "$n: docker exec ${NODE_CTR[$n]} tail -f $PROJECT_ROOT/logs/pt/llama3-8b_cold_${SCALE}_dcu_r${rank}.log"
    rank=$(( rank + 1 ))
done
