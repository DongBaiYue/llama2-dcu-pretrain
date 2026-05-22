# K100 DCU 精度对齐操作手册

与 GPU agent 配合，验证 DCU 两机训练精度。两边使用相同的 yaml 配置，可通信交换日志和细节。

## 环境激活

```bash
cd /workspace/llama2-dcu-pretrain
source /workspace/py310/bin/activate
```

## 远程节点操作（dnode.sh）

```bash
# 在 node5 容器内执行命令
docs/k100/dnode.sh exec node5 "rocm-smi --showmemuse"

# 在 node5 物理机执行命令
docs/k100/dnode.sh ssh node5 "docker ps"

# 广播到所有节点
docs/k100/dnode.sh bcast "hostname"
docs/k100/dnode.sh bcast-container "rocm-smi"
```

注意：容器内有 HTTP 代理（`agent.baidu.com:8188`），内网传输需先 `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY`。

## 当前推荐脚本

单机冷启动：

```bash
bash docs/k100/run_llama3_8b_cold.sh
```

两机冷启动：

```bash
bash docs/k100/run_llama3_8b_cold.sh node5
```

本机=rank 0，node5=rank 1。日志：

| 场景 | 日志 |
|------|------|
| 单机冷启动 | `logs/pt/llama3-8b_cold_local_dcu_r0.log` |
| 两机冷启动 | `logs/pt/llama3-8b_cold_node5_dcu_r0.log` |
| node5 rank1 | `/workspace/llama2-dcu-pretrain/logs/pt/llama3-8b_cold_node5_dcu_r1.log` |

`run_llama3_8b_cold.sh` 会自动：

- 删除 YAML 中的 `resume_from_checkpoint`
- 设置 `max_steps: 20`
- 设置 `gradient_accumulation_steps`：单机 4，两机 2
- 设置 `save_checkpoint_format: flex_checkpoint`
- 将修改后的 YAML 同步到远程 node5

## 跑两机前清理 checkpoint

两机训练前建议删除本地和 node5 的旧 checkpoint，避免空间不足或读到旧输出：

```bash
rm -rf /workspace/llama2-dcu-pretrain/checkpoints/pt/llama3-8b_mini
docs/k100/dnode.sh exec node5 'rm -rf /workspace/llama2-dcu-pretrain/checkpoints/pt/llama3-8b_mini'
```

清理后检查空间：

```bash
df -h /workspace/llama2-dcu-pretrain
docs/k100/dnode.sh exec node5 'df -h /workspace/llama2-dcu-pretrain'
```

## 两机 YAML 同步检查

两机启动后必须确认本地和 node5 的 YAML 关键字段一致：

```bash
python - <<'PY'
from pathlib import Path
p = Path('/workspace/llama2-dcu-pretrain/configs/pt/train_llama3_8b_mini.yaml')
for line in p.read_text().splitlines():
    if line.startswith(('num_hidden_layers:', 'max_steps:', 'gradient_accumulation_steps:', 'resume_from_checkpoint:', 'save_checkpoint_format:', 'sharding_parallel_size:')):
        print(line)
PY

docs/k100/dnode.sh exec node5 'python - <<"PY"
from pathlib import Path
p = Path("/workspace/llama2-dcu-pretrain/configs/pt/train_llama3_8b_mini.yaml")
for line in p.read_text().splitlines():
    if line.startswith(("num_hidden_layers:", "max_steps:", "gradient_accumulation_steps:", "resume_from_checkpoint:", "save_checkpoint_format:", "sharding_parallel_size:")):
        print(line)
PY'
```

历史问题：远程同步 YAML 如果不先 `cd /workspace/llama2-dcu-pretrain`，可能写到容器默认目录，导致 node5 仍使用旧配置。当前 `run_llama3_8b_cold.sh` 已修复该问题。

## DCU 特有配置

| 配置项 | DCU | GPU | 说明 |
|--------|-----|-----|------|
| `_attn_implementation` | eager | eager | 两边一致，均用 eager |
| 通信库 | RCCL | NCCL | `ncclCommInitRankConfigMemOpt is not supported` 为正常警告 |
| `save_checkpoint_format` | flex_checkpoint | 视场景 | DCU Llama3 冷启动短跑推荐 flex_checkpoint |

其他配置（bf16、lr、parallel strategy 等）与 GPU 完全一致，使用同一个 yaml。

`sharding_io` 在 Llama3 最终 `trainer.save_model(merge_tensor_parallel=True)` 时可能触发 TP merge 的 `NotImplementedError`；冷启动短跑用 `flex_checkpoint` 可以绕过该保存路径问题。

## PaddleFormers 分支要求

Hygon/DCU 训练使用 `/workspace/PaddleFormers`，要求在 `hygon_2030` 分支：

```bash
cd /workspace/PaddleFormers
git branch --show-current
git remote -v | grep DongBaiYue
```

该分支来自 `https://github.com/DongBaiYue/PaddleFormers/tree/hygon_2030`，包含 Hygon 环境问题修复。不要默认上游 PaddleFormers 行为与该分支一致。

## 4 层快速验证

可在 YAML 顶层添加：

```yaml
num_hidden_layers: 4
```

实测该字段会进入最终 `LlamaConfig`，可用于快速验证单机/两机训练链路、loss 对齐和保存逻辑。

为了对齐单机和两机的 global batch，`gradient_accumulation_steps` 按机器数配置：

```yaml
# 单机 8 卡
num_hidden_layers: 4
gradient_accumulation_steps: 4

# 两机 16 卡
num_hidden_layers: 4
gradient_accumulation_steps: 2
```

单机 4 层参考结果：

```text
world_size: 8
Total train batch size = 8
Gradient Accumulation steps = 4
Number of trainable parameters = 1,923,162,112
train_loss ≈ 10.7965
Exit code 0
```

两机 4 层参考结果：

```text
world_size: 16
Total train batch size = 8
Gradient Accumulation steps = 2
Number of trainable parameters = 1,923,162,112
train_loss ≈ 10.7945
Exit code 0
```

4 层单机和两机 loss 基本一致，说明跨节点通信和训练链路正常。

## 精度对比

### 提取 loss

```bash
grep "loss:" logs/pt/llama3-8b_cold_node5_dcu_r0.log | grep "global_step" | tail -20
```

将 DCU 日志发给 GPU agent 对比，或用对比脚本（需 GPU 日志路径）：

```bash
python3 scripts/plot_pt_loss_compare.py
```

### 判定标准

- 稳态 RE < 1% 达标
- warmup 阶段（前 10 步）RE 偏高属正常
- 比较单机和两机 loss 时必须先对齐 global batch；例如 32 层默认两机 `gradient_accumulation_steps: 4` 时 global batch 是 16，不能直接和单机 global batch 8 逐 step 对比
- 两机若需对齐单机 global batch 8，可将 `gradient_accumulation_steps` 改为 2

## 停止训练

```bash
pkill -f paddleformers
docs/k100/dnode.sh exec node5 "pkill -f paddleformers"
```

## 故障排查

| 现象 | 原因 | 解决 |
|------|------|------|
| 训练卡住无 loss | checkpoint/YAML 未同步或跨节点初始化未完成 | 检查两机 YAML、进程和日志；冷启动前清理 checkpoint |
| 容器内下载极慢 | HTTP 流量走代理 | unset http_proxy https_proxy |
| 最终保存 `NotImplementedError` | `sharding_io` 触发 Llama3 TP merge 未实现路径 | 使用 `save_checkpoint_format: flex_checkpoint` |
| 两机 loss 比单机偏大 | global batch 未对齐或短跑抖动 | 对齐 `Total train batch size` 后再比较 |
| RE 持续 >1% | 配置不一致 | 与 GPU agent 确认 yaml 是否相同 |

## 可用节点

node5 可用。node3 DCU 驱动 kill 进程，node4 仅 7 卡，均不可用。

## 关键路径

| 文件 | 用途 |
|------|------|
| `configs/pt/train_llama3_8b_mini.yaml` | 训练配置（与 GPU 共用）；当前可能保留 4 层快速验证配置 |
| `docs/k100/run_llama3_8b_cold.sh` | 当前推荐冷启动脚本，支持无参数单机和节点参数多机 |
| `docs/k100/run_llama3_8b_elastic.sh` | 弹性/续训脚本，不是当前冷启动精度验证首选 |
| `docs/k100/nodes.conf` | K100 节点、SSH target、容器名配置 |
| `docs/k100/dnode.sh` | 远程节点操作 |
| `logs/pt/llama3-8b_cold_local_dcu_r0.log` | 单机冷启动日志 |
| `logs/pt/llama3-8b_cold_node5_dcu_r0.log` | 两机本地 rank0 日志 |
| `/workspace/llama2-dcu-pretrain/logs/pt/llama3-8b_cold_node5_dcu_r1.log` | node5 rank1 日志 |
| `scripts/plot_pt_loss_compare.py` | DCU vs GPU loss 对比 |
