# 千亿参数模型深度扩张指南

## 1. 背景

以 Llama-3-70B 为基础，通过深度扩张方式构建千亿级参数模型（≈104B），验证国产 DCU 的大模型训练能力。

与宽度扩张不同，深度扩张仅增加 Transformer 层数，保持 hidden_size 和注意力头数不变，使得并行策略（TP/PP）可直接复用 70B 的方案。

---

## 2. Transformer 参数量计算

### 2.1 总参数量公式

```
总参数量 ≈ vocab_size × hidden_size           # 词嵌入
         + num_layers × hidden_size² × 12    # Transformer 层
         + num_layers × hidden_size × 3      # LayerNorm
```

简化估算（忽略 LayerNorm 和 bias）：

```
总参数量 ≈ num_layers × hidden_size² × 12
```

### 2.2 各组件参数量

| 组件 | 参数量 | 占比 |
|------|--------|------|
| Embedding | vocab_size × hidden_size | ~2% |
| Attention (Q/K/V/O) | 4 × hidden_size² | ~33% |
| MLP (gate/up/down) | 3 × hidden_size × intermediate_size | ~55% |
| LayerNorm | 2 × num_layers × hidden_size | <1% |

---

## 3. Llama-3-70B 基准配置

```yaml
hidden_size: 8192
intermediate_size: 28672        # hidden_size × 3.5
num_hidden_layers: 80
num_attention_heads: 64
num_key_value_heads: 8          # GQA
vocab_size: 128256

# 计算验证
# Attention: 4 × 8192² = 268,435,456
# MLP: 3 × 8192 × 28672 = 704,643,072
# Per Layer: ~973M
# Total: 80 × 973M + 128256 × 8192 ≈ 70B
```

---

## 4. 深度扩张策略

### 4.1 为什么选择深度扩张

| 方案 | 改动 | 对并行策略的影响 |
|------|------|-----------------|
| **深度扩张** (推荐) | 只增 num_hidden_layers | TP/PP 不变，直接复用 70B 配置 |
| 宽度扩张 | 增 hidden_size/heads | TP 需重新划分，可能无法整除 |
| 均衡扩张 | 深度+宽度同时增 | TP/PP 都需调整，复杂度高 |

### 4.2 100B 深度扩张参数

将 `num_hidden_layers` 从 80 扩到 120，其余结构参数不变：

```
P_new / P_70B = L_new / L_70B = 120 / 80 = 1.5
P_new ≈ 70B × 1.5 ≈ 104B (千亿级)
```

### 4.3 70B vs 100B 对比

| 参数 | 70B | 100B | 变化 |
|------|-----|------|------|
| hidden_size | 8192 | 8192 | 不变 |
| intermediate_size | 28672 | 28672 | 不变 |
| num_hidden_layers | 80 | 120 | 1.5× |
| num_attention_heads | 64 | 64 | 不变 |
| num_key_value_heads | 8 | 8 | 不变 |
| 参数量 | ~70B | ~104B | 1.5× |

---

## 5. 并行策略

### 5.1 显存估算

| 项目 | 70B | 100B | 说明 |
|------|-----|------|------|
| 模型权重 (bf16) | 140 GB | 208 GB | 参数量 × 2 bytes |
| 优化器状态 (Adam) | 560 GB | 832 GB | 参数量 × 8 bytes |
| 梯度 | 140 GB | 208 GB | 参数量 × 2 bytes |

### 5.2 4 节点 32 卡并行配置

TP=8 PP=4 与 70B 完全相同，每 PP stage 30 层（120/4）。

```yaml
tensor_model_parallel_size: 8
pipeline_model_parallel_size: 4
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
```

### 5.3 单卡显存估算

- 模型权重: 208GB / 32 = 6.5GB
- 优化器: 832GB / 32 = 26GB
- 梯度: 208GB / 32 = 6.5GB
- 激活值 + 临时: ~3-5GB
- **合计**: ~42-45GB (64GB BW1000 可承载)

---

## 6. 实现步骤

### 6.1 生成模型配置

使用脚本自动生成 100B 模型目录：

```bash
bash scripts/pt/prepare_100b_model.sh
```

脚本会基于 `models/Llama-3-70b/` 复制 tokenizer 文件，并将 `config.json` 中的 `num_hidden_layers` 改为 120。

### 6.2 训练配置

**文件**: `configs/pt/train_llama3_100b.yaml`

```yaml
model_name_or_path: ./models/Llama-3-100b
tensor_model_parallel_size: 8
pipeline_model_parallel_size: 4
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 1
bf16: true
```

### 6.3 启动训练

```bash
bash scripts/pt/run_llama3_100b.sh dcu    # 4 节点 32 卡
```

---

## 7. 验证检查点

| 阶段 | 检查项 | 预期结果 |
|------|--------|----------|
| 模型初始化 | 参数量统计 | ~104B 参数 |
| 前向传播 | 输出形状 | [batch, seq, 128256] |
| 反向传播 | 梯度正常 | 无 NaN/Inf |
| Loss | 初始值 | ~11 (随机初始化) |
| 训练 | Loss 下降 | 逐步收敛 |

---

## 8. 参考资源

- [Llama 3 Technical Report](https://ai.meta.com/blog/meta-llama-3/)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- PaddleFormers Llama 实现: `paddleformers/transformers/llama/`
