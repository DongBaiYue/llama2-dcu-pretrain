# 千亿参数模型等比扩张指南

## 1. 背景

以 Llama-2-13B 为基础，通过等比扩张方式构建千亿级参数模型，验证国产 DCU 的大模型训练能力。

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

## 3. Llama-2-13B 基准配置

```yaml
hidden_size: 5120
intermediate_size: 13824        # hidden_size × 2.7
num_hidden_layers: 40
num_attention_heads: 40
num_key_value_heads: 40         # 不使用 GQA
vocab_size: 32000

# 计算验证
# Attention: 4 × 5120² = 104,857,600
# MLP: 3 × 5120 × 13824 = 212,336,640
# Per Layer: ~317M
# Total: 40 × 317M + 32000 × 5120 ≈ 13B
```

---

## 4. 扩张策略

### 4.1 扩张维度

| 维度 | 符号 | 影响 | 扩展系数 |
|------|------|------|----------|
| 深度 (层数) | L | 线性增长 | k_d |
| 宽度 (hidden_size) | H | 平方增长 | k_w |
| 注意力头数 | A | 通常与 H 线性相关 | k_w |

### 4.2 参数量增长公式

```
P_new = L_new × H_new² × 12
      = (k_d × L) × (k_w × H)² × 12
      = k_d × k_w² × P_base
```

**目标**: P_new ≈ 110B (千亿)

```
k_d × k_w² = 110B / 13B ≈ 8.5
```

### 4.3 扩张方案对比

| 方案 | k_d | k_w | 参数量 | 特点 |
|------|-----|-----|--------|------|
| 深度优先 | 8.5 | 1.0 | 110B | 层数过多，训练慢 |
| 宽度优先 | 1.0 | 2.9 | 110B | 显存占用大 |
| 均衡扩张 | 2.0 | 2.0 | 104B | ✅ 推荐 |
| 深度略大 | 2.5 | 1.8 | 105B | 训练并行度更好 |

---

## 5. 推荐配置：110B 模型

### 5.1 模型配置

```json
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 10240,
  "intermediate_size": 27648,
  "num_hidden_layers": 80,
  "num_attention_heads": 80,
  "num_key_value_heads": 80,
  "vocab_size": 32000,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-05,
  "hidden_act": "silu",
  "tie_word_embeddings": false
}
```

### 5.2 参数量验证

```
# Embedding
vocab_size × hidden_size = 32000 × 10240 = 327,680,000 (0.3B)

# Per Layer
Attention: 4 × 10240² = 419,430,400
MLP: 3 × 10240 × 27648 = 849,346,560
Per Layer Total: ~1.27B

# Total
80 layers × 1.27B + 0.3B ≈ 102B

# 加上 output projection
Total ≈ 105B (千亿级)
```

### 5.3 与 13B 对比

| 参数 | 13B | 110B | 扩展倍数 |
|------|-----|------|----------|
| hidden_size | 5120 | 10240 | 2.0× |
| intermediate_size | 13824 | 27648 | 2.0× |
| num_layers | 40 | 80 | 2.0× |
| num_heads | 40 | 80 | 2.0× |
| 参数量 | 13B | 105B | 8× |

---

## 6. 并行策略调整

### 6.1 显存估算

| 项目 | 13B | 110B | 说明 |
|------|-----|------|------|
| 模型权重 (bf16) | 26 GB | 210 GB | 参数量 × 2 bytes |
| 优化器状态 (Adam) | 104 GB | 840 GB | 参数量 × 8 bytes |
| 梯度 | 26 GB | 210 GB | 参数量 × 2 bytes |
| 激活值 | ~10 GB | ~80 GB | 与 batch_size、seq_len 相关 |

### 6.2 推荐并行配置

**4卡 (当前环境)**:
```yaml
# 110B 显存不足，需要更多卡
tensor_model_parallel_size: 4    # TP=4
pipeline_model_parallel_size: 4  # PP=4
# 总共需要 16 卡
```

**8卡**:
```yaml
tensor_model_parallel_size: 4
pipeline_model_parallel_size: 4
# 使用 sharding 进一步优化
sharding: stage1
sharding_parallel_size: 2
```

**16卡 (推荐)**:
```yaml
tensor_model_parallel_size: 4
pipeline_model_parallel_size: 4
sharding: stage1
sharding_parallel_size: 1
```

### 6.3 重计算策略

```yaml
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 20   # 110B 需要更多重计算
```

---

## 7. 实现步骤

### 7.1 创建模型配置文件

```bash
# 创建 110B 模型目录
mkdir -p models/Llama-2-110B

# 创建 config.json
cat > models/Llama-2-110B/config.json << 'EOF'
{
  "architectures": ["LlamaForCausalLM"],
  "hidden_size": 10240,
  "intermediate_size": 27648,
  "num_hidden_layers": 80,
  "num_attention_heads": 80,
  "num_key_value_heads": 80,
  "vocab_size": 32000,
  "max_position_embeddings": 4096,
  "rms_norm_eps": 1e-05,
  "hidden_act": "silu",
  "tie_word_embeddings": false,
  "bos_token_id": 1,
  "eos_token_id": 2
}
EOF
```

### 7.2 复制 Tokenizer

```bash
# 从 13B 复制 tokenizer 文件
cp models/Llama-2-13b/tokenizer.model models/Llama-2-110B/
cp models/Llama-2-13b/tokenizer.json models/Llama-2-110B/
cp models/Llama-2-13b/tokenizer_config.json models/Llama-2-110B/
cp models/Llama-2-13b/special_tokens_map.json models/Llama-2-110B/
```

### 7.3 创建训练配置

**文件**: `configs/train_110b.yaml`

```yaml
### data
dataset_type: pretrain
input_dir: "1.0 ./data/pt/llama_openwebtext_100k"
split: "990,10,0"
max_seq_len: 2048          # 110B 减少 seq_len 降低显存

### model
model_name_or_path: ./models/Llama-2-110B
_attn_implementation: eager
continue_training: false

### finetuning
stage: PT
fine_tuning: full
seed: 23
do_train: true
do_eval: false
per_device_train_batch_size: 1
per_device_eval_batch_size: 1
max_steps: 100
gradient_accumulation_steps: 8   # 增大累积步数
logging_dir: ./logs
output_dir: ./checkpoints/llama2-110b-pretrain

# train
warmup_steps: 10
learning_rate: 1.0e-4

# performance (需要调整)
tensor_model_parallel_size: 4
pipeline_model_parallel_size: 4
sharding: stage1
recompute_granularity: full
recompute_method: uniform
recompute_num_layers: 20
bf16: true

# device
device: gpu
```

### 7.4 创建启动脚本

**文件**: `scripts/run_110b.sh`

```bash
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source /public/home/baidu_test/hygon_2030/py310/bin/activate

# 16卡配置
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

NNODES=1 MASTER_ADDR=127.0.0.1 MASTER_PORT=8080 \n  paddleformers-cli train configs/train_110b.yaml 2>&1 | tee logs/train_110b.log
```

---

## 8. 训练参数调优

### 8.1 学习率

| 模型规模 | 推荐学习率 | 说明 |
|----------|------------|------|
| 13B | 1e-4 | 基准 |
| 70B | 5e-5 | 略微降低 |
| 110B | 3e-5 | 进一步降低 |

### 8.2 Batch Size

```
全局 batch_size = per_device_batch_size × gradient_accumulation × data_parallel_size
```

| 模型 | 全局 batch | 推荐配置 |
|------|------------|----------|
| 13B | 16 | 1 × 4 × 4 |
| 110B | 32-64 | 1 × 8 × 4 |

### 8.3 序列长度

| 模型 | 推荐 max_seq_len | 原因 |
|------|------------------|------|
| 13B | 4096 | 基准 |
| 110B | 2048 | 降低显存压力 |

---

## 9. 验证检查点

| 阶段 | 检查项 | 预期结果 |
|------|--------|----------|
| 模型初始化 | 参数量统计 | ~105B 参数 |
| 前向传播 | 输出形状 | [batch, seq, vocab_size] |
| 反向传播 | 梯度正常 | 无 NaN/Inf |
| Loss | 初始值 | ~11 (随机初始化) |
| 训练 | Loss 下降 | 逐步收敛 |

---

## 10. 参考资源

- [Llama 2 Technical Report](https://arxiv.org/abs/2307.09288)
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)
- PaddleFormers Llama 实现: `paddleformers/transformers/llama/`
