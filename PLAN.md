# Llama-2-13B DCU 训练

曙光项目：在国产 DCU 上验证大模型训练能力。

**当前状态**: ✅ PT 预训练已跑通, ✅ SFT 全量微调已跑通, ✅ LoRA 已跑通, ✅ 推理验证完成

**更新日期**: 2026-04-10

---

## 目录结构

```
llama3_dcu_pretrain/
├── configs/
│   ├── pt/                     # 预训练配置
│   │   ├── train.yaml          # 13B 基准配置
│   │   └── train_26b.yaml      # 26B 2倍扩张配置
│   ├── sft/                    # SFT 配置
│   │   └── train.yaml
│   └── lora/                   # LoRA 配置
│       └── train.yaml
├── scripts/
│   ├── pt/
│   │   ├── run.sh              # 13B 基准脚本
│   │   └── run_26b.sh          # 26B 扩张脚本
│   ├── sft/
│   │   └── run.sh
│   └── lora/
│       └── run.sh
├── data/
│   ├── pt/                     # 预训练数据 (offline mmap)
│   │   ├── llama_openwebtext_100k.bin
│   │   └── llama_openwebtext_100k.idx
│   └── sft/                    # SFT 数据 (erniekit jsonl)
│       ├── train.jsonl
│       └── eval.jsonl
├── models/
│   ├── Llama-2-13b/            # 13B 基准模型
│   └── Llama-2-26B/            # 26B 扩张模型
├── checkpoints/
│   ├── pt/
│   ├── sft/
│   └── lora/
├── logs/
│   ├── pt/
│   ├── sft/
│   └── lora/
└── docs/
    └── scaling_to_110b.md      # 千亿模型扩张指南
```

---

## 训练任务

### 1. PT 预训练 (已完成 ✅)

```bash
bash scripts/pt/run.sh
```

**配置要点**:
- `continue_training: false` - 从头预训练
- `learning_rate: 1e-4` - 预训练学习率
- `dataset_type: pretrain` - 离线 mmap 数据集
- `tensor_model_parallel_size: 2` - 张量并行度为2
- `pipeline_model_parallel_size: 2` - 流水线并行度为2
- `sharding: stage1` - 使用Sharding Stage1优化显存

**验证结果** (2026-04-09):
- Loss: 8.496 (100 steps)
- Runtime: 14分17秒
- 数据集解析正确
- 检查点已保存至 `./checkpoints/pt/llama2-13b-pretrain/`

---

### 2. SFT 全量微调 (已完成 ✅)

```bash
bash scripts/sft/run.sh
```

**配置要点**:
- `stage: SFT`
- `fine_tuning: full` - 全量参数微调
- `learning_rate: 1e-5` - SFT 学习率较低
- `train_dataset_type: erniekit` - 使用erniekit数据格式
- `tensor_model_parallel_size: 2` - 张量并行度为2
- `pipeline_model_parallel_size: 2` - 流水线并行度为2
- `sharding: stage1` - 使用Sharding Stage1优化显存

**验证结果** (2026-04-10):
- Loss: 0.5989 (100 steps)
- Runtime: 15分11秒
- 数据集: school_math_0.25M (erniekit格式)
- 检查点已保存至 `./checkpoints/sft/llama2-13b-sft/`

---

### 3. LoRA 微调 (已完成 ✅)

```bash
bash scripts/lora/run.sh
```

**配置要点**:
- `fine_tuning: lora`
- `lora_rank: 8` - LoRA 秩
- `lora_alpha: 16`
- `learning_rate: 1e-4` - LoRA 学习率可更高
- `tensor_model_parallel_size: 2` - 张量并行度为2
- `pipeline_model_parallel_size: 2` - 流水线并行度为2
- `sharding: stage1` - 使用Sharding Stage1优化显存

**LoRA 目标模块**:
```yaml
lora_target_modules:
  - q_proj, v_proj, k_proj, o_proj  # Attention
  - gate_proj, up_proj, down_proj   # MLP
```

**验证结果** (2026-04-10):
- Loss: 0.6507 (100 steps)
- Runtime: 12分20秒
- 参数量: 仅训练 ~1% 参数 (13.59 MB)
- 检查点已保存至 `./checkpoints/lora/llama2-13b-lora/`
- 推理验证: LoRA 导出与推理成功

---

## 配置对比

### 训练类型对比

| 参数 | PT | SFT | LoRA |
|------|-----|-----|------|
| stage | PT | SFT | SFT |
| fine_tuning | full | full | lora |
| continue_training | false | true | true |
| learning_rate | 1e-4 | 1e-5 | 1e-4 |
| dataset_type | pretrain | erniekit | erniekit |
| tensor_model_parallel_size | 2 | 2 | 2 |
| pipeline_model_parallel_size | 2 | 2 | 2 |
| sharding | stage1 | stage1 | stage1 |
| 训练参数量 | 100% | 100% | ~1% |

### 等比扩缩对比

| 模型 | 层数 | hidden_size | 参数量 | 并行配置 | 卡数 |
|------|------|-------------|--------|----------|------|
| Llama-2-13B | 40 | 5120 | ~13B | TP=2, PP=2 | 4 |
| Llama-2-26B | 80 | 5120 | ~26B | TP=4, PP=2 | 8 |
| Llama-2-110B | 80 | 10240 | ~105B | TP=4, PP=4 | 16 |

---

## 启动命令

```bash
# PT 预训练 (13B)
bash scripts/pt/run.sh

# PT 预训练 (26B - 2倍扩张)
bash scripts/pt/run_26b.sh

# SFT 全量微调
bash scripts/sft/run.sh

# LoRA 微调
bash scripts/lora/run.sh
```


---

## 等比扩缩验证 (模型参数量扩张)

扩张公式: `P_new = k_d × k_w² × P_base` (k_d=深度系数, k_w=宽度系数)

### 基准配置 (Baseline - 13B)
- **模型**: Llama-2-13B
- **架构**: 40层, 5120 hidden_size
- **参数量**: ~13B
- **并行策略**: TP=2, PP=2, 4卡
- **配置**: `configs/pt/train.yaml`
- **脚本**: `bash scripts/pt/run.sh`

### 2倍扩张验证 (26B) - 进行中 ⏳
- **模型**: Llama-2-26B
- **架构**: 80层, 5120 hidden_size (深度翻倍, 宽度不变)
- **参数量**: ~26B (13B × 2)
- **并行策略**: TP=4, PP=2, 8卡
- **配置**: `configs/pt/train_26b.yaml`
- **脚本**: `bash scripts/pt/run_26b.sh`

**验证目标**: 验证26B模型训练稳定性，对比13B与26B的收敛特性

---

## 训练指标汇总

| 任务 | Loss | Runtime | 参数量 | 状态 |
|------|------|---------|--------|------|
| PT 预训练 | 8.496 | 14分17秒 | 100% (6.3 GB) | ✅ 完成 |
| SFT 全量微调 | 0.5989 | 15分11秒 | 100% (6.3 GB) | ✅ 完成 |
| LoRA 微调 | 0.6507 | 12分20秒 | ~1% (13.59 MB) | ✅ 完成 |

---

## 已完成

- [x] PT 预训练流程验证
- [x] SFT 训练流程验证
- [x] LoRA 训练流程验证

## 下一步

### 等比扩缩验证
- [ ] 2倍扩张验证 (26B: 80层, 8卡)
- [ ] 4倍扩张验证 (52B: 80层, 7240 hidden 或 160层)
- [ ] 8倍扩张验证 (110B: 详见 docs/scaling_to_110b.md)

### 精度对齐
- [ ] 与 GPU 基线 Loss 对齐（300 steps 收敛曲线对比）
