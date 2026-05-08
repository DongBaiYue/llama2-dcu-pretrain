# 曙光2030项目
## 指标
指标1.5 国产深度学习框架具备千卡国产芯片的大模型训练能力；可以对十亿到千亿级参数规模的语言大模型预训练及微调
指标1.6 支持千卡集群的分布式资源弹性，支持动态扩容方式提升大模型训练效率，具备在训练千亿大模型过程中将训练资源从百卡扩容到千卡的能力

## 指标1.6 实现方案：只扩DP，停机重启

### 原理

只扩DP不需要reshard：模型权重按TP/PP分片（与DP无关），优化器按sharding rank分片（仅dp_rank==0加载后broadcast）。`check_same_strategy()`不比较dp_degree，checkpoint直接可用。

### 操作步骤

```bash
# 阶段1: 4卡训练 (TP=2, PP=2, DP=1)
bash scripts/pt/run_elastic.sh 4card dcu

# 等待checkpoint保存后停止训练，修改YAML:
CKPT=$(ls -d checkpoints/pt/13b_elastic/checkpoint-* | sort -t- -k2 -n | tail -1)
sed -i "s|sharding_parallel_size: 1|sharding_parallel_size: 2|" configs/pt/train_elastic.yaml
sed -i "s|max_steps: 20|max_steps: 40|" configs/pt/train_elastic.yaml
# resume_from_checkpoint 由8card脚本自动设置

# 阶段2: 扩容到8卡 (TP=2, PP=2, DP=2)，自动从checkpoint恢复训练
bash scripts/pt/run_elastic.sh 8card dcu
```

### YAML配置说明

配置文件: `configs/pt/train_elastic.yaml`

| 配置项 | 阶段1 (4卡) | 阶段2 (8卡) | 说明 |
|--------|------------|------------|------|
| `max_steps` | 20 | 40 | 阶段1训练20步，阶段2从step 21继续到step 40 |
| `sharding_parallel_size` | 1 | 2 | 8卡时必须为2，否则OOM |
| `resume_from_checkpoint` | 无 | 自动设置 | 8card脚本自动检测并填入最新checkpoint路径 |

不要设置 `ignore_load_lr_and_optim: true`，否则优化器状态和学习率不会恢复。

### 验收标准

- loss曲线无缝衔接（扩容前后无跳变）
- 吞吐近线性提升（4卡→8卡吞吐≈2x）
- 扩容操作简单（只改卡数，不改YAML中的TP/PP/sharding/lr）
