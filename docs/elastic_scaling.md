# 曙光2030项目
## 指标
指标1.5 国产深度学习框架具备千卡国产芯片的大模型训练能力；可以对十亿到千亿级参数规模的语言大模型预训练及微调
指标1.6 支持千卡集群的分布式资源弹性，支持动态扩容方式提升大模型训练效率，具备在训练千亿大模型过程中将训练资源从百卡扩容到千卡的能力

## 指标1.6 实现方案：只扩DP，停机重启

### 原理

只扩DP不需要reshard：模型权重按TP/PP分片（与DP无关），优化器按sharding rank分片（仅dp_rank==0加载后broadcast）。`check_same_strategy()`不比较dp_degree，checkpoint直接可用。

缩容同理：减少DP rank，checkpoint同样直接可用。

### 操作步骤

```bash
# 扩容: 4卡 → 8卡 → 双机16卡
bash scripts/pt/run_elastic.sh 4card dcu    # step 0-30
bash scripts/pt/run_elastic.sh 8card dcu    # step 30-60
bash scripts/pt/run_elastic.sh 2node dcu    # step 60-90

# 缩容: 双机16卡 → 8卡 → 4卡
bash scripts/pt/run_elastic.sh 8card dcu    # step 90-120
bash scripts/pt/run_elastic.sh 4card dcu    # step 120-150
```

脚本自动完成：停掉上一阶段进程、设置 `resume_from_checkpoint`、调整 `sharding_parallel_size`。

### YAML配置说明

配置文件: `configs/pt/train_elastic.yaml`

| 配置项 | 4卡 | 8卡 | 双机16卡 | 说明 |
|--------|-----|-----|---------|------|
| `max_steps` | 150 | 150 | 150 | 各阶段相同，保证LR schedule一致 |
| `save_steps` | 30 | 30 | 30 | 每30步保存checkpoint |
| `sharding_parallel_size` | 1 | 2 | 4 | 脚本自动修改 |
| `resume_from_checkpoint` | 自动 | 自动 | 自动 | 脚本自动检测最新checkpoint |

不要设置 `ignore_load_lr_and_optim: true`，否则优化器状态和学习率不会恢复。

### 验收标准

- loss曲线无缝衔接（扩容/缩容前后无跳变）
- 吞吐随卡数近线性变化（4卡→8卡≈2x，8卡→16卡≈2x）
- 扩缩容操作简单（只改卡数，不改YAML中的TP/PP/lr）
