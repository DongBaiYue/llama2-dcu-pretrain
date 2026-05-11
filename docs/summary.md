# 曙光2030项目进展总结

## 指标完成情况

| 指标 | 要求 | 完成状态 |
|:-----|:-----|:--------:|
| **1.5** 大模型训练能力 | 千卡 BW1000 上支持十亿到千亿参数模型的预训练及微调 | ✅ 已验证至 52B，100B 方案就绪 |
| **1.6** 弹性扩缩容 | 百卡→千卡弹性扩容，支持千亿模型训练中动态扩缩 | ✅ 机制已验证（4→8→16 卡），待更大集群验证 |
| **1.5** 100B 千亿参数 | 千亿模型预训练 | 🔲 待更大集群执行 |
| **1.6** 百卡→千卡弹性 | 百卡→千卡扩缩容 | 🔲 待更大集群验证 |

---

## 指标 1.5：大模型训练能力

> 国产深度学习框架具备千卡海光 BW1000 芯片的大模型训练能力，支持十亿到千亿级参数规模的语言大模型预训练及微调

**验证模型选择**：采用 Llama-2 作为验证载体，因其架构公开、GPU 基线可对标，便于量化 DCU 训练精度。所验证的分布式并行、弹性扩缩容、精度对齐等能力与模型架构无关，可迁移至 Qwen、Llama-3 等新架构模型。

### 已完成验证

| 模型 | 参数量 | 训练任务 | 卡数 | 并行策略 | 精度（vs GPU） |
|:----:|:------:|:--------:|:----:|:--------:|:-------------:|
| Llama-2-13B | ~13B | PT / SFT / LoRA | 4 | TP=2, PP=2 | PT 0.93%，SFT 0.51%，LoRA 0.44% |
| Llama-2-26B | ~26B | PT / SFT / LoRA | 8 | TP=4, PP=2 | — |
| Llama-2-52B | ~52B | PT | 16 | TP=8, PP=2, offload | — |

- 精度对齐达标：单机任务与 GPU(A100) 相对误差 <1%（PT 稳态 0.93%）

  ![all_loss_compare_300step_v2](all_loss_compare_300step_v2.png)

### 待完成

- **100B 千亿参数**：方案已设计（hidden_size=10240, 80 层, 80 heads, ~105B 参数），需4机执行
- 详见 `docs/scaling_to_110b.md`

---

## 指标 1.6：弹性扩缩容

> 支持千卡 BW1000 集群的分布式资源弹性，具备在训练千亿大模型过程中将训练资源从百卡扩容到千卡的能力

### 已完成验证

**设计原理**：仅扩缩数据并行（DP）维度，不涉及模型权重 reshard

- 模型权重按 TP / PP 切分，不随 DP 变化
- 优化器状态按 sharding rank 切分，扩缩时重新分配
- 操作流程：**stop → 修改 DP 配置 → resume from checkpoint（恢复模型权重和优化器状态）**

**单节点：4 卡 → 8 卡**（13B PT）

加速比 **2.0x**，接近线性扩展，Loss 曲线在扩容点无跳变。

  ![elastic_loss_curve](elastic_loss_curve.png)

**跨节点：8 卡 → 2 节点 16 卡**（13B PT）

扩容点 Loss 无跳变，训练无缝衔接。2 节点吞吐达到单节点 8 卡的 86%，已启用 RDMA 多端口 SHCA 通信（4 端口 `shca_0-3` + `NCCL_NET_PLUGIN=shca`），较 TCP 以太网模式（0.084 samples/s）提升 8.5x。

  ![elastic_2node_loss](elastic_2node_loss.png)

**扩缩容脚本**（`scripts/pt/run_elastic.sh`）
自动完成：kill 上次训练进程 → 检测最新 checkpoint → 注入 `resume_from_checkpoint`

### 待完成

- 百卡 → 千卡弹性扩缩容验证，需更大集群

---

## 环境与已解决问题

**硬件**：海光 BW1000 DCU，node109 × 8 卡 + node110 × 8 卡，共 2 节点 16 卡

**软件栈**：

| 组件 | 版本 / 来源 |
|:-----|:-----------|
| DTK | 25.04 |
| PaddlePaddle | 自编译 develop 分支，适配 dtk25 |
| PaddleFormers | pip install |

**环境问题**：

- **DTK25 发包不适用**：官网发包不适用于 dtk25，自行编译 Paddle develop + dtk25
- **GitHub 无法访问**：加中转代理
- **dtk25 适配**：修改 Paddle 源码、配置相关环境变量
- **网络隔离**：登录节点与计算节点隔离，二阶段编译（登录节点下载依赖，计算节点执行编译）
- **flashmask 不支持**：DCU 不支持，`_attn_implementation` 改用 `eager`
- **`fuse_rms_norm` 报错**：参数解析异常，配置中移除

**精度问题**：

- **`paddle.uniform` 随机数不一致**：依赖 `multiProcessorCount`，GPU 与 DCU 生成不同随机数，实现跨设备一致的随机数生成（rand）
- **`nn.CrossEntropyLoss` WARP_SIZE 硬编码**：硬编码 `PADDLE_WARP_SIZE=32`（GPU 值），DCU 为 64，改为设备自适应
