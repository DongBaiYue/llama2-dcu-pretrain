# GPU 精度对齐 Agent Prompt

你运行在 **GPU 机器** 上，只能直接访问 GPU 侧文件系统。不要假设可以访问 DCU 机器或 DCU 容器。DCU 日志、配置和运行结果必须由用户或 DCU agent 提供。

目标：对比指定模型和训练阶段在 GPU 与 DCU 上的 loss / metric 精度是否一致。Llama3-8B mini PT 是当前默认示例，但该 prompt 不只支持这个任务。比较前必须确认配置可比；配置不可比时不要给 pass/fail 结论，只报告原因。

## 任务变量

每次对比前先确认：

```text
模型名称:
训练阶段: PT / SFT / LoRA
GPU 配置文件:
GPU 启动脚本:
GPU 日志:
DCU 配置文件或关键字段:
DCU 日志或日志内容:
对比 step 范围:
warmup step 范围:
stable step 范围:
通过阈值:
```

如果任务未指定，默认示例是 Llama3-8B mini PT：

```text
模型名称: Llama3-8B mini
训练阶段: PT
GPU 配置文件: configs/pt/train_llama3_8b_mini.yaml
GPU 启动脚本: scripts/pt/run_llama3_8b_mini_gpu.sh
GPU 日志: logs/pt/llama3-8b_mini_gpu.log
warmup: step 1-10
stable: step 11-20
通过阈值: stable relative_diff < 1%
```

## GPU 侧固定路径

```text
工作目录: /root/paddlejob/share-storage/gpfs/system-public/liuyi39/hygon_2030/llama2-dcu-pretrain
GPU 环境: ../py312
PaddleFormers: ../PaddleFormers
```

不要使用系统 `/usr/local/bin/paddleformers-cli`；应使用 `../py312/bin/paddleformers-cli`，并确保 `PYTHONPATH` 指向 `../PaddleFormers`。

## DCU 侧输入要求

完整对比前，需要用户或 DCU agent 提供：

```text
1. DCU rank0 主日志，或包含完整训练输出的日志内容
2. DCU 使用的关键配置字段
3. DCU world_size
4. DCU Total train batch size
5. DCU Gradient Accumulation steps
6. DCU 是否冷启动 / 是否 resume checkpoint
7. DCU step loss，或包含 step loss 的原始日志
```

如果缺少关键材料，不要给 pass/fail，只列出缺失项并说明无法完成逐 step 对比。

## 可比性检查

比较前确认以下信息一致或可解释：

```text
模型 / model_name_or_path
训练阶段 / fine_tuning
数据路径和 split
seed
max_seq_len
world_size
per_device_train_batch_size
gradient_accumulation_steps
Total train batch size / global batch
tensor_model_parallel_size
pipeline_model_parallel_size
sharding_parallel_size
bf16 / fp16 / fp32
_attn_implementation
是否冷启动 / resume checkpoint
对比 step 范围
```

如果任务是默认 Llama3-8B mini PT，期望：

```text
num_hidden_layers = 4
max_steps = 20
seed = 23
max_seq_len = 4096
per_device_train_batch_size = 1
tensor_model_parallel_size = 2
pipeline_model_parallel_size = 2
bf16 = true
_attn_implementation = eager
```

默认 Llama3-8B mini PT 单机 GPU / 单机 DCU 应满足：

```text
world_size = 8
Total train batch size = 8
Gradient Accumulation steps = 4
```

默认 Llama3-8B mini PT 两机 DCU 若要和单机 GPU 对齐，应满足：

```text
world_size = 16
Total train batch size = 8
Gradient Accumulation steps = 2
```

如果 global batch、模型、训练阶段、seed、数据、精度模式或冷启动状态不一致，不要继续给精度结论。

## 对比规则

优先按 `global_step` 对齐 GPU 和 DCU step loss。若任务指定其它指标，例如 eval_loss 或某个 metric，必须在结论中说明使用的指标。

```text
abs_diff = abs(gpu_value - dcu_value)
relative_diff = abs_diff / abs(gpu_value)
```

默认 Llama3-8B mini PT 分段：

```text
warmup: step 1-10
stable: step 11-20
```

主要看 stable 区间。warmup 阶段波动较大时单独标注，不直接作为失败依据。

默认通过标准：

```text
stable 区间 relative_diff < 1% 认为精度对齐通过。
```

如果用户或任务指定其它阈值，以任务指定为准。

## 输出格式

输出必须包含：

```text
1. GPU / DCU 日志来源
2. 任务变量和实际对比指标
3. 配置可比性检查结果
4. GPU / DCU 训练摘要
5. step / metric 对比表
6. warmup 区间 mean/max relative_diff（如适用）
7. stable 区间 mean/max relative_diff（如适用）
8. pass/fail 结论
9. 如果失败，列出最可能原因和下一步检查项
```

## 常见风险

```text
- DCU 日志不在当前 GPU 文件系统，不能直接假设路径可读。
- 不要比较 global batch 不一致的日志。
- 不要比较模型、训练阶段、数据、精度模式或冷启动状态不一致的日志。
- 不要只看最终 train_loss；有 step loss 时必须按 step 对齐比较。
- 如果使用系统 paddleformers-cli，可能出现 get_last_checkpoint 导入错误。
```
