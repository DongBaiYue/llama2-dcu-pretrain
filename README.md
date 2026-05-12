# Llama DCU 训练项目

在国产 DCU 上验证 Llama-2 / Llama-3 的预训练、SFT 和 LoRA 训练流程。

## 1. 环境准备

```bash
git clone https://github.com/DongBaiYue/llama2-dcu-pretrain.git
cd llama2-dcu-pretrain

source /public/home/baidu_test/hygon_2030/py310/bin/activate
```

### 安装 PaddleFormers

```bash
git clone https://github.com/PaddlePaddle/PaddleFormers.git
cd PaddleFormers
pip install -e .
# python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/gpu/
python -m pip install --pre paddlepaddle-dcu -i https://www.paddlepaddle.org.cn/packages/nightly/dcu/
```

## 2. 数据准备

### 预训练数据

```bash
cd ../llama2-dcu-pretrain
mkdir -p data/pt

wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.bin -O data/pt/llama_openwebtext_100k.bin
wget https://bj.bcebos.com/paddlenlp/models/transformers/llama/data/llama_openwebtext_100k.idx -O data/pt/llama_openwebtext_100k.idx
```

### SFT 数据

```bash
mkdir -p data/sft
wget https://paddlenlp.bj.bcebos.com/datasets/PDC_DATASETS/SFT/school_math_0.25M.tar.gz -O data/sft/school_math_0.25M.tar.gz
tar -xf data/sft/school_math_0.25M.tar.gz -C data/sft/
```

#### Llama-3 SFT 数据（Chat Template）

Llama-3 需要专用的对话格式标记，原始数据需转换：

```bash
python3 scripts/prepare_llama3_sft_data.py
```

## 3. 模型准备

### 13B 基座模型

```bash
mkdir -p models/Llama-2-13b
modelscope download --model modelscope/Llama-2-13b-ms --local_dir models/Llama-2-13b
```

### Llama-3-8B 模型

```bash
mkdir -p models/Llama-3-8b
modelscope download --model LLM-Research/Meta-Llama-3-8B --local_dir models/Llama-3-8b
```

### Llama-3-70B 模型

```bash
mkdir -p models/Llama-3-70b
modelscope download --model LLM-Research/Meta-Llama-3-70B --local_dir models/Llama-3-70b
```

### 100B 千亿模型

100B 由 Llama-3-70B 深度扩张得到，将 `num_hidden_layers` 从 80 扩到 120（≈104B 参数），其余结构不变。

模型目录可直接通过脚本生成：

```bash
bash scripts/pt/prepare_100b_model.sh
```

脚本会基于 `models/Llama-3-70b/` 复制 tokenizer 文件，并将 `config.json` 中的 `num_hidden_layers` 改为 120。

## 4. 启动训练

训练前建议先检查设备状态：

```bash
rocm-smi
```

### Llama-2 预训练

```bash
bash scripts/pt/run.sh          # 13B
```

### Llama-3 预训练

```bash
bash scripts/pt/run_llama3_8b.sh      # 8B
bash scripts/pt/run_llama3_70b.sh     # 70B (4 节点 32 卡)
bash scripts/pt/run_llama3_100b.sh    # 100B (4 节点 32 卡)
```

### Llama-2 SFT

```bash
bash scripts/sft/run.sh
```

### Llama-3 SFT

```bash
bash scripts/sft/run_llama3_8b.sh    # 8B
bash scripts/sft/run_llama3_70b.sh   # 70B (4 节点 32 卡)
```

### Llama-2 LoRA

```bash
bash scripts/lora/run.sh
```

### Llama-3 LoRA

```bash
bash scripts/lora/run_llama3_8b.sh    # 8B
bash scripts/lora/run_llama3_70b.sh   # 70B (4 节点 32 卡)
```

### 弹性扩缩容（Llama-3-8B）

支持通过调整数据并行度（sharding_parallel_size）实现弹性扩缩容，2 机可无缝扩至 4 机继续训练：

```bash
bash scripts/pt/run_llama3_8b_elastic.sh 2node dcu   # 2 机 16 卡
bash scripts/pt/run_llama3_8b_elastic.sh 4node dcu   # 扩容到 4 机 32 卡
bash scripts/pt/run_llama3_8b_elastic.sh 2node dcu   # 缩回 2 机 16 卡
```

脚本自动完成：kill 上次进程 → 检测最新 checkpoint → 设置 `resume_from_checkpoint` → 调整 `sharding_parallel_size`。

## 5. 项目结构

```text
.
├── configs/              # 训练配置
│   ├── pt/               # 预训练配置
│   ├── sft/              # SFT 配置
│   └── lora/             # LoRA 配置
├── data/                 # 数据目录
├── docs/                 # 说明文档
├── models/               # 模型目录
├── scripts/              # 启动脚本
├── checkpoints/          # 检查点输出
└── logs/                 # 训练日志
```

## 6. 训练监控

```bash
tail -f logs/pt/llama3-8b.log
watch -n 1 rocm-smi
```

## 7. 注意事项

1. 训练前确认 DCU 空闲。
2. 70B / 100B 需要 4 节点 32 卡，通过 `mpirun` 启动多节点训练；若未显式设置 `MASTER_ADDR`，脚本会自动解析**当前发起节点的首个 IP** 作为 master 地址。
3. 本仓库脚本默认使用 `/public/home/baidu_test/hygon_2030/py310` 虚拟环境。
4. 训练脚本会显式设置 `PYTHONPATH="$PROJECT_ROOT/../Paddle/build/python:${PYTHONPATH:-}"`，以优先使用本地 Paddle build。
5. DCU 不支持 `flashmask`，配置中的 `_attn_implementation` 请使用 `eager`。
6. 当前配置中不要再使用 `fuse_rms_norm`，否则 `paddleformers-cli` 会报参数解析错误。

## 8. 参考资料

- [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers)
- [千亿模型扩张指南](docs/scaling_to_110b.md)
