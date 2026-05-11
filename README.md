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

### 26B 模型

26B 由 13B 配置等比扩张得到。当前仓库使用 **深度翻倍** 的方式，从 40 层扩到 80 层；Tokenizer 相关文件沿用 13B。

```bash
mkdir -p models/Llama-2-26B

cp models/Llama-2-13b/added_tokens.json models/Llama-2-26B/
cp models/Llama-2-13b/config.json models/Llama-2-26B/
cp models/Llama-2-13b/generation_config.json models/Llama-2-26B/
cp models/Llama-2-13b/special_tokens_map.json models/Llama-2-26B/
cp models/Llama-2-13b/tokenizer_config.json models/Llama-2-26B/
cp models/Llama-2-13b/tokenizer.model models/Llama-2-26B/
```

这里实际修改的模型文件是 `models/Llama-2-26B/config.json`，需要将其中的 `num_hidden_layers` 从 `40` 调整为 `80`。

### 52B 模型

52B 使用 **2 倍深度 + 约 1.4 倍宽度** 的均衡扩张方案，适配 `node109 + node110` 两机 16 卡预训练。

推荐结构：

- `hidden_size: 7168`
- `intermediate_size: 19456`
- `num_hidden_layers: 80`
- `num_attention_heads: 56`
- `num_key_value_heads: 56`

模型目录可直接通过脚本生成：

```bash
bash scripts/pt/prepare_52b_model.sh
```

这里实际生成并修改的模型文件是 `models/Llama-2-52B/config.json`。脚本会基于 `models/Llama-2-13b/config.json` 生成它，并将以下字段改为 52B 结构：

- `hidden_size: 7168`
- `intermediate_size: 19456`
- `num_hidden_layers: 80`
- `num_attention_heads: 56`
- `num_key_value_heads: 56`

同时会复制 tokenizer 相关文件；不会复制 13B 的权重索引文件。

## 4. 启动训练

训练前建议先检查设备状态：

```bash
rocm-smi
```

### Llama-2 预训练

```bash
bash scripts/pt/run.sh          # 13B
bash scripts/pt/run_26b.sh      # 26B
bash scripts/pt/run_52b_2node.sh # 52B
```

### Llama-3 预训练

```bash
bash scripts/pt/run_llama3_8b.sh
```

### Llama-2 SFT

```bash
bash scripts/sft/run.sh
bash scripts/sft/run_26b.sh
```

### Llama-3 SFT

```bash
bash scripts/sft/run_llama3_8b.sh
```

### Llama-2 LoRA

```bash
bash scripts/lora/run.sh
bash scripts/lora/run_26b.sh
```

### Llama-3 LoRA

```bash
bash scripts/lora/run_llama3_8b.sh
```

### 按节点拆分执行训练脚本

```bash
# node109: 4卡任务
bash scripts/run_all_node109.sh

# node110: 8卡 / 26B 任务
bash scripts/run_all_node110.sh
```

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
tail -f logs/pt/13b.log
watch -n 1 rocm-smi
```

## 7. 注意事项

1. 训练前确认 DCU 空闲。
2. 26B 需要 8 卡。
3. 52B 需要 `node109 + node110` 两机共 16 卡，当前采用 `mpirun` 作为双机启动方式，并通过 `NNODES`、`MASTER_ADDR`、`MASTER_PORT`、`RANK` 组装多节点训练；若未显式设置 `MASTER_ADDR`，脚本会自动解析**当前发起节点的首个 IP** 作为 master 地址。
4. 本仓库脚本默认使用 `/public/home/baidu_test/hygon_2030/py310` 虚拟环境。
5. 训练脚本会显式设置 `PYTHONPATH="$PROJECT_ROOT/../Paddle/build/python:${PYTHONPATH:-}"`，以优先使用本地 Paddle build。
6. DCU 不支持 `flashmask`，配置中的 `_attn_implementation` 请使用 `eager`。
7. 当前配置中不要再使用 `fuse_rms_norm`，否则 `paddleformers-cli` 会报参数解析错误。

## 8. 参考资料

- [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers)
- [千亿模型扩张指南](docs/scaling_to_110b.md)
