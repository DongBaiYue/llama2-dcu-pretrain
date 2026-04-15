# Llama-2 DCU 训练项目

在国产 DCU 上复现 Llama-2 的预训练、SFT 和 LoRA 训练流程。

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
# python -m pip install paddlepaddle-dcu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/dcu/
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

## 3. 模型准备

### 13B 基座模型

```bash
mkdir -p models/Llama-2-13b
modelscope download --model modelscope/Llama-2-13b-ms --local_dir models/Llama-2-13b
```

### 26B 模型

26B 由 13B 配置扩张得到：

```bash
mkdir -p models/Llama-2-26B
cp models/Llama-2-13b/*.json models/Llama-2-13b/*.model models/Llama-2-26B/
```

> 说明：`num_hidden_layers` 由 40 调整为 80，具体配置见 `configs/pt/train_26b.yaml`。

## 4. 启动训练

训练前建议先检查设备状态：

```bash
rocm-smi
```

### 预训练

```bash
bash scripts/pt/run.sh
bash scripts/pt/run_26b.sh
```

### SFT

```bash
bash scripts/sft/run.sh
bash scripts/sft/run_26b.sh
```

### LoRA

```bash
bash scripts/lora/run.sh
bash scripts/lora/run_26b.sh
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
tail -f logs/pt/train.log
watch -n 1 rocm-smi
```

## 7. 注意事项

1. 训练前确认 DCU 空闲。
2. 26B 需要 8 卡。
3. 本仓库脚本默认使用 `/public/home/baidu_test/hygon_2030/py310` 虚拟环境。

## 8. 参考资料

- [PaddleFormers](https://github.com/PaddlePaddle/PaddleFormers)
- [千亿模型扩张指南](docs/scaling_to_110b.md)
