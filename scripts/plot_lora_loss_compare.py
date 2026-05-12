#!/usr/bin/env python3
"""绘制 Llama-3-8B LoRA DCU vs GPU loss 对比曲线"""

import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def extract_loss(path):
    """从日志提取 step -> loss 映射"""
    data = {}
    with open(path, errors="replace") as f:
        for line in f:
            line = re.sub(r'\x1b\[[0-9;]*m', '', line)
            m = re.search(r"loss: ([\d.eE+-]+).*?global_step: (\d+)", line)
            if m:
                step = int(m.group(2))
                loss = float(m.group(1))
                if step not in data:
                    data[step] = loss
    return data

dcu = extract_loss("logs/lora/llama3-8b.log")
gpu = extract_loss("logs/lora/llama3-8b_gpu.log")

print("DCU steps: %d, GPU steps: %d" % (len(dcu), len(gpu)))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [3, 1]})

# --- 上图: loss 曲线 ---
dcu_steps = sorted(dcu.keys())
gpu_steps = sorted(gpu.keys())

ax1.plot(dcu_steps, [dcu[s] for s in dcu_steps], alpha=0.7, linewidth=0.8, label="DCU (BW1000)")
ax1.plot(gpu_steps, [gpu[s] for s in gpu_steps], alpha=0.7, linewidth=0.8, label="GPU (A100)")
ax1.set_ylabel("Loss", fontsize=12)
ax1.set_title("Llama-3-8B LoRA: DCU vs GPU Loss Curve", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# --- 下图: AE (绝对误差) ---
common = sorted(set(dcu.keys()) & set(gpu.keys()))
ae = [abs(dcu[s] - gpu[s]) for s in common]

ax2.bar(common, ae, width=1.0, alpha=0.6, color="steelblue")
ax2.set_xlabel("Step", fontsize=12)
ax2.set_ylabel("AE", fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = "docs/llama3_lora_loss_compare.png"
plt.savefig(out, dpi=150)
print("Saved to %s" % out)
