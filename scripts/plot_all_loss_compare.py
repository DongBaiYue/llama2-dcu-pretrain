"""Llama-2-13B BW1000 vs GPU Loss & RE 对比绘图脚本 (6子图)

用法: python3 scripts/plot_all_loss_compare.py
输出: docs/all_loss_compare_300step_v2.png
数据: logs/pt/13b.log, logs/sft/13b.log, logs/lora/13b.log (BW1000)
      gpu_pt_300_loss.csv, gpu_sft_300_loss.csv, gpu_lora_300_loss.csv (GPU A100)
"""

import numpy as np
import csv
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_loss_from_log(logpath, max_steps=300):
    """从训练日志提取每步loss"""
    losses = []
    with open(logpath) as f:
        for line in f:
            m = re.search(r"loss: ([0-9.]+).*global_step: (\d+)", line)
            if m:
                step = int(m.group(2))
                loss = float(m.group(1))
                while len(losses) < step - 1:
                    losses.append(None)
                if step == len(losses) + 1:
                    losses.append(loss)
                else:
                    losses[step - 1] = loss
    return np.array(losses[:max_steps], dtype=float)


def load_csv(path):
    """从CSV加载loss数据"""
    data = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(float(list(row.values())[1]))
    return np.array(data)


def compute_re(dcu, gpu):
    """计算逐步相对误差"""
    return np.abs(dcu - gpu) / gpu * 100


def compute_metrics(dcu, gpu, start=10):
    """计算稳态Mean RE和Cosine Similarity"""
    re = compute_re(dcu[start:], gpu[start:])
    cos = np.dot(dcu[start:], gpu[start:]) / (
        np.linalg.norm(dcu[start:]) * np.linalg.norm(gpu[start:])
    )
    return np.mean(re), cos


# ====== 加载数据 ======
bw_pt = extract_loss_from_log("logs/pt/13b.log")
bw_sft = extract_loss_from_log("logs/sft/13b.log")
bw_lora = extract_loss_from_log("logs/lora/13b.log")
gpu_pt = load_csv("gpu_pt_300_loss.csv")
gpu_sft = load_csv("gpu_sft_300_loss.csv")
gpu_lora = load_csv("gpu_lora_300_loss.csv")

# Eval losses
bw_pt_eval, gpu_pt_eval = 6.1615, 6.1075
bw_sft_eval, gpu_sft_eval = 0.5106, 0.5114
bw_lora_eval, gpu_lora_eval = 0.5467, 0.5478

# ====== 计算指标 ======
warmup = 10
steps = np.arange(1, 301)

pt_re = compute_re(bw_pt, gpu_pt)
sft_re = compute_re(bw_sft, gpu_sft)
lora_re = compute_re(bw_lora, gpu_lora)

pt_mean_re, pt_cos = compute_metrics(bw_pt, gpu_pt)
sft_mean_re, sft_cos = compute_metrics(bw_sft, gpu_sft)
lora_mean_re, lora_cos = compute_metrics(bw_lora, gpu_lora)

pt_eval_re = abs(bw_pt_eval - gpu_pt_eval) / gpu_pt_eval * 100
sft_eval_re = abs(bw_sft_eval - gpu_sft_eval) / gpu_sft_eval * 100
lora_eval_re = abs(bw_lora_eval - gpu_lora_eval) / gpu_lora_eval * 100

# ====== 绘图 ======
fig, axes = plt.subplots(3, 2, figsize=(22, 20))
plt.rcParams["font.size"] = 12

gpu_color = "#1f77b4"
bw_color = "#ff7f0e"
re_color = "#d62728"

tasks = [
    ("PT", bw_pt, gpu_pt, pt_re, pt_mean_re, pt_cos, pt_eval_re, bw_pt_eval, gpu_pt_eval),
    ("SFT", bw_sft, gpu_sft, sft_re, sft_mean_re, sft_cos, sft_eval_re, bw_sft_eval, gpu_sft_eval),
    ("LoRA", bw_lora, gpu_lora, lora_re, lora_mean_re, lora_cos, lora_eval_re, bw_lora_eval, gpu_lora_eval),
]

for i, (name, bw, gpu, re, mean_re, cos, eval_re, bw_eval, gpu_eval) in enumerate(tasks):
    # 左列: Loss曲线
    ax_loss = axes[i, 0]
    ax_loss.plot(steps, gpu, color=gpu_color, linewidth=1.2, label="GPU (A100)", alpha=0.9)
    ax_loss.plot(steps, bw, color=bw_color, linewidth=1.2, label="DCU BW1000", alpha=0.9, linestyle="--")
    ax_loss.set_title(f"{name} Loss Curve", fontsize=14, fontweight="bold")
    ax_loss.set_xlabel("Step", fontsize=12)
    ax_loss.set_ylabel("Loss", fontsize=12)
    ax_loss.legend(fontsize=10, loc="upper right")
    ax_loss.grid(True, alpha=0.3)
    loss_text = f"Mean RE={mean_re:.2f}%  Cos={cos:.4f}\nEval RE={eval_re:.2f}%  (DCU={bw_eval:.4f}, GPU={gpu_eval:.4f})"
    ax_loss.text(
        0.02, 0.02, loss_text, transform=ax_loss.transAxes, fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
    )

    # 右列: RE曲线
    ax_re = axes[i, 1]
    ax_re.plot(steps, re, color=re_color, linewidth=0.8, alpha=0.7)
    ax_re.axvspan(1, warmup, alpha=0.1, color="gray", label="Warmup (1-10)")
    ax_re.axhline(y=mean_re, color="blue", linestyle=":", linewidth=1, alpha=0.7, label=f"Mean RE={mean_re:.2f}%")
    ax_re.set_title(f"{name} Step-wise Relative Error", fontsize=14, fontweight="bold")
    ax_re.set_xlabel("Step", fontsize=12)
    ax_re.set_ylabel("RE (%)", fontsize=12)
    ax_re.legend(fontsize=9, loc="upper right")
    ax_re.grid(True, alpha=0.3)
    re_pw = re[warmup:]
    ax_re.set_ylim(0, min(np.max(re) * 1.15, np.percentile(re_pw, 98) * 2.5))

plt.tight_layout(rect=[0, 0.02, 1, 0.97])
fig.suptitle(
    "Llama-2-13B BW1000 vs GPU Loss & RE Comparison (FLAGS_deterministic_rng=1, step 11-300)",
    fontsize=16, fontweight="bold", y=0.99,
)

output_path = "docs/all_loss_compare_300step_v2.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved to {output_path}")
plt.close()
