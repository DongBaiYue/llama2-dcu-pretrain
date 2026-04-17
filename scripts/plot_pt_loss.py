"""PT Loss 对比绘图脚本: DCU vs GPU

用法: python3 scripts/plot_pt_loss.py
输出: docs/pt_loss_compare.png
"""
import re
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def extract_loss(log_path):
    """从训练日志提取 (step, loss) 数据"""
    steps, losses = [], []
    pattern = re.compile(r'loss: ([0-9.]+), learning_rate: [0-9.e-]+, global_step: (\d+)')
    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                losses.append(float(m.group(1)))
                steps.append(int(m.group(2)))
    return np.array(steps), np.array(losses)

steps_dcu, loss_dcu = extract_loss(os.path.join(PROJECT_ROOT, "logs", "pt", "13b.log"))
steps_gpu, loss_gpu = extract_loss(os.path.join(PROJECT_ROOT, "logs", "pt", "13b_gpu.log"))

# Metrics
dcu_mean = np.mean(loss_dcu)
gpu_mean = np.mean(loss_gpu)
dcu_final10 = np.mean(loss_dcu[-10:])
gpu_final10 = np.mean(loss_gpu[-10:])
dcu_min = np.min(loss_dcu)
gpu_min = np.min(loss_gpu)
mean_diff_pct = (dcu_mean - gpu_mean) / dcu_mean * 100
final10_diff_pct = (dcu_final10 - gpu_final10) / dcu_final10 * 100

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
fig.suptitle("Llama-2-13B PT Loss: DCU vs GPU", fontsize=16, fontweight='bold', y=0.96)

# --- Top: Loss curve ---
ax1.plot(steps_dcu, loss_dcu, alpha=0.7, color='#1f77b4', linewidth=1.2, label=f'DCU (avg={dcu_mean:.3f})')
ax1.plot(steps_gpu, loss_gpu, alpha=0.7, color='#d62728', linewidth=1.2, label=f'GPU (avg={gpu_mean:.3f})')

# Annotate start (offset to avoid overlap)
ax1.annotate(f'DCU start: {loss_dcu[0]:.3f}',
             xy=(steps_dcu[0], loss_dcu[0]),
             xytext=(18, 18), textcoords='offset points',
             fontsize=9, color='#1f77b4', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2))
ax1.annotate(f'GPU start: {loss_gpu[0]:.3f}',
             xy=(steps_gpu[0], loss_gpu[0]),
             xytext=(18, -18), textcoords='offset points',
             fontsize=9, color='#d62728', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))

# Annotate end (offset to avoid overlap)
ax1.annotate(f'DCU end: {loss_dcu[-1]:.3f}',
             xy=(steps_dcu[-1], loss_dcu[-1]),
             xytext=(-110, 18), textcoords='offset points',
             fontsize=9, color='#1f77b4', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=1.2))
ax1.annotate(f'GPU end: {loss_gpu[-1]:.3f}',
             xy=(steps_gpu[-1], loss_gpu[-1]),
             xytext=(-110, -18), textcoords='offset points',
             fontsize=9, color='#d62728', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='#d62728', lw=1.2))

# Warmup end marker
warmup_step = 10
ax1.axvline(x=warmup_step, color='gray', linestyle='--', alpha=0.6, linewidth=1)
ylim = ax1.get_ylim()
ax1.text(warmup_step + 1.5, ylim[0] + (ylim[1] - ylim[0]) * 0.92,
         'warmup end\n(step 10)', fontsize=8, color='gray')

# Metrics box
textstr = (
    f"DCU  mean={dcu_mean:.3f}  final10={dcu_final10:.3f}  min={dcu_min:.3f}\n"
    f"GPU  mean={gpu_mean:.3f}  final10={gpu_final10:.3f}  min={gpu_min:.3f}\n"
    f"diff  mean={mean_diff_pct:+.2f}%  final10={final10_diff_pct:+.2f}%"
)
props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='gray')
ax1.text(0.98, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
         verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')

ax1.set_ylabel('Loss', fontsize=12)
ax1.set_xlabel('Step', fontsize=10)
ax1.legend(loc='upper center', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- Bottom: Diff ---
diff = loss_dcu - loss_gpu
diff_pct = diff / loss_dcu * 100
ax2.bar(steps_dcu, diff_pct, width=0.8, color=np.where(diff_pct >= 0, '#2ca02c', '#d62728'), alpha=0.6)
ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.axhline(y=mean_diff_pct, color='orange', linestyle='--', linewidth=1.5,
            label=f'mean diff = {mean_diff_pct:+.2f}%')
ax2.set_ylabel('DCU-GPU Diff (%)', fontsize=11)
ax2.set_xlabel('Step', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out_path = os.path.join(PROJECT_ROOT, "docs", "pt_loss_compare.png")
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved to {out_path}")
