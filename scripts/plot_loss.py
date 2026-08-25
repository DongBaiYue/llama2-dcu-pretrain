#!/usr/bin/env python3
"""从训练日志提取 loss 并绘制收敛曲线 (step vs loss).

用法:
    python3 scripts/plot_loss.py <日志文件> [输出png]
    python3 scripts/plot_loss.py <日志文件1> <日志文件2> ... -o 输出png
    python3 scripts/plot_loss.py <日志文件1> <日志文件2> ... --first-max-step 30
说明:
    传入多条日志时，第一条曲线默认只绘制到 step 30；可用 --first-max-step 覆盖。
示例:
    python3 scripts/plot_loss.py logs/pt/llama3-100b_8node.dist/node_0/workerlog.0
    python3 scripts/plot_loss.py logs/pt/llama3-100b_8node.log -o logs/pt/loss.png

    python3 scripts/plot_loss.py logs/pt/llama3-100b_128node.log  -o logs/pt/loss_llama3-100b_128node.png
    python3 scripts/plot_loss.py logs/sft/llama3-100b_128node.log  -o logs/sft/loss_llama3-100b_128node.png
    python3 scripts/plot_loss.py logs/lora/llama3-100b_128node.log  -o logs/lora/loss_llama3-100b_128node.png
    python3 scripts/plot_loss.py logs/pt/llama3-100b_elastic_16node_step0.log logs/pt/llama3-100b_elastic_128node_step30.log -o logs/pt/loss_llama3-100b_elastic.png
"""
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境, 直接存图
import matplotlib.pyplot as plt

# loss 在前: "loss: X, ..., global_step: N"; 兼容 step 在前的反序
LOSS_RE = re.compile(r"loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?).*?global_step:\s*(\d+)")
STEP_RE = re.compile(r"global_step:\s*(\d+).*?loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def extract_loss(log_path):
    steps, losses, seen = [], [], set()
    with open(log_path, "r", errors="ignore") as f:
        for line in f:
            line = ANSI_RE.sub("", line)
            m = LOSS_RE.search(line)
            if m:
                loss, step = float(m.group(1)), int(m.group(2))
            else:
                m = STEP_RE.search(line)
                if not m:
                    continue
                step, loss = int(m.group(1)), float(m.group(2))
            if step in seen:  # 同 step 多 rank 重复打印, 取第一个
                continue
            seen.add(step)
            steps.append(step)
            losses.append(loss)

    if not steps:
        raise ValueError(f"{log_path} 未提取到 loss 数据")

    steps, losses = zip(*sorted(zip(steps, losses)))
    return list(steps), list(losses)


def parse_args(argv):
    log_paths = []
    out_path = "loss_curve.png"
    labels = None
    first_max_step = None

    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg == "-o" and i + 1 < len(argv):
            out_path = argv[i + 1]
            i += 2
            continue
        if arg == "--labels" and i + 1 < len(argv):
            labels = [item.strip() for item in argv[i + 1].split(",") if item.strip()]
            i += 2
            continue
        if arg == "--first-max-step" and i + 1 < len(argv):
            first_max_step = int(argv[i + 1])
            if first_max_step <= 0:
                raise SystemExit("--first-max-step 必须是正整数")
            i += 2
            continue
        if arg.startswith("-"):
            raise SystemExit(f"未知参数: {arg}")
        log_paths.append(arg)
        i += 1

    if not log_paths:
        raise SystemExit(f"用法: python3 {argv[0]} <日志文件> [输出png]")

    if labels is not None and len(labels) != len(log_paths):
        raise SystemExit("--labels 数量必须和日志文件数量一致")

    if first_max_step is None and len(log_paths) > 1:
        first_max_step = 30

    return log_paths, out_path, labels, first_max_step


def main():
    try:
        log_paths, out_path, labels, first_max_step = parse_args(sys.argv)
        series = []
        for index, log_path in enumerate(log_paths):
            steps, losses = extract_loss(log_path)
            if index == 0 and first_max_step is not None:
                filtered = [(step, loss) for step, loss in zip(steps, losses) if step <= first_max_step]
                if not filtered:
                    raise ValueError(f"{log_path} 在 step <= {first_max_step} 范围内未提取到 loss 数据")
                steps, losses = map(list, zip(*filtered))
            if labels is None:
                label = Path(log_path).stem
            else:
                label = labels[index]
            series.append((log_path, label, steps, losses))
    except (OSError, ValueError, SystemExit) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    plt.figure(figsize=(10, 6))
    for log_path, label, steps, losses in series:
        plt.plot(steps, losses, marker=".", markersize=3, linewidth=1, label=label)
        print(f"{label}: step {steps[0]}-{steps[-1]}, loss {min(losses):.4f}-{max(losses):.4f}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"曲线已保存: {out_path}")


if __name__ == "__main__":
    main()
