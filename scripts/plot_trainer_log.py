# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Plot training curves (loss and learning rate) from a trainer_log.jsonl file.

Usage:
    python scripts/plot_trainer_log.py output/trainer_log.jsonl
    python scripts/plot_trainer_log.py output/trainer_log.jsonl --smooth-window 20 --output train_curve.png
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def parse_time(s: str) -> float:
    r"""Parse 'H:MM:SS' or 'M:SS' to minutes."""
    parts = s.split(":")
    return int(parts[-2]) + int(parts[-1]) / 60 if len(parts) == 3 else int(parts[0]) + int(parts[1]) / 60


def load_records(path: Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def smooth(values: list[float], window: int) -> list[float]:
    if window <= 1 or window > len(values):
        return list(values)
    smoothed: list[float] = []
    half = window // 2
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        smoothed.append(sum(values[lo:hi]) / (hi - lo))
    return smoothed


def aggregate_by_step(records: list[dict]) -> dict[str, list[float | int]]:
    r"""Average per-step metrics; one trainer log line is recorded every 5 steps, so there may be duplicate steps."""
    by_step: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        step = r["current_steps"]
        for key in ("loss", "lr", "epoch"):
            if key in r:
                by_step[step][key].append(float(r[key]))

    steps, losses, lrs, epochs = [], [], [], []
    for step in sorted(by_step):
        metrics = by_step[step]
        steps.append(step)
        losses.append(sum(metrics["loss"]) / len(metrics["loss"]))
        lrs.append(sum(metrics["lr"]) / len(metrics["lr"]))
        epochs.append(sum(metrics["epoch"]) / len(metrics["epoch"]))
    return {"steps": steps, "loss": losses, "lr": lrs, "epoch": epochs}


def make_plot(data: dict, smooth_window: int, title: str) -> "matplotlib.figure.Figure":
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False

    fig, (ax_loss, ax_lr) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax_loss.plot(data["steps"], data["loss"], color="lightgray", linewidth=0.8, label="原始 loss")
    ax_loss.plot(
        data["steps"],
        smooth(data["loss"], smooth_window),
        color="tab:blue",
        linewidth=1.6,
        label=f"滑动均值 (窗口 {smooth_window})",
    )
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title(title)
    ax_loss.legend(loc="upper right")
    ax_loss.grid(True, alpha=0.3)

    ax_lr.plot(data["steps"], data["lr"], color="tab:orange", linewidth=1.2)
    ax_lr.set_xlabel("Global step")
    ax_lr.set_ylabel("Learning rate")
    ax_lr.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 trainer_log.jsonl 中的 loss / 学习率曲线。")
    parser.add_argument("path", type=Path, help="trainer_log.jsonl 文件路径")
    parser.add_argument("--smooth-window", type=int, default=20, help="loss 滑动均值窗口大小（步数）")
    parser.add_argument("--output", type=str, default=None, help="图片输出路径（不指定则交互显示）")
    parser.add_argument("--title", type=str, default=None, help="自定义图表标题")
    args = parser.parse_args()

    if not args.path.exists():
        print(f"找不到文件: {args.path}", file=sys.stderr)
        sys.exit(1)

    records = load_records(args.path)
    if not records:
        print("文件为空，没有可绘制的记录", file=sys.stderr)
        sys.exit(1)

    data = aggregate_by_step(records)
    title = args.title or f"训练曲线 ({args.path.parent.name}, {len(records)} 条记录)"
    fig = make_plot(data, args.smooth_window, title)

    if args.output:
        fig.savefig(args.output, dpi=120, bbox_inches="tight")
        print(f"已保存图表: {args.output}")
    else:
        import matplotlib.pyplot as plt
        try:
            plt.show()
        except Exception:
            fallback = args.path.with_suffix(".png")
            fig.savefig(fallback, dpi=120, bbox_inches="tight")
            print(f"未提供交互显示，已保存图表: {fallback}")


if __name__ == "__main__":
    main()