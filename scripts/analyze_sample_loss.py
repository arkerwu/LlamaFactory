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

import argparse
import json
import statistics
import unicodedata
from collections import defaultdict


def analyze_sample_loss(path: str, z_threshold: float = 2.0, min_step: int = 0) -> list[dict]:
    r"""Aggregate per-sample loss records and flag anomalies.

    A sample is anomalous when its max loss exceeds `mean + z_threshold * std` over all records
    AND is not below its dataset's median loss (a sample below its own dataset median indicates
    the dataset is shifted as a whole rather than the sample being an outlier).
    Records with `global_step < min_step` are skipped, so early unstable phases can be excluded.
    """
    samples: dict[str, list[float]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record["global_step"] < min_step:
                continue
            samples[record["sample_id"]].append(record["loss"])

    all_losses = [loss for losses in samples.values() for loss in losses]
    if not all_losses:
        return []
    mean = statistics.mean(all_losses)
    std = statistics.stdev(all_losses) if len(all_losses) > 1 else 0.0
    threshold = mean + z_threshold * std

    # 数据集基线按样本等权计算：每个样本取其损失均值作为代表值，再对全体样本取统计量
    dataset_sample_losses: dict[str, list[float]] = defaultdict(list)
    for sample_id, losses in samples.items():
        dataset_sample_losses[sample_id.rpartition("_")[0]].append(sum(losses) / len(losses))
    dataset_medians = {dataset: statistics.median(losses) for dataset, losses in dataset_sample_losses.items()}

    rows = []
    for sample_id, losses in samples.items():
        dataset = sample_id.rpartition("_")[0]
        rows.append(
            {
                "sample_id": sample_id,
                "count": len(losses),
                "mean_loss": sum(losses) / len(losses),
                "max_loss": max(losses),
                "last_loss": losses[-1],
                "anomalous": max(losses) > threshold and max(losses) >= dataset_medians[dataset],
            }
        )

    rows.sort(key=lambda r: r["max_loss"], reverse=True)
    return rows


def _display_width(text: str) -> int:
    r"""Return terminal display width, counting CJK characters as 2 cells."""
    return sum(2 if unicodedata.east_asian_width(char) in ("W", "F") else 1 for char in text)


def _format_table(header: list[str], table: list[list[str]]) -> None:
    r"""Print an aligned table; the first column is left-aligned, others right-aligned."""
    widths = [_display_width(cell) for cell in header]
    for row in table:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _display_width(cell))
    for row in [header, *table]:
        cells = []
        for i, cell in enumerate(row):
            padding = " " * (widths[i] - _display_width(cell))
            cells.append(cell + padding if i == 0 else padding + cell)
        print("  ".join(cells).rstrip())


def print_report(rows: list[dict], top: int = 20) -> None:
    anomalous = [r for r in rows if r["anomalous"]]
    print(f"样本总数: {len(rows)}，异常样本数: {len(anomalous)}")
    if not rows:
        print("没有符合筛选条件的记录，请检查 --min-step 是否大于文件中的最大 global_step")
        return

    # 数据集基线按样本等权计算：每个样本取其损失均值作为代表值，再对全体样本取统计量
    by_dataset: dict[str, dict] = defaultdict(lambda: {"records": 0, "samples": 0, "sample_losses": [], "anomalous": 0})
    for r in rows:
        stats = by_dataset[r["sample_id"].rsplit("_", 1)[0]]
        stats["records"] += r["count"]
        stats["samples"] += 1
        stats["sample_losses"].append(r["mean_loss"])
        stats["anomalous"] += r["anomalous"]

    dataset_medians = {dataset: statistics.median(stats["sample_losses"]) for dataset, stats in by_dataset.items()}

    print("\n各数据集损失概览（按中位数降序）:")
    overview = [
        [
            dataset,
            str(stats["records"]),
            str(stats["samples"]),
            f"{statistics.mean(stats['sample_losses']):.4f}",
            f"{dataset_medians[dataset]:.4f}",
            str(stats["anomalous"]),
        ]
        for dataset, stats in sorted(
            by_dataset.items(), key=lambda x: statistics.median(x[1]["sample_losses"]), reverse=True
        )
    ]
    _format_table(["数据集", "记录数", "样本数", "均值", "中位数", "异常数"], overview)

    print("\n异常最严重的样本（按损失降序，取前 {} 个）:".format(top))
    detail = []
    for r in anomalous[:top]:
        dataset, _, index = r["sample_id"].rpartition("_")
        detail.append([dataset, index, f"{r['max_loss']:.4f}", f"{dataset_medians[dataset]:.4f}"])
    if detail:
        _format_table(["数据集", "序号", "损失", "数据集中位"], detail)
    else:
        print("无")


def main() -> None:
    parser = argparse.ArgumentParser(description="分析逐样本损失记录，找出异常样本。")
    parser.add_argument("path", type=str, help="sample_loss.jsonl 文件路径")
    parser.add_argument("--z-threshold", type=float, default=2.0, help="异常判定阈值系数，异常指最大损失超过 均值+该系数*标准差")
    parser.add_argument("--top", type=int, default=20, help="展示异常最严重的前 N 个样本")
    parser.add_argument(
        "--min-step",
        type=int,
        default=0,
        help="只统计 global_step >= 该值的记录（用于跳过前期不稳定的训练阶段）。",
    )
    args = parser.parse_args()
    print_report(analyze_sample_loss(args.path, args.z_threshold, args.min_step), top=args.top)


if __name__ == "__main__":
    main()
