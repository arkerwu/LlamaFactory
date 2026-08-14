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
from collections import defaultdict


def analyze_sample_loss(path: str, z_threshold: float = 2.0) -> list[dict]:
    r"""Aggregate per-sample loss records and flag anomalies (max_loss > mean + z_threshold * std)."""
    samples: dict[str, list[float]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            samples[record["sample_id"]].append(record["loss"])

    all_losses = [loss for losses in samples.values() for loss in losses]
    mean = statistics.mean(all_losses)
    std = statistics.stdev(all_losses) if len(all_losses) > 1 else 0.0
    threshold = mean + z_threshold * std

    rows = []
    for sample_id, losses in samples.items():
        rows.append(
            {
                "sample_id": sample_id,
                "count": len(losses),
                "mean_loss": sum(losses) / len(losses),
                "max_loss": max(losses),
                "last_loss": losses[-1],
                "anomalous": max(losses) > threshold,
            }
        )

    rows.sort(key=lambda r: r["max_loss"], reverse=True)
    return rows


def print_report(rows: list[dict], top: int = 20) -> None:
    anomalous = [r for r in rows if r["anomalous"]]
    print(f"Total samples: {len(rows)}, anomalous: {len(anomalous)}")

    by_dataset: dict[str, int] = defaultdict(int)
    for r in anomalous:
        by_dataset[r["sample_id"].rsplit("_", 1)[0]] += 1

    for dataset, count in sorted(by_dataset.items(), key=lambda x: -x[1]):
        print(f"  {dataset}: {count} anomalous")

    print("\nTop anomalous samples (sample_id / count / mean / max / last):")
    for r in anomalous[:top]:
        print(
            f"  {r['sample_id']}  n={r['count']}  mean={r['mean_loss']:.4f}  max={r['max_loss']:.4f}  "
            f"last={r['last_loss']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze per-sample loss records.")
    parser.add_argument("path", type=str, help="Path to sample_loss.jsonl")
    parser.add_argument("--z-threshold", type=float, default=2.0)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()
    print_report(analyze_sample_loss(args.path, args.z_threshold), top=args.top)


if __name__ == "__main__":
    main()
