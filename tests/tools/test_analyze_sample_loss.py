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

import json

from scripts.analyze_sample_loss import analyze_sample_loss


def test_analyze_sample_loss(tmp_path):
    path = tmp_path / "sample_loss.jsonl"
    # 20 normal samples at loss=2.0 and 1 anomalous at loss=100.0
    # all_losses = [2.0]*20 + [100.0], mean≈6.667, std≈21.385 (Bessel), threshold≈49.44
    # 100.0 > 49.44, so ds_2 IS anomalous
    rows = [(i, "ds_0", 2.0) for i in range(10)]
    rows += [(i, "ds_1", 2.0) for i in range(10, 20)]
    rows += [(20, "ds_2", 100.0)]
    path.write_text(
        "\n".join(
            json.dumps({"micro_step": m, "global_step": 0, "sample_id": sid, "loss": loss}) for m, sid, loss in rows
        ),
        encoding="utf-8",
    )

    result = analyze_sample_loss(str(path))
    anomalous = [r["sample_id"] for r in result if r["anomalous"]]
    assert anomalous == ["ds_2"]  # 只有 ds_2 显著异常
    assert result[0]["sample_id"] == "ds_2"  # 按 max_loss 降序
    ds_0 = next(r for r in result if r["sample_id"] == "ds_0")
    assert ds_0["count"] == 10


def test_analyze_sample_loss_min_step(tmp_path):
    path = tmp_path / "sample_loss.jsonl"
    # ds_1 在早期(global_step=0)异常达 100.0，在后期(global_step=5)回到 2.0
    rows = [
        {"micro_step": 0, "global_step": 0, "sample_id": "ds_0", "loss": 2.0},
        {"micro_step": 1, "global_step": 0, "sample_id": "ds_1", "loss": 100.0},
        {"micro_step": 2, "global_step": 5, "sample_id": "ds_0", "loss": 2.0},
        {"micro_step": 3, "global_step": 5, "sample_id": "ds_1", "loss": 2.0},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = analyze_sample_loss(str(path), min_step=5)
    anomalous = [r["sample_id"] for r in result if r["anomalous"]]
    assert anomalous == []  # 只看后期时 ds_1 不再异常
    ds_0 = next(r for r in result if r["sample_id"] == "ds_0")
    assert ds_0["count"] == 1  # 早期记录被过滤
    assert ds_0["last_loss"] == 2.0


def test_analyze_sample_loss_no_records(tmp_path):
    path = tmp_path / "sample_loss.jsonl"
    path.write_text(
        json.dumps({"micro_step": 0, "global_step": 1, "sample_id": "ds_0", "loss": 2.0}), encoding="utf-8"
    )

    # min-step 大于全部 global_step 时应返回空列表而不是崩溃
    assert analyze_sample_loss(str(path), min_step=10) == []


def test_analyze_sample_loss_below_dataset_median(tmp_path):
    path = tmp_path / "sample_loss.jsonl"
    # hi 数据集: hi_0 损失 8.5, hi_1..hi_4 损失 12.0(本数据集中位数 12.0); norm 数据集: 40 个样本损失 0.1
    # z_threshold=1.0 时全局阈值约 4.9: hi_0 高于全局阈值但低于本数据集中位数，不应判为异常
    rows = [{"micro_step": 0, "global_step": 0, "sample_id": "hi_0", "loss": 8.5}]
    rows += [
        {"micro_step": i + 1, "global_step": 0, "sample_id": f"hi_{j}", "loss": 12.0}
        for i, j in enumerate(range(1, 5))
    ]
    rows += [
        {"micro_step": i + 10, "global_step": 0, "sample_id": f"norm_{j}", "loss": 0.1}
        for i, j in enumerate(range(40))
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    result = analyze_sample_loss(str(path), z_threshold=1.0)
    anomalous = {r["sample_id"] for r in result if r["anomalous"]}
    assert anomalous == {f"hi_{j}" for j in range(1, 5)}  # 仅 12.0 的样本异常，8.5 的因低于数据集中位数被排除
