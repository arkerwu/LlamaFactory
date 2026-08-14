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
    # all_losses = [2.0]*20 + [100.0], mean≈6.76, std≈21.56 (Bessel), threshold≈49.88
    # 100.0 > 49.88, so ds_2 IS anomalous
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
