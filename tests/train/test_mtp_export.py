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

import pytest
import torch
from safetensors.torch import load_file, save_file

from llamafactory.train.tuner import _copy_mtp_weights


@pytest.fixture
def source_dir(tmp_path):
    d = tmp_path / "source"
    d.mkdir()
    return str(d)


@pytest.fixture
def export_dir(tmp_path):
    d = tmp_path / "export"
    d.mkdir()
    return str(d)


def _write_shard(directory: str, weights: dict, file_name: str = "model-00001-of-00002.safetensors") -> None:
    save_file(weights, f"{directory}/{file_name}", metadata={"format": "pt"})


def test_copies_mtp_and_updates_index(source_dir, export_dir):
    import os

    _write_shard(source_dir, {"model.layers.0.weight": torch.zeros(2), "mtp.0.fc.weight": torch.ones(3)})
    _write_shard(export_dir, {"model.layers.0.weight": torch.zeros(2)}, "model-00001-of-00001.safetensors")
    index = {
        "metadata": {"total_size": 4},
        "weight_map": {"model.layers.0.weight": "model-00001-of-00001.safetensors"},
    }
    with open(f"{export_dir}/model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f)

    _copy_mtp_weights(source_dir, export_dir)

    mtp = load_file(f"{export_dir}/model.mtp.safetensors")
    assert list(mtp.keys()) == ["mtp.0.fc.weight"]
    assert torch.equal(mtp["mtp.0.fc.weight"], torch.ones(3))
    with open(f"{export_dir}/model.safetensors.index.json", encoding="utf-8") as f:
        updated = json.load(f)
    assert updated["weight_map"]["mtp.0.fc.weight"] == "model.mtp.safetensors"
    assert updated["metadata"]["total_size"] == 4 + 3 * 4  # float32: 3 elements * 4 bytes
    # original weights untouched
    assert torch.equal(
        load_file(f"{export_dir}/model-00001-of-00001.safetensors")["model.layers.0.weight"], torch.zeros(2)
    )
    # source untouched
    assert set(os.listdir(export_dir)) >= {"model.mtp.safetensors", "model.safetensors.index.json"}


def test_no_mtp_in_source_is_noop(source_dir, export_dir):
    _write_shard(source_dir, {"model.layers.0.weight": torch.zeros(2)})
    _write_shard(export_dir, {"model.layers.0.weight": torch.zeros(2)})

    _copy_mtp_weights(source_dir, export_dir)

    import os

    assert not any(f.endswith(".mtp.safetensors") for f in os.listdir(export_dir))


def test_skips_when_export_already_has_mtp(source_dir, export_dir):
    _write_shard(source_dir, {"mtp.0.fc.weight": torch.ones(3)})
    _write_shard(export_dir, {"mtp.0.fc.weight": torch.full((3,), 9.0)})  # e.g. future transformers saved it

    _copy_mtp_weights(source_dir, export_dir)

    import os

    assert not any(f.endswith(".mtp.safetensors") for f in os.listdir(export_dir))
    assert torch.equal(
        load_file(f"{export_dir}/model-00001-of-00002.safetensors")["mtp.0.fc.weight"], torch.full((3,), 9.0)
    )


def test_empty_source_is_noop(source_dir, export_dir):
    _write_shard(export_dir, {"model.layers.0.weight": torch.zeros(2)})

    _copy_mtp_weights(source_dir, export_dir)

    import os

    assert not any(f.endswith(".mtp.safetensors") for f in os.listdir(export_dir))
