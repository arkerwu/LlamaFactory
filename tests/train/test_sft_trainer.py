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
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import DataCollatorWithPadding

from llamafactory.data import MultiModalDataCollatorForSeq2Seq, get_dataset, get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams import get_train_args
from llamafactory.model import load_model, load_tokenizer
from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer


DEMO_DATA = os.getenv("DEMO_DATA", "llamafactory/demo_data")

TINY_LLAMA3 = os.getenv("TINY_LLAMA3", "llamafactory/tiny-random-Llama-3")

TRAIN_ARGS = {
    "model_name_or_path": TINY_LLAMA3,
    "stage": "sft",
    "do_train": True,
    "finetuning_type": "lora",
    "dataset": "llamafactory/tiny-supervised-dataset",
    "dataset_dir": "ONLINE",
    "template": "llama3",
    "cutoff_len": 1024,
    "overwrite_output_dir": True,
    "per_device_train_batch_size": 1,
    "max_steps": 1,
    "report_to": "none",
}


@dataclass
class DataCollatorWithVerbose(DataCollatorWithPadding):
    verbose_list: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features = [
            {k: v for k, v in feature.items() if k in ["input_ids", "attention_mask", "labels"]}
            for feature in features
        ]
        self.verbose_list.extend(features)
        batch = super().__call__(features)
        return {k: v[:, :1] for k, v in batch.items()}  # truncate input length


@pytest.mark.parametrize("disable_shuffling", [False, True])
def test_shuffle(disable_shuffling: bool):
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        {
            "output_dir": os.path.join("output", f"shuffle{str(disable_shuffling).lower()}"),
            "disable_shuffling": disable_shuffling,
            **TRAIN_ARGS,
        }
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = DataCollatorWithVerbose(tokenizer=tokenizer)
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )
    trainer.train()
    if disable_shuffling:
        assert data_collator.verbose_list[0]["input_ids"] == dataset_module["train_dataset"][0]["input_ids"]
    else:
        assert data_collator.verbose_list[0]["input_ids"] != dataset_module["train_dataset"][0]["input_ids"]


def test_record_sample_loss_unit(tmp_path):
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.record_sample_loss = True
    trainer.state = SimpleNamespace(global_step=5)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), overwrite_output_dir=True)
    trainer._init_sample_loss_recording()
    trainer._record_sample_loss(["ds_0", "ds_1"], torch.tensor(2.5))
    trainer.flush_sample_loss()

    lines = (tmp_path / "sample_loss.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines]
    assert len(records) == 2
    assert records[0]["micro_step"] == 0 and records[1]["micro_step"] == 0
    assert records[0]["global_step"] == 5
    assert [r["sample_id"] for r in records] == ["ds_0", "ds_1"]
    assert records[0]["loss"] == 2.5


def test_record_sample_loss_disabled(tmp_path):
    trainer = object.__new__(CustomSeq2SeqTrainer)
    trainer.record_sample_loss = False
    trainer.state = SimpleNamespace(global_step=0)
    trainer.args = SimpleNamespace(output_dir=str(tmp_path), overwrite_output_dir=True)
    trainer._init_sample_loss_recording()
    trainer._record_sample_loss(["ds_0"], torch.tensor(2.5))
    assert not (tmp_path / "sample_loss.jsonl").exists()


@pytest.mark.runs_on(["cpu", "mps"])
def test_record_sample_loss_integration():
    output_dir = os.path.join("output", "record_sample_loss")
    model_args, data_args, training_args, finetuning_args, _ = get_train_args(
        {"output_dir": output_dir, "record_sample_loss": True, **TRAIN_ARGS}
    )
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    dataset_module = get_dataset(template, model_args, data_args, training_args, stage="sft", **tokenizer_module)
    model = load_model(tokenizer, model_args, finetuning_args, training_args.do_train)
    data_collator = MultiModalDataCollatorForSeq2Seq(
        template=template,
        pad_to_multiple_of=8,
        label_pad_token_id=IGNORE_INDEX,
        **tokenizer_module,
    )
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        **dataset_module,
        **tokenizer_module,
    )
    trainer.train()

    sample_loss_file = os.path.join(output_dir, "sample_loss.jsonl")
    assert os.path.exists(sample_loss_file)
    with open(sample_loss_file, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) > 0
    for record in records:
        assert set(record.keys()) == {"micro_step", "global_step", "sample_id", "loss"}
        assert str(record["sample_id"]).startswith("llamafactory/tiny-supervised-dataset_")
        assert isinstance(record["loss"], float)
