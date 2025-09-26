# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import inspect
import math

import numpy as np
import pytest
import torch
from transformers import ResNetConfig, TableTransformerConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

# nn.functional.grid_sample not support fp16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class TableTransformerModelTester:
    config_class = TableTransformerConfig

    def __init__(
        self,
        batch_size=8,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=12,
        num_channels=3,
        min_size=200,
        max_size=200,
        n_targets=8,
        num_labels=3,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.n_targets = n_targets
        self.num_labels = num_labels

        # we also set the expected seq length for both encoder and decoder
        self.encoder_seq_length = math.ceil(self.min_size / 32) * math.ceil(self.max_size / 32)
        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.min_size, self.max_size])
        pixel_mask = np.ones([self.batch_size, self.min_size, self.max_size])

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {
                    "class_labels": np.random.randint(low=0, high=self.num_labels, size=(self.n_targets,)),
                    "boxes": np.random.rand(self.n_targets, 4),
                    "masks": np.random.rand(self.n_targets, self.min_size, self.max_size),
                }
                labels.append(target)

        config = self.get_config()
        return config, pixel_values, pixel_mask, labels

    def get_config(self):
        resnet_config = ResNetConfig(
            num_channels=3,
            embeddings_size=10,
            hidden_sizes=[10, 20, 30, 40],
            depths=[1, 1, 2, 1],
            hidden_act="relu",
            num_labels=3,
            out_features=["stage2", "stage3", "stage4"],
            out_indices=[2, 3, 4],
        )
        return TableTransformerConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            use_timm_backbone=False,
            backbone_config=resnet_config,
            backbone=None,
            use_pretrained_backbone=False,
        )


model_tester = TableTransformerModelTester()
config, pixel_values, pixel_mask, labels = model_tester.prepare_config_and_inputs()

TABLE_TRANSFORMER_CASES = [
    [
        "TableTransformerModel",
        "transformers.TableTransformerModel",
        "mindone.transformers.TableTransformerModel",
        (config,),
        {},
        (),
        {"pixel_values": pixel_values, "pixel_mask": pixel_mask},
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "TableTransformerForObjectDetection",
        "transformers.TableTransformerForObjectDetection",
        "mindone.transformers.TableTransformerForObjectDetection",
        (config,),
        {},
        (),
        {"pixel_values": pixel_values, "pixel_mask": pixel_mask},
        {
            "last_hidden_state": 2,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in TABLE_TRANSFORMER_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    name,
    pt_module,
    ms_module,
    init_args,
    init_kwargs,
    inputs_args,
    inputs_kwargs,
    outputs_map,
    dtype,
    mode,
):
    ms.set_context(mode=mode)

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[ms_idx]
            if isinstance(pt_output, (list, tuple)):
                pt_outputs_n += list(pt_output)
                ms_outputs_n += list(ms_output)
            else:
                pt_outputs_n.append(pt_output)
                ms_outputs_n.append(ms_output)
        diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
    else:
        diffs = compute_diffs(pt_outputs, ms_outputs)

    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
