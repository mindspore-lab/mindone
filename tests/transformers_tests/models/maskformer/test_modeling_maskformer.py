# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
"""Testing suite for the MindSpore MaskFormer model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import DetrConfig, MaskFormerConfig, SwinConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class MaskFormerModelTester:
    def __init__(
        self,
        batch_size=2,
        is_training=True,
        use_auxiliary_loss=False,
        num_queries=10,
        num_channels=3,
        min_size=32 * 4,
        max_size=32 * 6,
        num_labels=4,
        mask_feature_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_auxiliary_loss = use_auxiliary_loss
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.min_size = min_size
        self.max_size = max_size
        self.num_labels = num_labels
        self.mask_feature_size = mask_feature_size
        # This is passed to the decoder config. We add it to the model tester here for testing
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.min_size, self.max_size])

        pixel_mask = np.ones([self.batch_size, self.min_size, self.max_size])

        mask_labels = (np.random.rand(self.batch_size, self.num_labels, self.min_size, self.max_size) > 0.5).astype(
            np.float32
        )
        class_labels = (np.random.rand(self.batch_size, self.num_labels) > 0.5).astype(np.int64)

        config = self.get_config()
        return config, pixel_values, pixel_mask, mask_labels, class_labels

    def get_config(self):
        return MaskFormerConfig.from_backbone_and_decoder_configs(
            backbone_config=SwinConfig(
                depths=[1, 1, 1, 1],
                embed_dim=16,
                hidden_size=32,
                num_heads=[1, 1, 2, 2],
            ),
            backbone=None,
            decoder_config=DetrConfig(
                decoder_ffn_dim=64,
                decoder_layers=self.num_hidden_layers,
                decoder_attention_heads=self.num_attention_heads,
                encoder_ffn_dim=64,
                encoder_layers=self.num_hidden_layers,
                encoder_attention_heads=self.num_attention_heads,
                num_queries=self.num_queries,
                d_model=self.mask_feature_size,
            ),
            mask_feature_size=self.mask_feature_size,
            fpn_feature_size=self.mask_feature_size,
            num_channels=self.num_channels,
            num_labels=self.num_labels,
        )


model_tester = MaskFormerModelTester()
config, pixel_values, pixel_mask, mask_labels, class_labels = model_tester.prepare_config_and_inputs()
MASKFORMER_CASES = [
    [
        "MaskFormerModel",
        "transformers.MaskFormerModel",
        "mindone.transformers.MaskFormerModel",
        (config,),
        {},
        (pixel_values,),
        {
            "pixel_mask": pixel_mask,
        },
        {
            "transformer_decoder_last_hidden_state": 2,
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
        for case in MASKFORMER_CASES
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

    # set `hidden_dtype` if requiring, for some modules always compute in float
    # precision and require specific `hidden_dtype` to cast before return
    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
    # print("ms:", ms_outputs)
    # print("pt:", pt_outputs)
    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            # print("===map", pt_key, ms_idx)
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
