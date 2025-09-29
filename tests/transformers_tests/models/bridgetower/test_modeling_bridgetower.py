# coding=utf-8
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the MindSpore BridgeTower model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class BridgeTowerTextModelTester:
    def __init__(
        self,
        hidden_act="gelu",
        hidden_size=64,
        initializer_factor=1,
        layer_norm_eps=1e-05,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        tie_word_embeddings=False,
        output_hidden_states=False,
    ):
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = 99
        self.seq_length = 4
        self.batch_size = 1
        self.is_training = False
        self.output_hidden_states = output_hidden_states

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        return config, input_ids, attention_mask

    def get_config(self):
        return BridgeTowerTextConfig(
            hidden_act=self.hidden_act,
            hidden_size=self.hidden_size,
            initializer_factor=self.initializer_factor,
            layer_norm_eps=self.layer_norm_eps,
            num_attention_heads=self.num_attention_heads,
            num_hidden_layers=self.num_hidden_layers,
            intermediate_size=self.intermediate_size,
            tie_word_embeddings=self.tie_word_embeddings,
            output_hidden_states=self.output_hidden_states,
            vocab_size=self.vocab_size,
        )


class BridgeTowerImageModelTester:
    def __init__(
        self,
        hidden_size=64,
        initializer_factor=1,
        layer_norm_eps=1e-05,
        num_hidden_layers=2,
        init_layernorm_from_vision_encoder=False,
        output_hidden_states=False,
        image_size=64,
    ):
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.num_hidden_layers = num_hidden_layers
        self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder
        self.num_channels = 3
        self.num_image_features = 17
        self.batch_size = 1
        self.image_size = image_size
        self.is_training = False
        self.output_hidden_states = output_hidden_states

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = random_attention_mask([self.batch_size, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values, pixel_mask

    def get_config(self):
        return BridgeTowerVisionConfig(
            hidden_size=self.hidden_size,
            initializer_factor=self.initializer_factor,
            layer_norm_eps=self.layer_norm_eps,
            num_hidden_layers=self.num_hidden_layers,
            init_layernorm_from_vision_encoder=self.init_layernorm_from_vision_encoder,
            num_channels=self.num_channels,
            num_image_features=self.num_image_features,
            batch_size=self.batch_size,
            image_size=self.image_size,
            is_training=self.is_training,
            output_hidden_states=self.output_hidden_states,
        )


class BridgeTowerModelTester:
    def __init__(
        self,
        text_kwargs=None,
        vision_kwargs=None,
        share_cross_modal_transformer_layers=True,
        share_link_tower_layers=False,
        link_tower_type="add",
        init_layernorm_from_vision_encoder=False,
        contrastive_hidden_size=512,
        logit_scale_init_value=2.6592,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if vision_kwargs is None:
            vision_kwargs = {}

        self.text_model_tester = BridgeTowerTextModelTester(**text_kwargs)
        self.vision_model_tester = BridgeTowerImageModelTester(**vision_kwargs)

        self.share_cross_modal_transformer_layers = share_cross_modal_transformer_layers
        self.share_link_tower_layers = share_link_tower_layers
        self.link_tower_type = link_tower_type
        self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder
        self.contrastive_hidden_size = contrastive_hidden_size
        self.logit_scale_init_value = logit_scale_init_value

        self.batch_size = 1
        self.expected_num_hidden_layers = 8
        self.is_training = False

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size

    def prepare_config_and_inputs(self):
        text_config, input_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        vision_config, pixel_values, pixel_mask = self.vision_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return (config, input_ids, attention_mask, pixel_values, pixel_mask)

    def get_config(self):
        return BridgeTowerConfig.from_text_vision_configs(
            text_config=self.text_model_tester.get_config(),
            vision_config=self.vision_model_tester.get_config(),
            share_cross_modal_transformer_layers=self.share_cross_modal_transformer_layers,
            share_link_tower_layers=self.share_link_tower_layers,
            link_tower_type=self.link_tower_type,
            init_layernorm_from_vision_encoder=self.init_layernorm_from_vision_encoder,
            contrastive_hidden_size=self.contrastive_hidden_size,
            logit_scale_init_value=self.logit_scale_init_value,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
        )


model_tester = BridgeTowerModelTester()
config, input_ids, attention_mask, pixel_values, pixel_mask = model_tester.prepare_config_and_inputs()
BRIDGETOWER_CASES = [
    [
        "BridgeTowerModel",
        "transformers.BridgeTowerModel",
        "mindone.transformers.BridgeTowerModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
        },
        {
            "text_features": 0,
            "image_features": 1,
            "pooler_output": 2,
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
        for case in BRIDGETOWER_CASES
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
