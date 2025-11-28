# coding=utf-8
# Copyright 2025 Google Inc. The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the MindSpore ShieldGemma2 model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import ShieldGemma2Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

# parity thresholds similar to other multimodal tests
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # run in pynative mode only for now


class ShieldGemma2ModelTester:
    def __init__(
        self,
        batch_size=2,
        seq_length=32,
        # text backbone (Gemma3Text-like tiny config)
        text_config={
            "model_type": "gemma3_text",
            "vocab_size": 128,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "intermediate_size": 37,
            "max_position_embeddings": 256,
            "hidden_act": "gelu_pytorch_tanh",
            "pad_token_id": 1,
        },
        # vision backbone (SiglipVision-like tiny config)
        vision_config={
            "model_type": "siglip_vision_model",
            "image_size": 16,
            "patch_size": 4,
            "num_channels": 3,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        # mm wrapper
        mm_tokens_per_image=16,  # 16x16 image with 4x4 patch -> 16 tokens
        boi_token_index=120,
        eoi_token_index=121,
        image_token_index=122,
        yes_token_index=123,
        no_token_index=124,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.text_config = text_config
        self.vision_config = vision_config
        self.mm_tokens_per_image = mm_tokens_per_image
        self.boi_token_index = boi_token_index
        self.eoi_token_index = eoi_token_index
        self.image_token_index = image_token_index
        self.pad_token_id = text_config["pad_token_id"]
        self.yes_token_index = yes_token_index
        self.no_token_index = no_token_index

    def get_config(self):
        return ShieldGemma2Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            mm_tokens_per_image=self.mm_tokens_per_image,
            boi_token_index=self.boi_token_index,
            eoi_token_index=self.eoi_token_index,
            image_token_index=self.image_token_index,
            yes_token_index=self.yes_token_index,
            no_token_index=self.no_token_index,
            attn_implementation="eager",
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        # pixel values (B, C, H, W)
        pixel_values = floats_numpy(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )

        # input ids (B, T) — reserve first K tokens for image placeholder tokens
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        # ensure no accidental image tokens except the region we set
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        # place image placeholders at the start; 1 boi + N image tokens + 1 eoi (ensure <= seq_length)
        num_img_tokens = min(self.mm_tokens_per_image, self.seq_length - 2)
        input_ids[:, : num_img_tokens + 2] = self.pad_token_id
        input_ids[:, 0] = config.boi_token_index
        input_ids[:, 1 : 1 + num_img_tokens] = config.image_token_index
        input_ids[:, 1 + num_img_tokens] = config.eoi_token_index

        attention_mask = np.not_equal(input_ids, self.pad_token_id)

        return config, pixel_values, input_ids, attention_mask


model_tester = ShieldGemma2ModelTester()
(
    config,
    pixel_values,
    input_ids,
    attention_mask,
) = model_tester.prepare_config_and_inputs()


SHIELDGEMMA2_CASES = [
    [
        "ShieldGemma2ForImageClassification",
        "transformers.ShieldGemma2ForImageClassification",
        "mindone.transformers.ShieldGemma2ForImageClassification",
        (config,),
        {},
        (),
        {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        {
            "logits": 0,
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
        for case in SHIELDGEMMA2_CASES
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

    # set `hidden_dtype` if requiring, for some modules always compute in float precision
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
