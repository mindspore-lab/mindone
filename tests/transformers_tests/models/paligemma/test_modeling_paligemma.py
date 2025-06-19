# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Mindspore PaliGemma model."""

import inspect

import numpy as np
import pytest
import torch
from transformers.models.paligemma.configuration_paligemma import PaliGemmaConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)

from ..modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODE = [1]


class PaliGemmaVisionText2TextModelTester:
    def __init__(
        self,
        ignore_index=-100,
        image_token_index=0,
        projector_hidden_act="gelu",
        seq_length=25,
        vision_feature_select_strategy="default",
        vision_feature_layer=-1,
        projection_dim=32,
        text_config={
            "model_type": "gemma",
            "seq_length": 128,
            "is_training": True,
            # "use_input_mask": True,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "intermediate_size": 37,
            "hidden_activation": "gelu_pytorch_tanh",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 512,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        is_training=True,
        vision_config={
            "use_labels": True,
            "image_size": 20,
            "patch_size": 5,
            "num_image_tokens": 4,
            "num_channels": 3,
            "is_training": True,
            "hidden_size": 32,
            "projection_dim": 32,
            "num_key_value_heads": 1,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        use_cache=False,
    ):
        self.ignore_index = ignore_index
        # `image_token_index` is set to 0 to pass "resize_embeddings" test, do not modify
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.text_config = text_config
        self.vision_config = vision_config
        self.seq_length = seq_length
        self.projection_dim = projection_dim
        self.pad_token_id = text_config["pad_token_id"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.num_channels = vision_config["num_channels"]
        self.image_size = vision_config["image_size"]
        self.encoder_seq_length = seq_length
        self.use_cache = use_cache

    def get_config(self):
        return PaliGemmaConfig(
            attn_implementation="eager",
            text_config=self.text_config,
            vision_config=self.vision_config,
            ignore_index=self.ignore_index,
            image_token_index=self.image_token_index,
            projector_hidden_act=self.projector_hidden_act,
            projection_dim=self.projection_dim,
            vision_feature_select_strategy=self.vision_feature_select_strategy,
            vision_feature_layer=self.vision_feature_layer,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size - 1) + 1
        attention_mask = np.not_equal(input_ids, self.pad_token_id)

        # set the 16 first tokens to be image, and ensure that no other tokens are image tokens
        # do not change this unless you modified image size or patch size
        input_ids[input_ids == config.image_token_index] = self.pad_token_id
        input_ids[:, :16] = config.image_token_index
        return (
            config,
            pixel_values,
            input_ids,
            attention_mask,
            np.array(input_ids, dtype=np.int32),
            np.zeros_like(input_ids),
        )


model_tester = PaliGemmaVisionText2TextModelTester()
(
    config,
    pixel_values,
    input_ids,
    attention_mask,
    labels,
    token_type_ids,
) = model_tester.prepare_config_and_inputs_for_common()


PALIGEMMA_CASES = [
    [
        "PaliGemmaForConditionalGeneration",
        "transformers.PaliGemmaForConditionalGeneration",
        "mindone.transformers.PaliGemmaForConditionalGeneration",
        (config,),
        {},
        (),
        {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "token_type_ids": token_type_ids,
        },
        {
            "logits": 1,
        },
    ],
]


# transformers need >= 4.41.2
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
        for case in PALIGEMMA_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODE
    ],
)
def test_named_modules_1(
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
    pt_inputs_kwargs.update({"labels": torch.tensor(labels, dtype=torch.long)})
    if mode == 0:
        ms_inputs_kwargs.update({"return_dict": False, "use_cache": False})
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
