# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the MindSpore VisionTextDualEncoder model."""

import numpy as np
import pytest
import torch
from transformers import BertConfig, CLIPVisionConfig, VisionTextDualEncoderConfig, ViTConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

from ..bert.test_modeling_bert import BertModelTester
from ..clip.test_modeling_clip import CLIPVisionModelTester

# from ..deit.test_modeling_deit import DeiTModelTester  # TODO
# from ..roberta.test_modeling_roberta import RobertaModelTester
from ..vit.test_modeling_vit import ViTModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class ViTBertModelTest:
    def get_pretrained_model_and_inputs(self):
        vision_config = ViTConfig.from_pretrained("hf-internal-testing/tiny-random-vit")
        text_config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-clip")
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)
        batch_size = 13
        pixel_values = floats_numpy(
            [
                batch_size,
                vision_config.num_channels,
                vision_config.image_size,
                vision_config.image_size,
            ]
        )
        input_ids = ids_numpy([batch_size, 4], text_config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return config, inputs

    def prepare_config_and_inputs(self):
        vit_model_tester = ViTModelTester()
        bert_model_tester = BertModelTester()
        vision_config_and_inputs = vit_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values, _ = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        vision_config._attn_implementation_internal = "eager"
        text_config._attn_implementation_internal = "eager"

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        vision_config = config_and_inputs["vision_config"]
        text_config = config_and_inputs["text_config"]
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)
        config._attn_implementation_internal = "eager"

        inputs_dict = {
            "pixel_values": config_and_inputs["pixel_values"],
            "attention_mask": config_and_inputs["attention_mask"],
            "input_ids": config_and_inputs["input_ids"],
            "token_type_ids": config_and_inputs["text_token_type_ids"],
        }
        return config, inputs_dict


class CLIPVisionBertModelTest:
    def get_pretrained_config_and_inputs(self):
        vision_config = CLIPVisionConfig.from_pretrained("hf-internal-testing/tiny-random-clip")
        text_config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-clip")
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)
        batch_size = 13
        pixel_values = floats_numpy(
            [
                batch_size,
                vision_config.num_channels,
                vision_config.image_size,
                vision_config.image_size,
            ]
        )
        input_ids = ids_numpy([batch_size, 4], text_config.vocab_size)
        attention_mask = random_attention_mask([batch_size, 4])
        inputs = {"pixel_values": pixel_values, "input_ids": input_ids, "attention_mask": attention_mask}

        return config, inputs

    def prepare_config_and_inputs(self):
        clip_model_tester = CLIPVisionModelTester()
        bert_model_tester = BertModelTester()
        vision_config_and_inputs = clip_model_tester.prepare_config_and_inputs()
        text_config_and_inputs = bert_model_tester.prepare_config_and_inputs()

        vision_config, pixel_values = vision_config_and_inputs

        (
            text_config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = text_config_and_inputs

        vision_config._attn_implementation_internal = "eager"
        text_config._attn_implementation_internal = "eager"

        return {
            "text_config": text_config,
            "vision_config": vision_config,
            "pixel_values": pixel_values,
            "attention_mask": input_mask,
            "input_ids": input_ids,
            "text_token_type_ids": token_type_ids,
            "text_sequence_labels": sequence_labels,
            "text_token_labels": token_labels,
            "text_choice_labels": choice_labels,
        }

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        vision_config = config_and_inputs["vision_config"]
        text_config = config_and_inputs["text_config"]
        config = VisionTextDualEncoderConfig.from_vision_text_configs(vision_config, text_config)
        config._attn_implementation_internal = "eager"
        inputs_dict = {
            "pixel_values": config_and_inputs["pixel_values"],
            "attention_mask": config_and_inputs["attention_mask"],
            "input_ids": config_and_inputs["input_ids"],
            "token_type_ids": config_and_inputs["text_token_type_ids"],
        }
        return config, inputs_dict


clip_bert_model_tester = CLIPVisionBertModelTest()
clip_bert_config, clip_bert_inputs_dict = clip_bert_model_tester.prepare_config_and_inputs_for_common()
vit_bert_model_tester = ViTBertModelTest()
vit_bert_config, vit_bert_inputs_dict = vit_bert_model_tester.prepare_config_and_inputs_for_common()


_CASES = [
    [
        "VisionTextDualEncoderModel",
        "transformers.VisionTextDualEncoderModel",
        "mindone.transformers.VisionTextDualEncoderModel",
        (clip_bert_config,),
        {},
        (),
        clip_bert_inputs_dict,
        {
            "logits_per_image": 0,
            "logits_per_text": 1,
        },
    ],
    [
        "VisionTextDualEncoderModel",
        "transformers.VisionTextDualEncoderModel",
        "mindone.transformers.VisionTextDualEncoderModel",
        (vit_bert_config,),
        {},
        (),
        vit_bert_inputs_dict,
        {
            "logits_per_image": 0,
            "logits_per_text": 1,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in _CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
)
def test_named_modules(
    name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype, mode
):
    ms.set_context(mode=mode)

    (pt_model, ms_model, pt_dtype, ms_dtype) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

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
