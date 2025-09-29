# coding=utf-8
# Copyright 2020 HuggingFace Inc. team.
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

import importlib
import inspect

import numpy as np
import pytest
import torch

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)

from ..bert.test_modeling_bert import BertModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
# graph mode is not supported in this model
MODES = [1]


def prepare_config_and_inputs():
    model_tester = BertModelTester()
    encoder_config_and_inputs = model_tester.prepare_config_and_inputs()
    decoder_config_and_inputs = model_tester.prepare_config_and_inputs_for_decoder()
    (
        config,
        input_ids,
        token_type_ids,
        input_mask,
        sequence_labels,
        token_labels,
        choice_labels,
    ) = encoder_config_and_inputs
    (
        decoder_config,
        decoder_input_ids,
        decoder_token_type_ids,
        decoder_input_mask,
        decoder_sequence_labels,
        decoder_token_labels,
        decoder_choice_labels,
        encoder_hidden_states,
        encoder_attention_mask,
    ) = decoder_config_and_inputs

    # make sure that cross attention layers are added
    decoder_config.add_cross_attention = True
    return {
        "config": config,
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "decoder_config": decoder_config,
        "decoder_input_ids": decoder_input_ids,
        "decoder_token_type_ids": decoder_token_type_ids,
        "decoder_attention_mask": decoder_input_mask,
        "decoder_sequence_labels": decoder_sequence_labels,
        "decoder_token_labels": decoder_token_labels,
        "decoder_choice_labels": decoder_choice_labels,
        "encoder_hidden_states": encoder_hidden_states,
        "labels": decoder_token_labels,
    }


input_dict = prepare_config_and_inputs()
ENCODER_DECODER_CASES = [
    [
        "BertModel",
        "transformers.BertModel",
        "mindone.transformers.BertModel",
        "BertLMHeadModel",
        "transformers.BertLMHeadModel",
        "mindone.transformers.BertLMHeadModel",
        "EncoderDecoderModel",
        "transformers.EncoderDecoderModel",
        "mindone.transformers.EncoderDecoderModel",
        (),
        input_dict,
        (input_dict["input_ids"],),
        {
            "attention_mask": input_dict["attention_mask"],
            "decoder_input_ids": input_dict["decoder_input_ids"],
            "decoder_attention_mask": input_dict["decoder_attention_mask"],
        },
        {
            "logits": 0,
        },
    ],
]


@pytest.mark.parametrize(
    "encoder_name,pt_encoder,ms_encoder,decoder_name,pt_decoder,ms_decoder,name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",  # noqa: E501
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in ENCODER_DECODER_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_named_modules(
    encoder_name,
    pt_encoder,
    ms_encoder,
    decoder_name,
    pt_decoder,
    ms_decoder,
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

    encoder_kwargs = {"config": init_kwargs["config"]}
    decoder_kwargs = {"config": init_kwargs["decoder_config"]}
    pt_encoder, ms_encoder, pt_dtype, ms_dtype = get_modules(
        pt_encoder, ms_encoder, dtype, *init_args, **encoder_kwargs
    )
    pt_decoder, ms_decoder, _, _ = get_modules(pt_decoder, ms_decoder, dtype, *init_args, **decoder_kwargs)

    pt_path, pt_cls_name = pt_module.rsplit(".", 1)
    ms_path, ms_cls_name = ms_module.rsplit(".", 1)
    pt_module_cls = getattr(importlib.import_module(pt_path), pt_cls_name)
    ms_module_cls = getattr(importlib.import_module(ms_path), ms_cls_name)

    pt_model = pt_module_cls(encoder=pt_encoder, decoder=pt_decoder)
    ms_model = ms_module_cls(encoder=ms_encoder, decoder=ms_decoder)

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
