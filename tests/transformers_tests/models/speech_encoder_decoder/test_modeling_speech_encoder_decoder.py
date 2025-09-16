"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/speech_encoder_decoder/test_modeling_speech_encoder_decoder.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import numpy as np
import pytest
import torch
from transformers import SpeechEncoderDecoderConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

from ..bert.test_modeling_bert import BertModelTester
from ..wav2vec2.test_modeling_wav2vec2 import Wav2Vec2ModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class Wav2Vec2BertModelTest:
    def __init__(
        self,
        attn_implementation="eager",
    ):
        self.attn_implementation = attn_implementation

    def prepare_config_and_inputs(self):
        bert_model_tester = BertModelTester(attn_implementation=self.attn_implementation)
        wav2vec2_model_tester = Wav2Vec2ModelTester(attn_implementation=self.attn_implementation)
        encoder_config_and_inputs = wav2vec2_model_tester.prepare_config_and_inputs()
        decoder_config_and_inputs = bert_model_tester.prepare_config_and_inputs()
        (
            config,
            input_values,
            input_mask,
        ) = encoder_config_and_inputs
        (
            decoder_config,
            decoder_input_ids,
            decoder_token_type_ids,
            decoder_input_mask,
            decoder_sequence_labels,
            decoder_token_labels,
            decoder_choice_labels,
        ) = decoder_config_and_inputs

        # make sure that cross attention layers are added
        decoder_config.add_cross_attention = True
        encoder_decoder_config = SpeechEncoderDecoderConfig.from_encoder_decoder_configs(
            config, decoder_config, attn_implementation=self.attn_implementation
        )

        return encoder_decoder_config, input_values, input_mask, decoder_input_ids, decoder_input_mask


model_tester = Wav2Vec2BertModelTest()

(
    config,
    input_values,
    attention_mask,
    decoder_input_ids,
    decoder_attention_mask,
) = model_tester.prepare_config_and_inputs()

TEST_CASES = [
    [
        "SpeechEncoderDecoderModel",
        "transformers.SpeechEncoderDecoderModel",
        "mindone.transformers.SpeechEncoderDecoderModel",
        (config,),
        {},
        (),
        {
            "input_values": input_values,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
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
        for case in TEST_CASES
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
