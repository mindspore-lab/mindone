"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/speech_to_text/test_modeling_speech_to_text.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.
import inspect

import numpy as np
import pytest
import torch
from transformers import Speech2TextConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 2e-2}
MODES = [1]


def prepare_speech_to_text_inputs_dict(
    config,
    input_features,
    decoder_input_ids,
    attention_mask=None,
    decoder_attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = np.not_equal(input_features, 0)
    if decoder_attention_mask is None:
        decoder_attention_mask = np.not_equal(decoder_input_ids, config.pad_token_id)
    if head_mask is None:
        head_mask = np.ones((config.encoder_layers, config.encoder_attention_heads))
    if decoder_head_mask is None:
        decoder_head_mask = np.ones((config.decoder_layers, config.decoder_attention_heads))
    if cross_attn_head_mask is None:
        cross_attn_head_mask = np.ones((config.decoder_layers, config.decoder_attention_heads))
    return {
        # "input_ids": input_features,
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "decoder_attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class Speech2TextModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        num_conv_layers=2,
        conv_kernel_sizes=(5, 5),
        conv_channels=32,
        input_feat_per_channel=24,
        input_channels=1,
        hidden_act="relu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=20,
        max_source_positions=20,
        max_target_positions=20,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_channels = conv_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_features = floats_numpy([self.batch_size, self.seq_length, self.input_feat_per_channel], self.vocab_size)
        attention_mask = np.ones([self.batch_size, self.seq_length], dtype=np.int64)
        decoder_input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        decoder_input_ids = np.clip(decoder_input_ids, 2, None)

        config = self.get_config()
        inputs_dict = prepare_speech_to_text_inputs_dict(
            config,
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            attention_mask=attention_mask,
        )
        return config, inputs_dict

    def get_config(self):
        return Speech2TextConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            num_conv_layers=self.num_conv_layers,
            conv_kernel_sizes=self.conv_kernel_sizes,
            conv_channels=self.conv_channels,
            input_feat_per_channel=self.input_feat_per_channel,
            input_channels=self.input_channels,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            max_source_positions=self.max_source_positions,
            max_target_positions=self.max_target_positions,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )


model_tester = Speech2TextModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs()


SPEECH2TEXT_CASES = [
    [
        "Speech2TextModel",
        "transformers.Speech2TextModel",
        "mindone.transformers.Speech2TextModel",
        (config,),
        {},
        (
            inputs_dict["input_features"],
            inputs_dict["attention_mask"],
            inputs_dict["decoder_input_ids"],
            inputs_dict["decoder_attention_mask"],
            inputs_dict["head_mask"],
            inputs_dict["decoder_head_mask"],
            inputs_dict["cross_attn_head_mask"],
        ),
        {},
        {
            "last_hidden_state": "last_hidden_state",
            "past_key_values": "past_key_values",
            "encoder_last_hidden_state": "encoder_last_hidden_state",
        },
    ],
    [
        "Speech2TextForConditionalGeneration",
        "transformers.Speech2TextForConditionalGeneration",
        "mindone.transformers.Speech2TextForConditionalGeneration",
        (config,),
        {},
        (
            inputs_dict["input_features"],
            inputs_dict["attention_mask"],
            inputs_dict["decoder_input_ids"],
            inputs_dict["decoder_attention_mask"],
            inputs_dict["head_mask"],
            inputs_dict["decoder_head_mask"],
            inputs_dict["cross_attn_head_mask"],
        ),
        {},
        {
            "logits": "logits",
            "past_key_values": "past_key_values",
            "encoder_last_hidden_state": "encoder_last_hidden_state",
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
        for case in SPEECH2TEXT_CASES
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
