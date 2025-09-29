"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/moonshine/test_modeling_moonshine.py."""

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
from transformers import MoonshineConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 9e-3}
MODES = [1]


class MoonshineModelTester:
    def __init__(
        self,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=1000,
        is_training=False,
        use_labels=False,
        vocab_size=147,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        encoder_hidden_act="gelu",
        decoder_hidden_act="silu",
        decoder_start_token_id=85,
        bos_token_id=98,
        eos_token_id=98,
        pad_token_id=0,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.encoder_hidden_act = encoder_hidden_act
        self.decoder_hidden_act = decoder_hidden_act
        self.decoder_start_token_id = decoder_start_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        input_values = floats_numpy([self.batch_size, self.seq_length], scale=1.0)
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        decoder_input_ids = np.array(self.batch_size * [[self.decoder_start_token_id]])
        decoder_attention_mask = np.not_equal(decoder_input_ids, self.pad_token_id)

        config = self.get_config()

        return config, input_values, attention_mask, decoder_input_ids, decoder_attention_mask

    def get_config(self):
        return MoonshineConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            encoder_num_hidden_layers=self.num_hidden_layers,
            decoder_num_hidden_layers=self.num_hidden_layers,
            encoder_num_attention_heads=self.num_attention_heads,
            decoder_num_attention_heads=self.num_attention_heads,
            encoder_num_key_value_heads=self.num_key_value_heads,
            decoder_num_key_value_heads=self.num_key_value_heads,
            encoder_hidden_act=self.encoder_hidden_act,
            decoder_hidden_act=self.decoder_hidden_act,
            decoder_start_token_id=self.decoder_start_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )


model_tester = MoonshineModelTester()
(
    config,
    input_values,
    attention_mask,
    decoder_input_ids,
    decoder_attention_mask,
) = model_tester.prepare_config_and_inputs()


Moonshine_CASES = [
    [
        "MoonshineForConditionalGeneration",
        "transformers.MoonshineForConditionalGeneration",
        "mindone.transformers.MoonshineForConditionalGeneration",
        (config,),
        {},
        (
            input_values,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        ),
        {},
        {
            "logits": 0,
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "MoonshineModel",
        "transformers.MoonshineModel",
        "mindone.transformers.MoonshineModel",
        (config,),
        {},
        (
            input_values,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
        ),
        {},
        {
            "last_hidden_state": 0,
            "encoder_last_hidden_state": 2,
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
        for case in Moonshine_CASES
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
