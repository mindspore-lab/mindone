"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/dia/test_modeling_dia.py."""

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
from transformers import DiaConfig, DiaDecoderConfig, DiaEncoderConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 6e-3}
MODES = [1]


class DiaModelTester:
    def __init__(
        self,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=7,
        max_length=50,
        is_training=True,
        vocab_size=100,
        hidden_size=16,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=2,
        head_dim=8,
        decoder_hidden_size=32,  # typically larger than encoder
        hidden_act="silu",
        eos_token_id=97,  # special tokens all occur after eos
        pad_token_id=98,
        bos_token_id=99,
        delay_pattern=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.max_length = max_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.hidden_act = hidden_act
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # Set default delay pattern if not provided
        self.delay_pattern = delay_pattern if delay_pattern is not None else [0, 1, 2]
        self.num_channels = len(self.delay_pattern)

    def get_config(self):
        encoder_config = DiaEncoderConfig(
            max_position_embeddings=self.max_length,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_attention_heads,  # same as num_attention_heads for testing
            head_dim=self.head_dim,
            intermediate_size=self.intermediate_size,
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
        )

        decoder_config = DiaDecoderConfig(
            max_position_embeddings=self.max_length,
            num_hidden_layers=self.num_hidden_layers,
            hidden_size=self.decoder_hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=1,  # GQA
            head_dim=self.head_dim,
            cross_num_attention_heads=self.num_attention_heads,
            cross_head_dim=self.head_dim,
            cross_num_key_value_heads=1,  # GQA
            cross_hidden_size=self.hidden_size,  # match encoder hidden size
            vocab_size=self.vocab_size,
            hidden_act=self.hidden_act,
            num_channels=self.num_channels,
        )

        config = DiaConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            delay_pattern=self.delay_pattern,
        )

        return config

    def prepare_config_and_inputs(self) -> tuple[DiaConfig, dict]:
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = np.not_equal(input_ids, self.pad_token_id)

        decoder_input_ids = ids_numpy([self.batch_size, self.seq_length, self.num_channels], self.vocab_size)
        decoder_attention_mask = np.not_equal(decoder_input_ids[..., 0], self.pad_token_id)

        config = self.get_config()
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


model_tester = DiaModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs()


DIA_CASES = [
    [
        "DiaModel",
        "transformers.DiaModel",
        "mindone.transformers.DiaModel",
        (config,),
        {},
        (
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            inputs_dict["decoder_input_ids"],
            None,
            inputs_dict["decoder_attention_mask"],
        ),
        {},
        {
            "last_hidden_state": "last_hidden_state",
            "encoder_last_hidden_state": "encoder_last_hidden_state",
        },
    ],
    [
        "DiaForConditionalGeneration",
        "transformers.DiaForConditionalGeneration",
        "mindone.transformers.DiaForConditionalGeneration",
        (config,),
        {},
        (
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            inputs_dict["decoder_input_ids"],
            None,
            inputs_dict["decoder_attention_mask"],
        ),
        {},
        {
            "logits": "logits",
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
        for case in DIA_CASES
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
