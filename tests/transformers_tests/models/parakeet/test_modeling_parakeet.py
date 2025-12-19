"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/moshi/test_modeling_moshi.py."""

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
from transformers import ParakeetCTCConfig, ParakeetEncoderConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, random_attention_mask

# In test case of fp16, pt model and inputs will be cast to fp32 that leads diffs of ms and pt outputs larger.
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 2e-2, "bf16": 9e-3}
MODES = [1]


class ParakeetEncoderModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=1024,
        is_training=True,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="silu",
        dropout=0,  # so gradient checkpointing doesn't fail
        conv_kernel_size=9,
        subsampling_factor=8,
        subsampling_conv_channels=32,
        use_bias=True,
        num_mel_bins=80,
        scale_input=True,
    ):
        # testing suite parameters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_mel_bins = num_mel_bins
        self.is_training = is_training

        # config parameters
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.dropout = dropout
        self.conv_kernel_size = conv_kernel_size
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels
        self.use_bias = use_bias
        self.num_mel_bins = num_mel_bins
        self.scale_input = scale_input

        # Calculate output sequence length after subsampling
        self.output_seq_length = seq_length // subsampling_factor
        self.encoder_seq_length = self.output_seq_length
        self.key_length = self.output_seq_length

    def prepare_config_and_inputs(self):
        input_features = floats_numpy([self.batch_size, self.seq_length, self.num_mel_bins])
        attention_mask = random_attention_mask([self.batch_size, self.seq_length])
        config = self.get_config()

        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetEncoderConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            dropout=self.dropout,
            dropout_positions=self.dropout,
            layerdrop=self.dropout,
            activation_dropout=self.dropout,
            attention_dropout=self.dropout,
            conv_kernel_size=self.conv_kernel_size,
            subsampling_factor=self.subsampling_factor,
            subsampling_conv_channels=self.subsampling_conv_channels,
            use_bias=self.use_bias,
            num_mel_bins=self.num_mel_bins,
            scale_input=self.scale_input,
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class ParakeetForCTCModelTester:
    def __init__(self, encoder_kwargs=None, is_training=True, vocab_size=128, pad_token_id=0):
        if encoder_kwargs is None:
            encoder_kwargs = {}

        self.encoder_model_tester = ParakeetEncoderModelTester(**encoder_kwargs)
        self.is_training = is_training

        self.batch_size = self.encoder_model_tester.batch_size
        self.output_seq_length = self.encoder_model_tester.output_seq_length
        self.num_hidden_layers = self.encoder_model_tester.num_hidden_layers
        self.seq_length = vocab_size
        self.hidden_size = self.encoder_model_tester.hidden_size

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        _, input_features, attention_mask = self.encoder_model_tester.prepare_config_and_inputs()
        config = self.get_config()
        return config, input_features, attention_mask

    def get_config(self):
        return ParakeetCTCConfig.from_encoder_config(
            encoder_config=self.encoder_model_tester.get_config(),
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config, input_features, attention_mask = self.prepare_config_and_inputs()
        inputs_dict = {
            "input_features": input_features,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


ctc_model_tester = ParakeetForCTCModelTester()
ctc_config, ctc_inputs_dict = ctc_model_tester.prepare_config_and_inputs_for_common()
encoder_model_tester = ParakeetEncoderModelTester()
encoder_config, encoder_inputs_dict = encoder_model_tester.prepare_config_and_inputs_for_common()


TEST_CASES = [
    [
        "ParakeetEncoder",
        "transformers.ParakeetEncoder",
        "mindone.transformers.ParakeetEncoder",
        (encoder_config,),
        {},
        (encoder_inputs_dict["input_features"], None),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "ParakeetForCTC",
        "transformers.ParakeetForCTC",
        "mindone.transformers.ParakeetForCTC",
        (ctc_config,),
        {},
        (ctc_inputs_dict["input_features"], None),
        {},
        {
            "logits": "logits",
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
