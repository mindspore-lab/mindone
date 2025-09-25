"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/fastspeech2_conformer/test_modeling_fastspeech2_conformer.py."""

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
from transformers import FastSpeech2ConformerConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4}
MODES = [1]


class FastSpeech2ConformerModelTester:
    def __init__(
        self,
        batch_size=13,
        num_hidden_layers=1,
        num_attention_heads=2,
        hidden_size=24,
        seq_length=7,
        encoder_linear_units=384,
        decoder_linear_units=384,
        is_training=False,
        speech_decoder_postnet_units=128,
        speech_decoder_postnet_layers=2,
        pitch_predictor_layers=1,
        energy_predictor_layers=1,
        duration_predictor_layers=1,
        num_mel_bins=8,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = hidden_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.encoder_linear_units = encoder_linear_units
        self.decoder_linear_units = decoder_linear_units
        self.speech_decoder_postnet_units = speech_decoder_postnet_units
        self.speech_decoder_postnet_layers = speech_decoder_postnet_layers
        self.pitch_predictor_layers = pitch_predictor_layers
        self.energy_predictor_layers = energy_predictor_layers
        self.duration_predictor_layers = duration_predictor_layers
        self.num_mel_bins = num_mel_bins

    def prepare_config_and_inputs(self):
        config = self.get_config()
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        return config, input_ids

    def get_config(self):
        return FastSpeech2ConformerConfig(
            hidden_size=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_linear_units=self.encoder_linear_units,
            decoder_linear_units=self.decoder_linear_units,
            speech_decoder_postnet_units=self.speech_decoder_postnet_units,
            speech_decoder_postnet_layers=self.speech_decoder_postnet_layers,
            num_mel_bins=self.num_mel_bins,
            pitch_predictor_layers=self.pitch_predictor_layers,
            energy_predictor_layers=self.energy_predictor_layers,
            duration_predictor_layers=self.duration_predictor_layers,
        )


model_tester = FastSpeech2ConformerModelTester()
config, input_ids = model_tester.prepare_config_and_inputs()


FASTSPEECH2_CONFORMER_CASES = [
    [
        "FastSpeech2ConformerModel",
        "transformers.FastSpeech2ConformerModel",
        "mindone.transformers.FastSpeech2ConformerModel",
        (config,),
        {},
        (input_ids,),
        {
            "return_dict": True,
        },
        {
            "spectrogram": 0,
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
        for case in FASTSPEECH2_CONFORMER_CASES
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
