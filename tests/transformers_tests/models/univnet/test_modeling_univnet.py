"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/univnet/test_modeling_univnet.py."""

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
from transformers import UnivNetConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

# in fp16 condition/torch 2.4.0, "torch.nn.functional.pad"'s reflection_mode is not supported in cpu
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "bf16": 6e-3}
MODES = [1]


class UnivNetModelTester:
    def __init__(
        self,
        batch_size=2,
        seq_length=7,
        in_channels=8,
        hidden_channels=8,
        num_mel_bins=20,
        kernel_predictor_hidden_channels=8,
        seed=0,
        is_training=False,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_mel_bins = num_mel_bins
        self.kernel_predictor_hidden_channels = kernel_predictor_hidden_channels
        self.seed = seed
        self.is_training = is_training

    def prepare_noise_sequence(self):
        noise_shape = (self.batch_size, self.seq_length, self.in_channels)
        # Create noise on CPU for reproducibility
        noise_sequence = floats_numpy(noise_shape)
        return noise_sequence

    def prepare_config_and_inputs(self):
        spectrogram = floats_numpy([self.batch_size, self.seq_length, self.num_mel_bins], scale=1.0)
        noise_sequence = self.prepare_noise_sequence()
        config = self.get_config()
        return config, spectrogram, noise_sequence

    def get_config(self):
        return UnivNetConfig(
            model_in_channels=self.in_channels,
            model_hidden_channels=self.hidden_channels,
            num_mel_bins=self.num_mel_bins,
            kernel_predictor_hidden_channels=self.kernel_predictor_hidden_channels,
        )


model_tester = UnivNetModelTester()
config, spectrogram, noise_sequence = model_tester.prepare_config_and_inputs()


UNIVNET_CASES = [
    [
        "UnivNetModel",
        "transformers.UnivNetModel",
        "mindone.transformers.UnivNetModel",
        (config,),
        {},
        (spectrogram, noise_sequence),
        {},
        {
            "waveforms": "waveforms",
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
        for case in UNIVNET_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
# FIXME there is "core dump" error if running all model ut. We have not figured out, so skip this ut firstly.
@pytest.mark.skip
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
