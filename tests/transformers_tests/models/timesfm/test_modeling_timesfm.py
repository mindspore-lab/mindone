# coding=utf-8
# Copyright 2025 Google LLC and HuggingFace Inc. team.
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

import inspect

import numpy as np
import pytest
import torch
from transformers import TimesFmConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class TimesFmModelTester:
    def __init__(
        self,
        patch_length: int = 32,
        context_length: int = 512,
        horizon_length: int = 128,
        freq_size: int = 3,
        num_hidden_layers: int = 1,
        hidden_size: int = 16,
        intermediate_size: int = 32,
        head_dim: int = 8,
        num_heads: int = 2,
        tolerance: float = 1e-6,
        rms_norm_eps: float = 1e-6,
        quantiles: list[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        pad_val: float = 1123581321.0,
        use_positional_embedding: bool = True,
        initializer_factor: float = 0.0,
        is_training: bool = False,
        batch_size: int = 3,
    ):
        self.patch_length = patch_length
        self.context_length = context_length
        self.horizon_length = horizon_length
        self.quantiles = quantiles
        self.pad_val = pad_val
        self.freq_size = freq_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_heads
        self.tolerance = tolerance
        self.rms_norm_eps = rms_norm_eps
        self.use_positional_embedding = use_positional_embedding
        self.initializer_factor = initializer_factor
        self.is_training = is_training
        self.batch_size = batch_size

        # The size of test input
        self.seq_length = context_length // patch_length
        self.hidden_size = hidden_size

    def get_config(self):
        return TimesFmConfig(
            patch_length=self.patch_length,
            context_length=self.context_length,
            horizon_length=self.horizon_length,
            quantiles=self.quantiles,
            pad_val=self.pad_val,
            freq_size=self.freq_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            head_dim=self.head_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            tolerance=self.tolerance,
            rms_norm_eps=self.rms_norm_eps,
            use_positional_embedding=self.use_positional_embedding,
            initializer_factor=self.initializer_factor,
        )

    def get_pipeline_config(self):
        return self.get_config()

    def prepare_config_and_inputs(self):
        forecast_input = [
            np.sin(np.linspace(0, 20, 100)).astype(np.float32),
            np.cos(np.linspace(0, 20, 100)).astype(np.float32),
            np.tan(np.linspace(0, 20, 100)).astype(np.float32),
        ]
        frequency_input = np.array([0, 1, 2], dtype=np.int64)

        return (self.get_config(), np.stack(forecast_input, axis=0), frequency_input)


model_tester = TimesFmModelTester()
config, past_values, freq = model_tester.prepare_config_and_inputs()
TIMESFM_CASES = [
    [
        "TimesFmModelForPrediction",
        "transformers.TimesFmModelForPrediction",
        "mindone.transformers.TimesFmModelForPrediction",
        (config,),
        {},
        (past_values,),
        {
            "freq": freq,
        },
        {
            "last_hidden_state": 0,
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
        for case in TIMESFM_CASES
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
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs, return_dict=False)
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
