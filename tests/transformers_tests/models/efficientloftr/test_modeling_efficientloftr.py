# Copyright 2025 The HuggingFace Team. All rights reserved.
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
from transformers import EfficientLoFTRConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3}
# MaxPool2d doesn't support bf16.
MODES = [1]


class EfficientLoFTRModelTester:
    def __init__(
        self,
        batch_size=2,
        image_width=80,
        image_height=60,
        stage_num_blocks: list[int] = [1, 1, 1],
        out_features: list[int] = [32, 32, 64],
        stage_stride: list[int] = [2, 1, 2],
        q_aggregation_kernel_size: int = 1,
        kv_aggregation_kernel_size: int = 1,
        q_aggregation_stride: int = 1,
        kv_aggregation_stride: int = 1,
        num_attention_layers: int = 2,
        num_attention_heads: int = 8,
        hidden_size: int = 64,
        coarse_matching_threshold: float = 0.0,
        fine_kernel_size: int = 2,
        coarse_matching_border_removal: int = 0,
    ):
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.stage_num_blocks = stage_num_blocks
        self.out_features = out_features
        self.stage_stride = stage_stride
        self.q_aggregation_kernel_size = q_aggregation_kernel_size
        self.kv_aggregation_kernel_size = kv_aggregation_kernel_size
        self.q_aggregation_stride = q_aggregation_stride
        self.kv_aggregation_stride = kv_aggregation_stride
        self.num_attention_layers = num_attention_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.coarse_matching_threshold = coarse_matching_threshold
        self.coarse_matching_border_removal = coarse_matching_border_removal
        self.fine_kernel_size = fine_kernel_size

    def prepare_config_and_inputs(self):
        # EfficientLoFTR expects a grayscale image as input
        pixel_values = floats_numpy([self.batch_size, 2, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return EfficientLoFTRConfig(
            stage_num_blocks=self.stage_num_blocks,
            out_features=self.out_features,
            stage_stride=self.stage_stride,
            q_aggregation_kernel_size=self.q_aggregation_kernel_size,
            kv_aggregation_kernel_size=self.kv_aggregation_kernel_size,
            q_aggregation_stride=self.q_aggregation_stride,
            kv_aggregation_stride=self.kv_aggregation_stride,
            num_attention_layers=self.num_attention_layers,
            num_attention_heads=self.num_attention_heads,
            hidden_size=self.hidden_size,
            coarse_matching_threshold=self.coarse_matching_threshold,
            coarse_matching_border_removal=self.coarse_matching_border_removal,
            fine_kernel_size=self.fine_kernel_size,
        )


model_tester = EfficientLoFTRModelTester()
config, pixel_values = model_tester.prepare_config_and_inputs()
EFFICIENTLOFTR_CASES = [
    [
        "EfficientLoFTRModel",
        "transformers.EfficientLoFTRModel",
        "mindone.transformers.EfficientLoFTRModel",
        (config,),
        {},
        (pixel_values,),
        {},
        {},
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
        for case in EFFICIENTLOFTR_CASES
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
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)["feature_maps"][-1]
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)["feature_maps"][-1]
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
