# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the DepthPro model."""

import numpy as np
import pytest
import torch
from transformers import DepthProConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 1e-3, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # not support graph mode yet


class DepthProModelTester:
    def __init__(
        self,
        batch_size=8,
        image_size=64,
        patch_size=16,
        num_channels=3,
        is_training=True,
        use_labels=True,
        fusion_hidden_size=16,
        intermediate_hook_ids=[1, 0],
        intermediate_feature_dims=[10, 8],
        scaled_images_ratios=[0.5, 1.0],
        scaled_images_overlap_ratios=[0.0, 0.2],
        scaled_images_feature_dims=[12, 12],
        initializer_range=0.02,
        use_fov_model=False,
        image_model_config={
            "model_type": "dinov2",
            "num_hidden_layers": 2,
            "hidden_size": 16,
            "num_attention_heads": 1,
            "patch_size": 4,
        },
        patch_model_config={
            "model_type": "vit",
            "num_hidden_layers": 2,
            "hidden_size": 24,
            "num_attention_heads": 2,
            "patch_size": 6,
        },
        fov_model_config={
            "model_type": "vit",
            "num_hidden_layers": 2,
            "hidden_size": 32,
            "num_attention_heads": 4,
            "patch_size": 8,
        },
        num_labels=3,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.fusion_hidden_size = fusion_hidden_size
        self.intermediate_hook_ids = intermediate_hook_ids
        self.intermediate_feature_dims = intermediate_feature_dims
        self.scaled_images_ratios = scaled_images_ratios
        self.scaled_images_overlap_ratios = scaled_images_overlap_ratios
        self.scaled_images_feature_dims = scaled_images_feature_dims
        self.initializer_range = initializer_range
        self.use_fov_model = use_fov_model
        self.image_model_config = image_model_config
        self.patch_model_config = patch_model_config
        self.fov_model_config = fov_model_config
        self.num_labels = num_labels

        self.hidden_size = image_model_config["hidden_size"]
        self.num_hidden_layers = image_model_config["num_hidden_layers"]
        self.num_attention_heads = image_model_config["num_attention_heads"]

        # may be different for a backbone other than dinov2
        self.out_size = patch_size // image_model_config["patch_size"]
        self.seq_length = self.out_size**2 + 1  # we add 1 for the [CLS] token

        n_fusion_blocks = len(intermediate_hook_ids) + len(scaled_images_ratios)
        self.expected_depth_size = 2 ** (n_fusion_blocks + 1) * self.out_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size, self.image_size, self.image_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return DepthProConfig(
            patch_size=self.patch_size,
            fusion_hidden_size=self.fusion_hidden_size,
            intermediate_hook_ids=self.intermediate_hook_ids,
            intermediate_feature_dims=self.intermediate_feature_dims,
            scaled_images_ratios=self.scaled_images_ratios,
            scaled_images_overlap_ratios=self.scaled_images_overlap_ratios,
            scaled_images_feature_dims=self.scaled_images_feature_dims,
            initializer_range=self.initializer_range,
            image_model_config=self.image_model_config,
            patch_model_config=self.patch_model_config,
            fov_model_config=self.fov_model_config,
            use_fov_model=self.use_fov_model,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, labels = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


model_tester = DepthProModelTester()
config, pixel_values, labels = model_tester.prepare_config_and_inputs()


_CASES = [
    [
        "DepthProForDepthEstimation",
        "transformers.DepthProForDepthEstimation",
        "mindone.transformers.DepthProForDepthEstimation",
        (config,),
        {},
        (pixel_values,),
        {},
        {
            "predicted_depth": 0,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in _CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
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
