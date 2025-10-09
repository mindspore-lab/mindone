"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/qwen3/test_modeling_qwen3.py."""

# tests/models/llama/test_modeling_llama.py
import inspect
from typing import List

import numpy as np
import pytest
import torch
from transformers.models.superglue.configuration_superglue import SuperGlueConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

# nn.function.grid_sample not support fp16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "bf16": 5e-2}
MODES = [1]


class SuperGlueModelTester:
    config_class = SuperGlueConfig

    def __init__(
        self,
        batch_size=2,
        image_width=80,
        image_height=60,
        keypoint_detector_config=None,
        hidden_size: int = 64,
        keypoint_encoder_sizes: List[int] = [32, 64],
        gnn_layers_types: List[str] = ["self", "cross"] * 2,
        num_attention_heads: int = 4,
        sinkhorn_iterations: int = 100,
        matching_threshold: float = 0.2,
    ):
        if keypoint_detector_config is None:
            keypoint_detector_config = {
                "encoder_hidden_sizes": [32, 64],
                "decoder_hidden_size": 64,
                "keypoint_decoder_dim": 65,
                "descriptor_decoder_dim": 64,
                "keypoint_threshold": 0.005,
                "max_keypoints": 256,
                "nms_radius": 4,
                "border_removal_distance": 4,
            }
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height

        self.keypoint_detector_config = keypoint_detector_config
        self.hidden_size = hidden_size
        self.keypoint_encoder_sizes = keypoint_encoder_sizes
        self.gnn_layers_types = gnn_layers_types
        self.num_attention_heads = num_attention_heads
        self.sinkhorn_iterations = sinkhorn_iterations
        self.matching_threshold = matching_threshold

    def prepare_config_and_inputs(self):
        # SuperGlue expects a grayscale image as input
        pixel_values = floats_numpy([self.batch_size, 2, 3, self.image_height, self.image_width])
        config = self.get_config()
        return config, pixel_values

    def get_config(self):
        return SuperGlueConfig(
            keypoint_detector_config=self.keypoint_detector_config,
            hidden_size=self.hidden_size,
            keypoint_encoder_sizes=self.keypoint_encoder_sizes,
            gnn_layers_types=self.gnn_layers_types,
            num_attention_heads=self.num_attention_heads,
            sinkhorn_iterations=self.sinkhorn_iterations,
            matching_threshold=self.matching_threshold,
        )


model_tester = SuperGlueModelTester()
config, pixel_values = model_tester.prepare_config_and_inputs()

LLAMA_CASES = [
    [
        "SuperGlueModel",
        "transformers.SuperGlueForKeypointMatching",
        "mindone.transformers.SuperGlueForKeypointMatching",
        (config,),
        {},
        (pixel_values,),
        {},
        {
            "keypoints": 2,
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
        for case in LLAMA_CASES
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

    if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
        pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
        ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

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
