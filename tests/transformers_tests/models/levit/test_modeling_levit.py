"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/levit/test_modeling_levit.py."""

import logging

import numpy as np
import pytest
import torch
from transformers import LevitConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

# -------------------------------------------------------------
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

# fp16 NaN
DTYPE_AND_THRESHOLDS = {"fp32": 1e-3, "bf16": 5e-2}
MODES = [1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LevitModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=64,
        num_channels=3,
        kernel_size=3,
        stride=2,
        padding=1,
        patch_size=16,
        hidden_sizes=[16, 32, 48],
        num_attention_heads=[1, 2, 3],
        depths=[2, 3, 4],
        key_dim=[8, 8, 8],
        drop_path_rate=0,
        mlp_ratio=[2, 2, 2],
        attention_ratio=[2, 2, 2],
        initializer_range=0.02,
        is_training=True,
        use_labels=True,
        num_labels=2,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.hidden_sizes = hidden_sizes
        self.num_attention_heads = num_attention_heads
        self.depths = depths
        self.key_dim = key_dim
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.attention_ratio = attention_ratio
        self.mlp_ratio = mlp_ratio
        self.initializer_range = initializer_range
        self.down_ops = [
            ["Subsample", key_dim[0], hidden_sizes[0] // key_dim[0], 4, 2, 2],
            ["Subsample", key_dim[0], hidden_sizes[1] // key_dim[0], 4, 2, 2],
        ]
        self.is_training = is_training
        self.use_labels = use_labels
        self.num_labels = num_labels
        self.initializer_range = initializer_range

    def prepare_config_and_inputs(self):
        # Generate pixel values (B, C, H, W) as numpy float arrays
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            # Generate labels if needed for classification task
            labels = ids_numpy([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Create LevitConfig using parameters from __init__
        return LevitConfig(
            image_size=self.image_size,
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            patch_size=self.patch_size,
            hidden_sizes=self.hidden_sizes,
            num_attention_heads=self.num_attention_heads,
            depths=self.depths,
            key_dim=self.key_dim,
            drop_path_rate=self.drop_path_rate,
            mlp_ratio=self.mlp_ratio,
            attention_ratio=self.attention_ratio,
            initializer_range=self.initializer_range,
            down_ops=self.down_ops,
        )


model_tester = LevitModelTester()
config, pixel_values, labels = model_tester.prepare_config_and_inputs()

LEVIT_CASES = [
    [
        "LevitModel",
        "transformers.LevitModel",
        "mindone.transformers.LevitModel",
        (config,),
        {},
        (pixel_values,),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "LevitForImageClassification",
        "transformers.LevitForImageClassification",
        "mindone.transformers.LevitForImageClassification",
        (config,),
        {},
        (pixel_values,),
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
        for case in LEVIT_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_levit_modules_comparison(
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
    """
    Compares the forward pass outputs of PyTorch and MindSpore Levit models.
    """
    ms.set_context(mode=mode)
    threshold = DTYPE_AND_THRESHOLDS[dtype]

    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    pt_model.eval()
    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)

    # MindSpore
    ms_model.set_train(False)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    pt_outputs_to_compare = []
    ms_outputs_to_compare = []

    for pt_key, ms_key in outputs_map.items():
        if pt_key not in pt_outputs.__dict__:
            raise AttributeError(f"Output key '{pt_key}' not in PyTorch output object {type(pt_outputs)}.")
        if ms_key not in ms_outputs.__dict__:
            raise IndexError(f"Output index {ms_key} not in MindSpore output object {type(ms_outputs)}.")

        pt_output = getattr(pt_outputs, pt_key)
        ms_output = getattr(ms_outputs, ms_key)

        pt_outputs_to_compare.append(pt_output)
        ms_outputs_to_compare.append(ms_output)

    # Compute differences between the aligned lists
    diffs = compute_diffs(pt_outputs_to_compare, ms_outputs_to_compare)

    logger.info(f"Computed Differences: {diffs}")

    # --- Assertion ---
    assert (np.array(diffs) < threshold).all(), (
        f"Test Failed for {name} (Mode: {mode}, DType: {dtype})\n"
        f"MindSpore dtype: {ms_dtype}, PyTorch dtype: {pt_dtype}\n"
        f"Outputs differences {np.array(diffs).tolist()} exceeded threshold {threshold}"
    )
    logger.info(f"--- Test Passed: {name} | Mode: {mode} | DType: {dtype} ---")
