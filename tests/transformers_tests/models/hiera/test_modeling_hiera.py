"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/hiera/test_modeling_hiera.py."""

import logging

import numpy as np
import pytest
import torch
from transformers import HieraConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

# -------------------------------------------------------------
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4}
MODES = [1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HieraModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=[64, 64],
        mlp_ratio=1.0,
        num_channels=3,
        depths=[1, 1, 1, 1],
        patch_stride=[4, 4],
        patch_size=[7, 7],
        patch_padding=[3, 3],
        masked_unit_size=[8, 8],
        num_heads=[1, 1, 1, 1],
        embed_dim_multiplier=2.0,
        is_training=True,
        use_labels=True,
        embed_dim=8,
        hidden_act="gelu",
        decoder_hidden_size=2,
        decoder_depth=1,
        decoder_num_heads=1,
        initializer_range=0.02,
        scope=None,
        type_sequence_label_size=10,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.mlp_ratio = mlp_ratio
        self.num_channels = num_channels
        self.depths = depths
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.patch_padding = patch_padding
        self.masked_unit_size = masked_unit_size
        self.num_heads = num_heads
        self.embed_dim_multiplier = embed_dim_multiplier
        self.is_training = is_training
        self.use_labels = use_labels
        self.embed_dim = embed_dim
        self.hidden_act = hidden_act
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_depth = decoder_depth
        self.decoder_num_heads = decoder_num_heads
        self.initializer_range = initializer_range
        self.scope = scope
        self.type_sequence_label_size = type_sequence_label_size

    def prepare_config_and_inputs(self):
        # Generate pixel values (B, C, H, W) as numpy float arrays
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size[0], self.image_size[1]])

        labels = None
        if self.use_labels:
            # Generate labels if needed for classification task
            labels = ids_numpy([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Create HieraConfig using parameters from __init__
        return HieraConfig(
            embed_dim=self.embed_dim,
            image_size=self.image_size,
            patch_stride=self.patch_stride,
            patch_size=self.patch_size,
            patch_padding=self.patch_padding,
            masked_unit_size=self.masked_unit_size,
            mlp_ratio=self.mlp_ratio,
            num_channels=self.num_channels,
            depths=self.depths,
            num_heads=self.num_heads,
            embed_dim_multiplier=self.embed_dim_multiplier,
            hidden_act=self.hidden_act,
            decoder_hidden_size=self.decoder_hidden_size,
            decoder_depth=self.decoder_depth,
            decoder_num_heads=self.decoder_num_heads,
            initializer_range=self.initializer_range,
        )


model_tester = HieraModelTester()
config, pixel_values, labels = model_tester.prepare_config_and_inputs()

HIERA_CASES = [
    [
        "HieraModel",
        "transformers.HieraModel",
        "mindone.transformers.HieraModel",
        (config,),
        {},
        (pixel_values,),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "HieraForImageClassification_Logits",
        "transformers.HieraForImageClassification",
        "mindone.transformers.HieraForImageClassification",
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
        for case in HIERA_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_hiera_modules_comparison(
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
    Compares the forward pass outputs of PyTorch and MindSpore Hiera models.
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
