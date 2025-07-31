"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/ijepa/test_modeling_ijepa.py."""

import logging

import numpy as np
import pytest
import torch
from transformers import IJepaConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

# -------------------------------------------------------------
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 1e-2}
MODES = [1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IJepaModelTester:
    def __init__(
        self,
        batch_size=13,
        image_size=30,
        patch_size=2,
        num_channels=3,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        type_sequence_label_size=10,
        initializer_range=0.02,
        scope=None,
        encoder_stride=2,
        mask_ratio=0.5,
        attn_implementation="eager",
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.scope = scope
        self.encoder_stride = encoder_stride
        self.attn_implementation = attn_implementation

        # in IJEPA, the seq length equals the number of patches (we don't add 1 for the [CLS] token)
        num_patches = (image_size // patch_size) ** 2
        self.seq_length = num_patches
        self.mask_ratio = mask_ratio
        self.num_masks = int(mask_ratio * self.seq_length)
        self.mask_length = num_patches

    def prepare_config_and_inputs(self):
        # Generate pixel values (B, C, H, W) as numpy float arrays
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            # Generate labels if needed for classification task
            labels = ids_numpy([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Create LevitConfig using parameters from __init__
        return IJepaConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            is_decoder=False,
            initializer_range=self.initializer_range,
            encoder_stride=self.encoder_stride,
            attn_implementation=self.attn_implementation,
        )


model_tester = IJepaModelTester()
config, pixel_values, labels = model_tester.prepare_config_and_inputs()

IJEPA_CASES = [
    [
        "IJepaModel",
        "transformers.IJepaModel",
        "mindone.transformers.IJepaModel",
        (config,),
        {},
        (pixel_values,),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "IJepaForImageClassification",
        "transformers.IJepaForImageClassification",
        "mindone.transformers.IJepaForImageClassification",
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
        for case in IJEPA_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
        for mode in MODES
    ],
)
def test_ijepa_modules_comparison(
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
