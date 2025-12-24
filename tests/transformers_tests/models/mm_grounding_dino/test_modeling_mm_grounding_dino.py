"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/mobilebert/test_modeling_mobilebert.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import copy
import inspect
import math

import numpy as np
import pytest
import torch
from transformers import MMGroundingDinoConfig, SwinConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy

# CrossEntropyLoss not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


# Copied from tests.models.grounding_dino.test_modeling_grounding_dino.generate_fake_bounding_boxes
def generate_fake_bounding_boxes(n_boxes):
    """Generate bounding boxes in the format (center_x, center_y, width, height) using NumPy"""
    if not isinstance(n_boxes, int):
        raise TypeError("n_boxes must be an integer")
    if n_boxes <= 0:
        raise ValueError("n_boxes must be a positive integer")

    bounding_boxes = np.random.rand(n_boxes, 4)

    center_x = bounding_boxes[:, 0]
    center_y = bounding_boxes[:, 1]
    width = bounding_boxes[:, 2]
    height = bounding_boxes[:, 3]

    width = np.minimum(width, 1.0)
    height = np.minimum(height, 1.0)

    center_x = np.where(center_x - width / 2 < 0, width / 2, center_x)
    center_x = np.where(center_x + width / 2 > 1, 1 - width / 2, center_x)
    center_y = np.where(center_y - height / 2 < 0, height / 2, center_y)
    center_y = np.where(center_y + height / 2 > 1, 1 - height / 2, center_y)

    bounding_boxes = np.stack([center_x, center_y, width, height], axis=1)

    return bounding_boxes


# Copied from tests.models.grounding_dino.test_modeling_grounding_dino.GroundingDinoModelTester with GroundingDino->MMGroundingDino
class MMGroundingDinoModelTester:
    def __init__(
        self,
        batch_size=4,
        is_training=True,
        use_labels=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        num_queries=2,
        num_channels=3,
        image_size=98,
        n_targets=8,
        num_labels=2,
        num_feature_levels=4,
        encoder_n_points=2,
        decoder_n_points=6,
        max_text_len=7,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.num_queries = num_queries
        self.num_channels = num_channels
        self.image_size = image_size
        self.n_targets = n_targets
        self.num_labels = num_labels
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.max_text_len = max_text_len

        # we also set the expected seq length for both encoder and decoder
        self.encoder_seq_length_vision = (
            math.ceil(self.image_size / 8) ** 2
            + math.ceil(self.image_size / 16) ** 2
            + math.ceil(self.image_size / 32) ** 2
            + math.ceil(self.image_size / 64) ** 2
        )

        self.encoder_seq_length_text = self.max_text_len

        self.decoder_seq_length = self.num_queries

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        pixel_mask = np.ones([self.batch_size, self.image_size, self.image_size])

        # When using `MMGroundingDino` the text input template is '{label1}. {label2}. {label3. ... {labelN}.'
        # Therefore to avoid errors when running tests with `labels` `input_ids` have to follow this structure.
        # Otherwise when running `build_label_maps` it will throw an error when trying to split the input_ids into segments.
        input_ids = np.array([101, 3869, 1012, 11420, 3869, 1012, 102])
        input_ids = np.tile(input_ids[None, :], (self.batch_size, 1))

        labels = None
        if self.use_labels:
            # labels is a list of Dict (each Dict being the labels for a given example in the batch)
            labels = []
            for i in range(self.batch_size):
                target = {}
                target["class_labels"] = torch.randint(high=self.num_labels, size=(self.n_targets,))
                target["boxes"] = generate_fake_bounding_boxes(self.n_targets)
                target["masks"] = torch.rand(self.n_targets, self.image_size, self.image_size)
                labels.append(target)

        config = self.get_config()
        return config, pixel_values, pixel_mask, input_ids, labels

    def get_config(self):
        swin_config = SwinConfig(
            window_size=7,
            embed_dim=8,
            depths=[1, 1, 1, 1],
            num_heads=[1, 1, 1, 1],
            image_size=self.image_size,
            out_features=["stage2", "stage3", "stage4"],
            out_indices=[2, 3, 4],
        )
        text_backbone = {
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 8,
            "max_position_embeddings": 8,
            "model_type": "bert",
        }
        return MMGroundingDinoConfig(
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            num_queries=self.num_queries,
            num_labels=self.num_labels,
            num_feature_levels=self.num_feature_levels,
            encoder_n_points=self.encoder_n_points,
            decoder_n_points=self.decoder_n_points,
            use_timm_backbone=False,
            backbone_config=swin_config,
            max_text_len=self.max_text_len,
            text_config=text_backbone,
        )

    def prepare_config_and_inputs_for_common(self):
        config, pixel_values, pixel_mask, input_ids, labels = self.prepare_config_and_inputs()
        inputs_dict = {"pixel_values": pixel_values, "pixel_mask": pixel_mask, "input_ids": input_ids}
        return config, inputs_dict


model_tester = MMGroundingDinoModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()
config_has_num_labels = copy.deepcopy(config)
config_has_num_labels.num_labels = model_tester.num_labels

MMGROUNDINGDINO_CASES = [
    [
        "MMGroundingDinoModel",
        "transformers.MMGroundingDinoModel",
        "mindone.transformers.MMGroundingDinoModel",
        (config,),
        {},
        (),
        {**inputs_dict},
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
        for case in MMGROUNDINGDINO_CASES
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
