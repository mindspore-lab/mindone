"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/upernet/test_modeling_upernet.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.

import inspect

import numpy as np
import pytest
import torch
from transformers import ConvNextConfig, UperNetConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # 1: pynative mode, UperNet doesn't support graph mode


class UperNetModelTester:
    config_class = UperNetConfig

    def __init__(
        self,
        batch_size=13,
        image_size=32,
        num_channels=3,
        num_stages=4,
        hidden_sizes=[10, 20, 30, 40],
        depths=[1, 1, 1, 1],
        is_training=True,
        use_labels=True,
        intermediate_size=37,
        hidden_act="gelu",
        type_sequence_label_size=10,
        initializer_range=0.02,
        out_features=["stage2", "stage3", "stage4"],
        num_labels=3,
        scope=None,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.num_stages = num_stages
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.out_features = out_features
        self.num_labels = num_labels
        self.scope = scope

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size], self.type_sequence_label_size)

        config = self.get_config()

        return config, pixel_values, labels

    def get_backbone_config(self):
        return ConvNextConfig(
            num_channels=self.num_channels,
            num_stages=self.num_stages,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            is_training=self.is_training,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            out_features=self.out_features,
        )

    def get_config(self):
        return UperNetConfig(
            backbone_config=self.get_backbone_config(),
            backbone=None,
            hidden_size=64,
            pool_scales=[1, 2, 3, 6],
            use_auxiliary_head=True,
            auxiliary_loss_weight=0.4,
            auxiliary_in_channels=40,
            auxiliary_channels=32,
            auxiliary_num_convs=1,
            auxiliary_concat_input=False,
            loss_ignore_index=255,
            num_labels=self.num_labels,
        )


model_tester = UperNetModelTester()
config, pixel_values, labels = model_tester.prepare_config_and_inputs()


UPERNET_CASES = [
    [
        "UperNetForSemanticSegmentation",
        "transformers.UperNetForSemanticSegmentation",
        "mindone.transformers.UperNetForSemanticSegmentation",
        (config,),
        {},
        (pixel_values,),
        {"labels": labels} if labels is not None else {},
        {
            "logits": 0,
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
        for case in UPERNET_CASES
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
        pt_outputs = pt_model(*pt_inputs_args)
    ms_outputs = ms_model(*ms_inputs_args)

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
