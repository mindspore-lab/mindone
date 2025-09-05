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
from transformers import UperNetConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [1]  # 1: pynative mode, UperNet doesn't support graph mode


class UperNetModelTester:
    config_class = UperNetConfig

    def __init__(
        self,
        batch_size=2,
        num_channels=3,
        height=32,
        width=32,
        num_labels=3,
        is_training=False,
        use_labels=True,
        # config
        hidden_size=32,
        initializer_range=0.02,
        pool_scales=[1, 2, 3],
        use_auxiliary_head=True,
        auxiliary_loss_weight=0.4,
        auxiliary_in_channels=24,
        auxiliary_channels=16,
        auxiliary_num_convs=1,
        auxiliary_concat_input=False,
        loss_ignore_index=255,
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.num_labels = num_labels
        self.is_training = is_training
        self.use_labels = use_labels

        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.pool_scales = pool_scales
        self.use_auxiliary_head = use_auxiliary_head
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.auxiliary_in_channels = auxiliary_in_channels
        self.auxiliary_channels = auxiliary_channels
        self.auxiliary_num_convs = auxiliary_num_convs
        self.auxiliary_concat_input = auxiliary_concat_input
        self.loss_ignore_index = loss_ignore_index

    def prepare_config_and_inputs(self):
        pixel_values = ids_numpy([self.batch_size, self.num_channels, self.height, self.width], 256)

        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size, self.height, self.width], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        # Use ConvNext as backbone for testing (similar to ResNet but available in MindSpore)
        from mindone.transformers.models.convnext import ConvNextConfig
        backbone_config = ConvNextConfig(
            num_channels=self.num_channels,
            hidden_sizes=[16, 32, 64, 128],
            depths=[1, 1, 1, 1],
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
        
        return self.config_class(
            backbone_config=backbone_config,
            hidden_size=self.hidden_size,
            initializer_range=self.initializer_range,
            pool_scales=self.pool_scales,
            use_auxiliary_head=self.use_auxiliary_head,
            auxiliary_loss_weight=self.auxiliary_loss_weight,
            auxiliary_in_channels=self.auxiliary_in_channels,
            auxiliary_channels=self.auxiliary_channels,
            auxiliary_num_convs=self.auxiliary_num_convs,
            auxiliary_concat_input=self.auxiliary_concat_input,
            loss_ignore_index=self.loss_ignore_index,
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