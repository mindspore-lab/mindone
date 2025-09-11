"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/upernet/test_modeling_upernet.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.

import inspect
import math

import numpy as np
import pytest
import torch
from transformers import RTDetrConfig, RTDetrResNetConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy, floats_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # 1: pynative mode, UperNet doesn't support graph mode


class RTDetrModelTester:
    config_class = RTDetrConfig

    def __init__(
        self,
            batch_size=3,
            is_training=True,
            use_labels=True,
            n_targets=3,
            num_labels=10,
            initializer_range=0.02,
            layer_norm_eps=1e-5,
            batch_norm_eps=1e-5,
            # backbone
            backbone_config=None,
            # encoder HybridEncoder
            encoder_hidden_dim=32,
            encoder_in_channels=[128, 256, 512],
            feat_strides=[8, 16, 32],
            encoder_layers=1,
            encoder_ffn_dim=64,
            encoder_attention_heads=2,
            dropout=0.0,
            activation_dropout=0.0,
            encode_proj_layers=[2],
            positional_encoding_temperature=10000,
            encoder_activation_function="gelu",
            activation_function="silu",
            eval_size=None,
            normalize_before=False,
            # decoder RTDetrTransformer
            d_model=32,
            num_queries=30,
            decoder_in_channels=[32, 32, 32],
            decoder_ffn_dim=64,
            num_feature_levels=3,
            decoder_n_points=4,
            decoder_layers=2,
            decoder_attention_heads=2,
            decoder_activation_function="relu",
            attention_dropout=0.0,
            num_denoising=0,
            label_noise_ratio=0.5,
            box_noise_scale=1.0,
            learn_initial_query=False,
            anchor_image_size=None,
            image_size=64,
            disable_custom_kernels=True,
            with_box_refine=True,
    ):
        self.batch_size = batch_size
        self.num_channels = 3
        self.is_training = is_training
        self.use_labels = use_labels
        self.n_targets = n_targets
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_norm_eps = batch_norm_eps
        self.backbone_config = backbone_config
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_in_channels = encoder_in_channels
        self.feat_strides = feat_strides
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.dropout = dropout
        self.activation_dropout = activation_dropout
        self.encode_proj_layers = encode_proj_layers
        self.positional_encoding_temperature = positional_encoding_temperature
        self.encoder_activation_function = encoder_activation_function
        self.activation_function = activation_function
        self.eval_size = eval_size
        self.normalize_before = normalize_before
        self.d_model = d_model
        self.num_queries = num_queries
        self.decoder_in_channels = decoder_in_channels
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_feature_levels = num_feature_levels
        self.decoder_n_points = decoder_n_points
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_activation_function = decoder_activation_function
        self.attention_dropout = attention_dropout
        self.num_denoising = num_denoising
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale
        self.learn_initial_query = learn_initial_query
        self.anchor_image_size = anchor_image_size
        self.image_size = image_size
        self.disable_custom_kernels = disable_custom_kernels
        self.with_box_refine = with_box_refine

        self.encoder_seq_length = math.ceil(self.image_size / 32) * math.ceil(self.image_size / 32)


    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        pixel_mask = np.ones([self.batch_size, self.image_size, self.image_size])

        config = self.get_config()
        config.num_labels = self.num_labels

        return config, pixel_values, pixel_mask


    def get_config(self):
        hidden_sizes = [10, 20, 30, 40]
        backbone_config = RTDetrResNetConfig(
            embeddings_size=10,
            hidden_sizes=hidden_sizes,
            depths=[1, 1, 2, 1],
            out_features=["stage2", "stage3", "stage4"],
            out_indices=[2, 3, 4],
        )
        return RTDetrConfig.from_backbone_configs(
            backbone_config=backbone_config,
            encoder_hidden_dim=self.encoder_hidden_dim,
            encoder_in_channels=hidden_sizes[1:],
            feat_strides=self.feat_strides,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            encode_proj_layers=self.encode_proj_layers,
            positional_encoding_temperature=self.positional_encoding_temperature,
            encoder_activation_function=self.encoder_activation_function,
            activation_function=self.activation_function,
            eval_size=self.eval_size,
            normalize_before=self.normalize_before,
            d_model=self.d_model,
            num_queries=self.num_queries,
            decoder_in_channels=self.decoder_in_channels,
            decoder_ffn_dim=self.decoder_ffn_dim,
            num_feature_levels=self.num_feature_levels,
            decoder_n_points=self.decoder_n_points,
            decoder_layers=self.decoder_layers,
            decoder_attention_heads=self.decoder_attention_heads,
            decoder_activation_function=self.decoder_activation_function,
            attention_dropout=self.attention_dropout,
            num_denoising=self.num_denoising,
            label_noise_ratio=self.label_noise_ratio,
            box_noise_scale=self.box_noise_scale,
            learn_initial_query=self.learn_initial_query,
            anchor_image_size=self.anchor_image_size,
            image_size=self.image_size,
            disable_custom_kernels=self.disable_custom_kernels,
            with_box_refine=self.with_box_refine,
        )


model_tester = RTDetrModelTester()
config, pixel_values, pixel_mask = model_tester.prepare_config_and_inputs()


RTDETR_CASES = [
    [
        "RTDetrModel",
        "transformers.RTDetrModel",
        "mindone.transformers.RTDetrModel",
        (config,),
        {},
        (),
        {'pixel_values':pixel_values, 'pixel_mask':pixel_mask},
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
        for case in RTDETR_CASES
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
