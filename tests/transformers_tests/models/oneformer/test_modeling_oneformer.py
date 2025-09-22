"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/oneformer/test_modeling_oneformer.py."""

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
from transformers import OneFormerConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 9e-3}
# MODES = [0, 1] # 0: graph mode, 1: pynative mode
# FIXME: OneFormer does not support graph mode yet, so we only test in pynative mode.
MODES = [1]


class OneFormerModelTester:
    config_class = OneFormerConfig

    def __init__(
        self,
        batch_size=2,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        # config
        num_queries=10,
        no_object_weight=0.1,
        class_weight=2.0,
        mask_weight=5.0,
        dice_weight=5.0,
        contrastive_weight=0.5,
        contrastive_temperature=0.07,
        train_num_points=1024,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        init_std=0.02,
        init_xavier_std=1.0,
        layer_norm_eps=1e-05,
        use_auxiliary_loss=True,
        output_auxiliary_logits=True,
        strides=[4, 8, 16, 32],
        task_seq_len=77,
        text_encoder_width=64,
        text_encoder_context_length=77,
        text_encoder_num_layers=2,
        text_encoder_vocab_size=49408,
        text_encoder_proj_layers=2,
        text_encoder_n_ctx=16,
        conv_dim=64,
        mask_dim=64,
        hidden_dim=64,
        encoder_feedforward_dim=256,
        norm="GN",
        encoder_layers=2,
        decoder_layers=2,
        use_task_norm=True,
        num_attention_heads=4,
        dropout=0.1,
        dim_feedforward=512,
        pre_norm=False,
        enforce_input_proj=False,
        query_dec_layers=1,
        common_stride=4,
        # Reduced backbone config for testing
        backbone_config=None,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels

        # Reduced model size for testing
        self.num_queries = num_queries
        self.no_object_weight = no_object_weight
        self.class_weight = class_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.train_num_points = train_num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.init_std = init_std
        self.init_xavier_std = init_xavier_std
        self.layer_norm_eps = layer_norm_eps
        self.use_auxiliary_loss = use_auxiliary_loss
        self.output_auxiliary_logits = output_auxiliary_logits
        self.strides = strides
        self.task_seq_len = task_seq_len
        self.text_encoder_width = text_encoder_width
        self.text_encoder_context_length = text_encoder_context_length
        self.text_encoder_num_layers = text_encoder_num_layers
        self.text_encoder_vocab_size = text_encoder_vocab_size
        self.text_encoder_proj_layers = text_encoder_proj_layers
        self.text_encoder_n_ctx = text_encoder_n_ctx
        self.conv_dim = conv_dim
        self.mask_dim = mask_dim
        self.hidden_dim = hidden_dim
        self.encoder_feedforward_dim = encoder_feedforward_dim
        self.norm = norm
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.use_task_norm = use_task_norm
        self.num_attention_heads = num_attention_heads
        self.dropout = dropout
        self.dim_feedforward = dim_feedforward
        self.pre_norm = pre_norm
        self.enforce_input_proj = enforce_input_proj
        self.query_dec_layers = query_dec_layers
        self.common_stride = common_stride

        # Use a minimal backbone config for testing
        if backbone_config is None:
            # Reduced Swin backbone for testing
            self.backbone_config = {
                "model_type": "swin",
                "image_size": 224,
                "num_channels": 3,
                "patch_size": 4,
                "embed_dim": 32,  # Reduced from 96
                "depths": [1, 1, 2, 1],  # Reduced from [2, 2, 6, 2]
                "num_heads": [1, 2, 4, 8],  # Reduced from [3, 6, 12, 24]
                "window_size": 7,
                "drop_path_rate": 0.1,  # Reduced from 0.3
                "use_absolute_embeddings": False,
                "out_features": ["stage1", "stage2", "stage3", "stage4"],
            }
        else:
            self.backbone_config = backbone_config

    def prepare_config_and_inputs(self):
        # Create minimal test inputs
        pixel_values = np.random.randn(self.batch_size, 3, 224, 224).astype(np.float32)

        # Task input - simplified for testing
        task_inputs = ids_numpy([self.batch_size, self.task_seq_len], self.text_encoder_vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.ones((self.batch_size, 224, 224), dtype=np.int64)

        config = self.get_config()

        return config, pixel_values, task_inputs, input_mask

    def get_config(self):
        return self.config_class(
            backbone_config=self.backbone_config,
            num_queries=self.num_queries,
            no_object_weight=self.no_object_weight,
            class_weight=self.class_weight,
            mask_weight=self.mask_weight,
            dice_weight=self.dice_weight,
            contrastive_weight=self.contrastive_weight,
            contrastive_temperature=self.contrastive_temperature,
            train_num_points=self.train_num_points,
            oversample_ratio=self.oversample_ratio,
            importance_sample_ratio=self.importance_sample_ratio,
            init_std=self.init_std,
            init_xavier_std=self.init_xavier_std,
            layer_norm_eps=self.layer_norm_eps,
            is_training=self.is_training,
            use_auxiliary_loss=self.use_auxiliary_loss,
            output_auxiliary_logits=self.output_auxiliary_logits,
            strides=self.strides,
            task_seq_len=self.task_seq_len,
            text_encoder_width=self.text_encoder_width,
            text_encoder_context_length=self.text_encoder_context_length,
            text_encoder_num_layers=self.text_encoder_num_layers,
            text_encoder_vocab_size=self.text_encoder_vocab_size,
            text_encoder_proj_layers=self.text_encoder_proj_layers,
            text_encoder_n_ctx=self.text_encoder_n_ctx,
            conv_dim=self.conv_dim,
            mask_dim=self.mask_dim,
            hidden_dim=self.hidden_dim,
            encoder_feedforward_dim=self.encoder_feedforward_dim,
            norm=self.norm,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            use_task_norm=self.use_task_norm,
            num_attention_heads=self.num_attention_heads,
            dropout=self.dropout,
            dim_feedforward=self.dim_feedforward,
            pre_norm=self.pre_norm,
            enforce_input_proj=self.enforce_input_proj,
            query_dec_layers=self.query_dec_layers,
            common_stride=self.common_stride,
        )


model_tester = OneFormerModelTester()
(
    config,
    pixel_values,
    task_inputs,
    input_mask,
) = model_tester.prepare_config_and_inputs()


ONEFORMER_CASES = [
    [
        "OneFormerModel",
        "transformers.OneFormerModel",
        "mindone.transformers.OneFormerModel",
        (config,),
        {},
        (pixel_values,),
        {
            "task_inputs": task_inputs,
            "output_hidden_states": True,
        },
        {
            "encoder_hidden_states": 0,
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
        for case in ONEFORMER_CASES
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
