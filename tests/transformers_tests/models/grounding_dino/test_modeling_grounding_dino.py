"""Test script for Grounding DINO model."""

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
from transformers import BertConfig, GroundingDinoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
# MODES = [0, 1] # 0: graph mode, 1: pynative mode
# FIXME: Grounding DINO does not support graph mode yet, so we only test in pynative mode.
MODES = [1]


class GroundingDinoModelTester:
    config_class = GroundingDinoConfig

    def __init__(
        self,
        batch_size=2,
        num_queries=2,
        seq_length=16,
        image_size=224,
        num_channels=3,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        num_labels=3,
        # config parameters
        text_config=None,
        encoder_layers=1,
        encoder_ffn_dim=24,
        encoder_attention_heads=4,
        decoder_layers=1,
        decoder_ffn_dim=24,
        decoder_attention_heads=4,
        is_encoder_decoder=True,
        activation_function="relu",
        d_model=256,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        auxiliary_loss=False,
        position_embedding_type="sine",
        num_feature_levels=4,
        encoder_n_points=1,
        decoder_n_points=1,
        two_stage=True,
        class_cost=1.0,
        bbox_cost=5.0,
        giou_cost=2.0,
        bbox_loss_coefficient=5.0,
        giou_loss_coefficient=2.0,
        focal_alpha=0.25,
        disable_custom_kernels=False,
        # other parameters
        max_text_len=256,
        text_enhancer_dropout=0.0,
        fusion_droppath=0.1,
        fusion_dropout=0.0,
        embedding_init_target=True,
        query_dim=4,
        decoder_bbox_embed_share=True,
        two_stage_bbox_embed_share=False,
        positional_embedding_temperature=20,
        init_std=0.02,
        layer_norm_eps=1e-5,
        # text config parameters
        text_vocab_size=1000,
        text_hidden_size=128,
        text_num_hidden_layers=2,
        text_num_attention_heads=4,
        text_intermediate_size=256,
        text_max_position_embeddings=64,
    ):
        self.batch_size = batch_size
        self.num_queries = num_queries
        self.seq_length = seq_length
        self.image_size = image_size
        self.num_channels = num_channels
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.num_labels = num_labels

        # config parameters
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_attention_heads = decoder_attention_heads
        self.is_encoder_decoder = is_encoder_decoder
        self.activation_function = activation_function
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.auxiliary_loss = auxiliary_loss
        self.position_embedding_type = position_embedding_type
        self.num_feature_levels = num_feature_levels
        self.encoder_n_points = encoder_n_points
        self.decoder_n_points = decoder_n_points
        self.two_stage = two_stage
        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        self.bbox_loss_coefficient = bbox_loss_coefficient
        self.giou_loss_coefficient = giou_loss_coefficient
        self.focal_alpha = focal_alpha
        self.disable_custom_kernels = disable_custom_kernels
        self.max_text_len = max_text_len
        self.text_enhancer_dropout = text_enhancer_dropout
        self.fusion_droppath = fusion_droppath
        self.fusion_dropout = fusion_dropout
        self.embedding_init_target = embedding_init_target
        self.query_dim = query_dim
        self.decoder_bbox_embed_share = decoder_bbox_embed_share
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.positional_embedding_temperature = positional_embedding_temperature
        self.init_std = init_std
        self.layer_norm_eps = layer_norm_eps
        self.text_config = text_config
        self.max_text_len = max_text_len

        # Text Enhancer
        self.text_enhancer_dropout = text_enhancer_dropout
        # Fusion
        self.fusion_droppath = fusion_droppath
        self.fusion_dropout = fusion_dropout
        # Others
        self.embedding_init_target = embedding_init_target
        self.query_dim = query_dim
        self.decoder_bbox_embed_share = decoder_bbox_embed_share
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        if two_stage_bbox_embed_share and not decoder_bbox_embed_share:
            raise ValueError("If two_stage_bbox_embed_share is True, decoder_bbox_embed_share must be True.")
        self.positional_embedding_temperature = positional_embedding_temperature
        self.init_std = init_std
        self.layer_norm_eps = layer_norm_eps

        self.text_vocab_size = text_vocab_size
        self.text_hidden_size = text_hidden_size
        self.text_num_hidden_layers = text_num_hidden_layers
        self.text_num_attention_heads = text_num_attention_heads
        self.text_intermediate_size = text_intermediate_size
        self.text_max_position_embeddings = text_max_position_embeddings

    def prepare_config_and_inputs(self):
        pixel_values = np.random.randn(self.batch_size, self.num_channels, self.image_size, self.image_size)
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.text_vocab_size)

        pixel_mask = None
        input_mask = None
        if self.use_input_mask:
            pixel_mask = np.ones((self.batch_size, self.image_size, self.image_size))
            input_mask = np.ones_like(input_ids)

        labels = None
        if self.use_labels:
            # Create dummy labels for object detection
            labels = []
            for _ in range(self.batch_size):
                num_targets = np.random.randint(1, 4)  # 1-3 targets per image
                label_dict = {
                    "class_labels": np.random.randint(0, self.num_labels, size=(num_targets,)),
                    "boxes": np.random.rand(num_targets, 4),  # normalized boxes [x, y, w, h]
                }
                labels.append(label_dict)

        config = self.get_config()

        return config, pixel_values, input_ids, pixel_mask, input_mask, labels

    def get_config(self):
        # Create text config
        self.text_config = BertConfig(
            vocab_size=self.text_vocab_size,
            hidden_size=self.text_hidden_size,
            num_hidden_layers=self.text_num_hidden_layers,
            num_attention_heads=self.text_num_attention_heads,
            intermediate_size=self.text_intermediate_size,
            max_position_embeddings=self.text_max_position_embeddings,
        )

        # Create simple backbone config for testing
        self.backbone_config = CONFIG_MAPPING["swin"](
            window_size=7,
            image_size=224,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            out_indices=[2, 3, 4],
        )
        return self.config_class(
            backbone_config=self.backbone_config,
            text_config=self.text_config,
            num_queries=self.num_queries,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_attention_heads=self.decoder_attention_heads,
            is_encoder_decoder=self.is_encoder_decoder,
            activation_function=self.activation_function,
            d_model=self.d_model,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            auxiliary_loss=self.auxiliary_loss,
            position_embedding_type=self.position_embedding_type,
            num_feature_levels=self.num_feature_levels,
            encoder_n_points=self.encoder_n_points,
            decoder_n_points=self.decoder_n_points,
            two_stage=self.two_stage,
            class_cost=self.class_cost,
            bbox_cost=self.bbox_cost,
            giou_cost=self.giou_cost,
            bbox_loss_coefficient=self.bbox_loss_coefficient,
            giou_loss_coefficient=self.giou_loss_coefficient,
            focal_alpha=self.focal_alpha,
            disable_custom_kernels=self.disable_custom_kernels,
            max_text_len=self.max_text_len,
            text_enhancer_dropout=self.text_enhancer_dropout,
            fusion_droppath=self.fusion_droppath,
            fusion_dropout=self.fusion_dropout,
            embedding_init_target=self.embedding_init_target,
            query_dim=self.query_dim,
            decoder_bbox_embed_share=self.decoder_bbox_embed_share,
            two_stage_bbox_embed_share=self.two_stage_bbox_embed_share,
            positional_embedding_temperature=self.positional_embedding_temperature,
            init_std=self.init_std,
            layer_norm_eps=self.layer_norm_eps,
        )


model_tester = GroundingDinoModelTester()
(
    config,
    pixel_values,
    input_ids,
    pixel_mask,
    input_mask,
    labels,
) = model_tester.prepare_config_and_inputs()


GROUNDING_DINO_CASES = [
    [
        "GroundingDinoModel",
        "transformers.GroundingDinoModel",
        "mindone.transformers.GroundingDinoModel",
        (config,),
        {},
        (pixel_values, input_ids),
        {
            "pixel_mask": pixel_mask,
            "attention_mask": input_mask,
        },
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
        for case in GROUNDING_DINO_CASES
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
