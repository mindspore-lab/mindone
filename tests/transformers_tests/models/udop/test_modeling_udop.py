"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/udop/test_modeling_udop.py."""

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
from transformers import UdopConfig

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
# FIXME: UDOP does not support graph mode yet, so we only test in pynative mode.
MODES = [1]


class UdopModelTester:
    config_class = UdopConfig

    def __init__(
        self,
        batch_size=1,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        # config - reduced values for testing
        vocab_size=99,
        d_model=32,
        d_kv=2,
        d_ff=128,
        num_layers=2,
        num_decoder_layers=None,
        num_heads=16,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        relative_bias_args=[{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}],
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        is_encoder_decoder=True,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=1,
        max_2d_position_embeddings=1024,
        image_size=224,
        patch_size=16,
        num_channels=3,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids

        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.relative_bias_args = relative_bias_args
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.is_encoder_decoder = is_encoder_decoder
        self.use_cache = use_cache
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.ones([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        # Create bbox (bounding box) inputs for UDOP
        bbox = np.random.randint(0, self.max_2d_position_embeddings, size=[self.batch_size, self.seq_length, 4])
        # Ensure bbox coordinates are valid (x0 <= x1, y0 <= y1)
        bbox[:, :, 2] = np.maximum(bbox[:, :, 2], bbox[:, :, 0])
        bbox[:, :, 3] = np.maximum(bbox[:, :, 3], bbox[:, :, 1])

        # Create pixel_values for image input
        pixel_values = np.random.randn(self.batch_size, self.num_channels, self.image_size, self.image_size).astype(
            np.float32
        )

        config = self.get_config()
        decoder_input_ids = np.array([[0]])

        # set _attn_implementation
        config._attn_implementation = "eager"

        return (
            config,
            input_ids,
            decoder_input_ids,
            token_type_ids,
            input_mask,
            bbox,
            pixel_values,
        )

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            d_kv=self.d_kv,
            d_ff=self.d_ff,
            num_layers=self.num_layers,
            num_decoder_layers=self.num_decoder_layers,
            num_heads=self.num_heads,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance,
            relative_bias_args=self.relative_bias_args,
            dropout_rate=self.dropout_rate,
            layer_norm_epsilon=self.layer_norm_epsilon,
            initializer_factor=self.initializer_factor,
            feed_forward_proj=self.feed_forward_proj,
            is_encoder_decoder=self.is_encoder_decoder,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_2d_position_embeddings=self.max_2d_position_embeddings,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
        )


model_tester = UdopModelTester()
(
    config,
    input_ids,
    decoder_input_ids,
    token_type_ids,
    input_mask,
    bbox,
    pixel_values,
) = model_tester.prepare_config_and_inputs()


UDOP_CASES = [
    [
        "UdopModel",
        "transformers.UdopModel",
        "mindone.transformers.UdopModel",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": input_mask,
            "bbox": bbox,
            "pixel_values": pixel_values,
        },
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "UdopEncoderModel",
        "transformers.UdopEncoderModel",
        "mindone.transformers.UdopEncoderModel",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": input_mask,
            "bbox": bbox,
            "pixel_values": pixel_values,
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
        for case in UDOP_CASES
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
