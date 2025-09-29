"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/perceiver/test_modeling_perceiver.py."""

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
from transformers import PerceiverConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 8e-3}
# MODES = [0, 1] # 0: graph mode, 1: pynative mode
# FIXME: Perceiver does not support graph mode yet, so we only test in pynative mode.
MODES = [1]


class PerceiverModelTester:
    config_class = PerceiverConfig

    def __init__(
        self,
        batch_size=2,
        seq_length=7,
        is_training=False,
        use_input_mask=False,
        use_token_type_ids=False,
        use_labels=True,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        # config - reduced sizes for testing
        num_latents=64,
        d_latents=320,
        d_model=192,
        num_blocks=1,
        num_self_attends_per_block=2,
        num_self_attention_heads=2,
        num_cross_attention_heads=2,
        qk_channels=None,
        v_channels=None,
        cross_attention_shape_for_attention="kv",
        self_attention_widening_factor=1,
        cross_attention_widening_factor=1,
        hidden_act="gelu",
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        use_query_residual=True,
        vocab_size=262,
        max_position_embeddings=2048,
        image_size=56,
        train_size=[368, 496],
        num_frames=16,
        audio_samples_per_frame=1920,
        samples_per_patch=16,
        output_shape=[1, 16, 224, 224],
        output_num_channels=512,
        _label_trainable_num_channels=1024,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.num_latents = num_latents
        self.d_latents = d_latents
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_self_attention_heads = num_self_attention_heads
        self.num_cross_attention_heads = num_cross_attention_heads
        self.qk_channels = qk_channels
        self.v_channels = v_channels
        self.cross_attention_shape_for_attention = cross_attention_shape_for_attention
        self.self_attention_widening_factor = self_attention_widening_factor
        self.cross_attention_widening_factor = cross_attention_widening_factor
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_query_residual = use_query_residual
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.image_size = image_size
        self.train_size = train_size
        self.num_frames = num_frames
        self.audio_samples_per_frame = audio_samples_per_frame
        self.samples_per_patch = samples_per_patch
        self.output_shape = output_shape
        self.output_num_channels = output_num_channels
        self._label_trainable_num_channels = _label_trainable_num_channels

    def prepare_config_and_inputs(self):
        input_embeds = np.random.randn(self.batch_size, self.seq_length, self.d_model)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.ones([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config()

        # set _attn_implementation
        config._attn_implementation = "eager"

        return config, input_embeds, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return self.config_class(
            num_latents=self.num_latents,
            d_latents=self.d_latents,
            d_model=self.d_model,
            num_blocks=self.num_blocks,
            num_self_attends_per_block=self.num_self_attends_per_block,
            num_self_attention_heads=self.num_self_attention_heads,
            num_cross_attention_heads=self.num_cross_attention_heads,
            qk_channels=self.qk_channels,
            v_channels=self.v_channels,
            cross_attention_shape_for_attention=self.cross_attention_shape_for_attention,
            self_attention_widening_factor=self.self_attention_widening_factor,
            cross_attention_widening_factor=self.cross_attention_widening_factor,
            hidden_act=self.hidden_act,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            use_query_residual=self.use_query_residual,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            image_size=self.image_size,
            train_size=self.train_size,
            num_frames=self.num_frames,
            audio_samples_per_frame=self.audio_samples_per_frame,
            samples_per_patch=self.samples_per_patch,
            output_shape=self.output_shape,
            output_num_channels=self.output_num_channels,
            _label_trainable_num_channels=self._label_trainable_num_channels,
        )


model_tester = PerceiverModelTester()
(
    config,
    input_embeds,
    token_type_ids,
    input_mask,
    sequence_labels,
    token_labels,
    choice_labels,
) = model_tester.prepare_config_and_inputs()


PERCEIVER_CASES = [
    [
        "PerceiverModel",
        "transformers.PerceiverModel",
        "mindone.transformers.PerceiverModel",
        (config,),
        {},
        (input_embeds,),
        {
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
        for case in PERCEIVER_CASES
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
