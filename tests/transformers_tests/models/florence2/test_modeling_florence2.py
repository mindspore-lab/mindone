"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/florence2/test_modeling_florence2.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.
import inspect

import numpy as np
import pytest
import torch
from transformers import Florence2Config

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

MODES = [1]


class Florence2VisionText2TextModelTester:
    def __init__(
        self,
        batch_size=13,
        num_channels=3,
        image_size=8,
        seq_length=13,
        encoder_seq_length=18,
        is_training=True,
        vocab_size=99,
        max_position_embeddings=64,
        encoder_layers=1,
        encoder_ffn_dim=16,
        decoder_layers=1,
        decoder_ffn_dim=16,
        num_attention_heads=1,
        d_model=16,
        activation_function="gelu",
        dropout=0.1,
        eos_token_id=2,
        bos_token_id=0,
        pad_token_id=1,
        image_token_id=4,
        depths=[1],
        patch_size=[7],
        patch_stride=[4],
        patch_padding=[3],
        patch_prenorm=[False],
        embed_dim=[16],
        num_heads=[1],
        num_groups=[1],
        window_size=12,
        drop_path_rate=0.1,
        projection_dim=16,
    ):
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.is_training = is_training
        self.num_hidden_layers = decoder_layers
        self.hidden_size = d_model

        # Language model configs
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.num_attention_heads = num_attention_heads
        self.d_model = d_model
        self.activation_function = activation_function
        self.dropout = dropout
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.image_token_id = image_token_id

        # Vision model configs
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.depths = depths
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.patch_prenorm = patch_prenorm
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.window_size = window_size
        self.projection_dim = projection_dim

        self.num_channels = 3
        self.num_image_tokens = 5
        self.seq_length = seq_length + self.num_image_tokens
        self.encoder_seq_length = encoder_seq_length

    def get_config(self):
        text_config = {
            "model_type": "bart",
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "encoder_layers": self.encoder_layers,
            "encoder_ffn_dim": self.encoder_ffn_dim,
            "encoder_attention_heads": self.num_attention_heads,
            "decoder_layers": self.decoder_layers,
            "decoder_ffn_dim": self.decoder_ffn_dim,
            "decoder_attention_heads": self.num_attention_heads,
            "d_model": self.d_model,
            "activation_function": self.activation_function,
            "dropout": self.dropout,
            "attention_dropout": self.dropout,
            "activation_dropout": self.dropout,
            "eos_token_id": self.eos_token_id,
            "bos_token_id": self.bos_token_id,
            "pad_token_id": self.pad_token_id,
        }

        vision_config = {
            "drop_path_rate": self.drop_path_rate,
            "patch_size": self.patch_size,
            "depths": self.depths,
            "patch_stride": self.patch_stride,
            "patch_padding": self.patch_padding,
            "patch_prenorm": self.patch_prenorm,
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_groups": self.num_groups,
            "window_size": self.window_size,
            "activation_function": self.activation_function,
            "projection_dim": self.projection_dim,
        }

        return Florence2Config(
            text_config=text_config,
            vision_config=vision_config,
            image_token_id=self.image_token_id,
            initializer_range=0.02,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy(
            [
                self.batch_size,
                self.num_channels,
                self.image_size,
                self.image_size,
            ]
        )
        input_ids = ids_numpy([self.batch_size, self.encoder_seq_length], self.vocab_size - 1) + 1
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.num_image_tokens] = self.image_token_id
        input_ids[:, -1] = self.eos_token_id
        decoder_input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        decoder_attention_mask = np.not_equal(decoder_input_ids, self.pad_token_id)

        inputs_dict = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }

        config = self.get_config()
        return config, inputs_dict

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


model_tester = Florence2VisionText2TextModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs()


TEST_CASES = [
    [
        "Florence2Model",
        "transformers.Florence2Model",
        "mindone.transformers.Florence2Model",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "Florence2ForConditionalGeneration",
        "transformers.Florence2ForConditionalGeneration",
        "mindone.transformers.Florence2ForConditionalGeneration",
        (config,),
        {},
        (),
        inputs_dict,
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
        for case in TEST_CASES
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
