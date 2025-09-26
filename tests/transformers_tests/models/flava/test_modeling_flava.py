# coding=utf-8
# Copyright 2022 Meta Platforms authors and The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Testing suite for the MindSpore FLAVA model."""

import numpy as np
import pytest
import torch
from transformers import FlavaConfig, FlavaImageCodebookConfig, FlavaImageConfig, FlavaMultimodalConfig, FlavaTextConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # not support graph mode yet


class FlavaImageModelTester:
    def __init__(
        self,
        batch_size=12,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        image_size=30,
        patch_size=2,
        num_channels=3,
        qkv_bias=True,
        mask_token=True,
        vocab_size=99,
    ):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.mask_token = mask_token
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        num_patches = self.image_size // self.patch_size
        bool_masked_pos = (np.random.rand(self.batch_size, num_patches, num_patches) < 0.9).astype(np.int64)
        config = self.get_config()
        return config, pixel_values, bool_masked_pos

    def get_config(self):
        return FlavaImageConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
            qkv_bias=self.qkv_bias,
            mask_token=self.mask_token,
            vocab_size=self.vocab_size,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, bool_masked_pos = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values, "bool_masked_pos": bool_masked_pos}
        return config, inputs_dict


class FlavaTextModelTester:
    def __init__(
        self,
        batch_size=12,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        vocab_size=102,
        type_vocab_size=2,
        max_position_embeddings=512,
        position_embedding_type="absolute",
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        qkv_bias=True,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.position_embedding_type = position_embedding_type
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.pad_token_id = pad_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        token_type_ids = None

        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()

        return config, input_ids, token_type_ids, input_mask

    def get_config(self):
        return FlavaTextConfig(
            vocab_size=self.vocab_size,
            type_vocab_size=self.type_vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            position_embedding_type=self.position_embedding_type,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            qkv_bias=self.qkv_bias,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, token_type_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "attention_mask": input_mask}
        return config, inputs_dict


class FlavaMultimodalModelTester:
    def __init__(
        self,
        batch_size=12,
        seq_length=44,
        use_input_mask=True,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        qkv_bias=True,
        ce_ignore_index=-100,
        use_cls_token=True,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_input_mask = use_input_mask
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.qkv_bias = qkv_bias
        self.ce_ignore_index = ce_ignore_index
        self.use_cls_token = use_cls_token

    def prepare_config_and_inputs(self):
        hidden_states = floats_numpy([self.batch_size, self.seq_length - 1, self.hidden_size])

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        config = self.get_config()

        return config, hidden_states, input_mask

    def get_config(self):
        return FlavaMultimodalConfig(
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            qkv_bias=self.qkv_bias,
            use_cls_token=self.use_cls_token,
            ce_ignore_index=self.ce_ignore_index,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, hidden_states, input_mask = config_and_inputs
        inputs_dict = {"hidden_states": hidden_states, "attention_mask": input_mask}
        return config, inputs_dict


class FlavaImageCodebookTester:
    def __init__(
        self,
        batch_size=12,
        image_size=112,
        num_channels=3,
        hidden_size=32,
        num_groups=2,
        vocab_size=99,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.vocab_size = vocab_size

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])
        config = self.get_config()

        return config, pixel_values

    def get_config(self):
        return FlavaImageCodebookConfig(
            hidden_size=self.hidden_size, num_groups=self.num_groups, vocab_size=self.vocab_size
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs
        inputs_dict = {"pixel_values": pixel_values}
        return config, inputs_dict


class FlavaModelTester:
    def __init__(
        self,
        text_kwargs=None,
        image_kwargs=None,
        multimodal_kwargs=None,
        image_codebook_kwargs=None,
        is_training=True,
        hidden_size=32,
        projection_dim=32,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
    ):
        if text_kwargs is None:
            text_kwargs = {}
        if image_kwargs is None:
            image_kwargs = {}
        if multimodal_kwargs is None:
            multimodal_kwargs = {}
        if image_codebook_kwargs is None:
            image_codebook_kwargs = {}

        self.image_model_tester = FlavaImageModelTester(**image_kwargs)
        self.text_model_tester = FlavaTextModelTester(**text_kwargs)
        self.multimodal_model_tester = FlavaMultimodalModelTester(**multimodal_kwargs)
        self.image_codebook_tester = FlavaImageCodebookTester(**image_codebook_kwargs)
        self.is_training = is_training
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.batch_size = self.text_model_tester.batch_size  # need bs for batching_equivalence test

    def prepare_config_and_inputs_for_common(self):
        _, pixel_values, bool_masked_pos = self.image_model_tester.prepare_config_and_inputs()
        _, input_ids, token_type_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()

        config = self.get_config()

        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "bool_masked_pos": bool_masked_pos,
        }

    def get_config(self):
        return FlavaConfig.from_configs(
            self.image_model_tester.get_config(),
            self.text_model_tester.get_config(),
            self.multimodal_model_tester.get_config(),
            self.image_codebook_tester.get_config(),
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
        )


class FlavaForPreTrainingTester(FlavaModelTester):
    def prepare_config_and_inputs_for_common(self):
        _, pixel_values, bool_masked_pos = self.image_model_tester.prepare_config_and_inputs()
        _, input_ids, token_type_ids, attention_mask = self.text_model_tester.prepare_config_and_inputs()
        config = self.get_config()

        input_ids_masked = input_ids.copy()
        input_ids_masked[:, 1:3] = 100
        mlm_labels = input_ids.copy()
        mlm_labels[:, :] = config.ce_ignore_index
        mlm_labels[:, 1:3] = input_ids[:, 1:3]
        mim_labels = np.random.randint(0, self.image_model_tester.vocab_size, bool_masked_pos.shape).astype(np.int64)
        mim_labels[bool_masked_pos != 1] = config.ce_ignore_index
        itm_labels = np.ones(mlm_labels.shape[0]).astype(np.int64)

        return config, {
            "input_ids": input_ids,
            "input_ids_masked": input_ids_masked,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "bool_masked_pos": bool_masked_pos,
            "mlm_labels": mlm_labels,
            "mim_labels": mim_labels,
            "itm_labels": itm_labels,
            "return_loss": True,
        }


model_tester = FlavaModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


_CASES = [
    [
        "FlavaModelTester",
        "transformers.FlavaModel",
        "mindone.transformers.FlavaModel",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "multimodal_embeddings": 4,
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
        for case in _CASES
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
