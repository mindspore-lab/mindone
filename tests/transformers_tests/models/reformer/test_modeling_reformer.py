# coding=utf-8 # Copyright 2020 Huggingface
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on MindSpore.
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
# limitations under the License

import inspect

import numpy as np
import pytest
import torch
from transformers import ReformerConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy, random_attention_mask

# CrossEntropyLoss not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class ReformerModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=32,
        text_seq_length=None,
        is_training=True,
        is_decoder=True,
        use_input_mask=True,
        use_labels=True,
        vocab_size=32,
        attention_head_size=16,
        hidden_size=32,
        num_attention_heads=2,
        local_attn_chunk_length=4,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        num_buckets=None,
        num_hashes=1,
        lsh_attn_chunk_length=None,
        lsh_num_chunks_before=None,
        lsh_num_chunks_after=None,
        chunk_size_lm_head=0,
        chunk_size_feed_forward=0,
        feed_forward_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        local_attention_probs_dropout_prob=0.1,
        lsh_attention_probs_dropout_prob=None,
        max_position_embeddings=512,
        initializer_range=0.02,
        axial_norm_std=1.0,
        layer_norm_eps=1e-12,
        axial_pos_embds=True,
        axial_pos_shape=[4, 8],
        axial_pos_embds_dim=[16, 16],
        attn_layers=["local", "local", "local", "local"],
        pad_token_id=0,
        eos_token_id=2,
        scope=None,
        hash_seed=0,
        num_labels=2,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.is_decoder = is_decoder
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.attention_head_size = attention_head_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = len(attn_layers) if attn_layers is not None else 0
        self.local_attn_chunk_length = local_attn_chunk_length
        self.local_num_chunks_after = local_num_chunks_after
        self.local_num_chunks_before = local_num_chunks_before
        self.num_hashes = num_hashes
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        self.lsh_num_chunks_after = lsh_num_chunks_after
        self.lsh_num_chunks_before = lsh_num_chunks_before
        self.hidden_act = hidden_act
        self.feed_forward_size = feed_forward_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.axial_pos_embds = axial_pos_embds
        self.axial_pos_shape = tuple(axial_pos_shape)
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        self.axial_norm_std = axial_norm_std
        self.chunk_size_lm_head = chunk_size_lm_head
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.scope = scope
        self.attn_layers = attn_layers
        self.pad_token_id = pad_token_id
        self.hash_seed = hash_seed
        self.text_seq_length = text_seq_length or seq_length

        attn_chunk_length = local_attn_chunk_length if local_attn_chunk_length is not None else lsh_attn_chunk_length
        num_chunks_after = local_num_chunks_after if local_num_chunks_after is not None else lsh_num_chunks_after
        num_chunks_before = local_num_chunks_before if local_num_chunks_before is not None else lsh_num_chunks_before

        self.encoder_seq_length = seq_length // attn_chunk_length + (self.seq_length % attn_chunk_length != 0)
        self.key_length = (num_chunks_before + num_chunks_after + 1) * attn_chunk_length
        self.chunk_length = attn_chunk_length
        self.num_labels = num_labels

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        choice_labels = None
        if self.use_labels:
            choice_labels = ids_numpy([self.batch_size], 2)

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            choice_labels,
        )

    def get_config(self):
        return ReformerConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            feed_forward_size=self.feed_forward_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            local_attention_probs_dropout_prob=self.local_attention_probs_dropout_prob,
            lsh_attention_probs_dropout_prob=self.lsh_attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=self.is_decoder,
            axial_pos_embds=self.axial_pos_embds,
            axial_pos_shape=self.axial_pos_shape,
            axial_pos_embds_dim=self.axial_pos_embds_dim,
            local_attn_chunk_length=self.local_attn_chunk_length,
            local_num_chunks_after=self.local_num_chunks_after,
            local_num_chunks_before=self.local_num_chunks_before,
            num_hashes=self.num_hashes,
            num_buckets=self.num_buckets,
            lsh_attn_chunk_length=self.lsh_attn_chunk_length,
            lsh_num_chunks_after=self.lsh_num_chunks_after,
            lsh_num_chunks_before=self.lsh_num_chunks_before,
            attn_layers=self.attn_layers,
            pad_token_id=self.pad_token_id,
            hash_seed=self.hash_seed,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 100
        config.max_position_embeddings = 100
        config.axial_pos_shape = (4, 25)
        config.is_decoder = False
        return config

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (config, input_ids, input_mask, choice_labels) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return config, inputs_dict


model_tester = ReformerModelTester()
(
    config,
    inputs_dict,
) = model_tester.prepare_config_and_inputs_for_common()

Reformer_CASES = [
    [
        "ReformerModel",
        "transformers.ReformerModel",
        "mindone.transformers.ReformerModel",
        (config,),
        {},
        (),
        {
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["attention_mask"],
        },
        {
            "last_hidden_state": 0,
        },
    ],
]


# FIXME the test requires MindSpore to support the ParameterList feature.
# https://gitee.com/mindspore/mindspore/pulls/88092
@pytest.mark.skipif(ms.__version__ <= "2.7.0", reason="mindspore has not yet supported nn.ParameterList")
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
        for case in Reformer_CASES
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
