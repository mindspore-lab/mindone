# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team, The Microsoft Research team.
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
import inspect

import numpy as np
import pytest
import torch
from transformers import ProphetNetConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

# CrossEntropyLoss not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 6e-3, "fp16": 6e-3, "bf16": 1e-2}
MODES = [1]


class ProphetNetModelTester:
    def __init__(
        self,
        vocab_size=99,
        batch_size=13,
        hidden_size=16,
        encoder_seq_length=7,
        decoder_seq_length=9,
        # For common tests
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        decoder_start_token_id=0,
        encoder_ffn_dim=32,
        num_encoder_layers=2,
        num_encoder_attention_heads=4,
        decoder_ffn_dim=32,
        num_decoder_layers=2,
        num_decoder_attention_heads=4,
        max_position_embeddings=30,
        is_encoder_decoder=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        ngram=2,
        num_buckets=32,
        relative_max_distance=128,
        disable_ngram_loss=False,
        scope=None,
    ):
        self.batch_size = batch_size
        self.encoder_seq_length = encoder_seq_length
        self.decoder_seq_length = decoder_seq_length
        # For common tests
        self.seq_length = self.decoder_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_decoder_layers
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.num_attention_heads = num_decoder_attention_heads
        self.num_encoder_attention_heads = num_encoder_attention_heads
        self.num_decoder_attention_heads = num_decoder_attention_heads
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id
        self.ngram = ngram
        self.num_buckets = num_buckets
        self.relative_max_distance = relative_max_distance
        self.disable_ngram_loss = disable_ngram_loss
        self.max_position_embeddings = max_position_embeddings
        self.is_encoder_decoder = is_encoder_decoder

        self.scope = None
        self.decoder_key_length = decoder_seq_length
        self.base_model_out_len = 7
        self.num_hidden_states_types = 3  # encoder, decoder_main, decoder_ngram
        self.decoder_attention_idx = 2

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_numpy([self.batch_size, self.decoder_seq_length], self.vocab_size)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_numpy([self.batch_size, self.decoder_seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_numpy([self.batch_size, self.decoder_seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def get_config(self):
        return ProphetNetConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            encoder_ffn_dim=self.encoder_ffn_dim,
            num_encoder_attention_heads=self.num_encoder_attention_heads,
            num_decoder_attention_heads=self.num_decoder_attention_heads,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            decoder_start_token_id=self.decoder_start_token_id,
            ngram=self.ngram,
            num_buckets=self.num_buckets,
            relative_max_distance=self.relative_max_distance,
            disable_ngram_loss=self.disable_ngram_loss,
            max_position_embeddings=self.max_position_embeddings,
            is_encoder_decoder=self.is_encoder_decoder,
        )


model_tester = ProphetNetModelTester()
(
    config,
    input_ids,
    decoder_input_ids,
    attention_mask,
    decoder_attention_mask,
    lm_labels,
) = model_tester.prepare_config_and_inputs()

BERT_CASES = [
    [
        "ProphetNetForConditionalGeneration",
        "transformers.ProphetNetForConditionalGeneration",
        "mindone.transformers.ProphetNetForConditionalGeneration",
        (config,),
        {},
        (input_ids,),
        {
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "labels": lm_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "ProphetNetForCausalLM",
        "transformers.ProphetNetForCausalLM",
        "mindone.transformers.ProphetNetForCausalLM",
        (config,),
        {},
        (decoder_input_ids,),
        {
            "attention_mask": decoder_attention_mask,
            "labels": lm_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "ProphetNetModel",
        "transformers.ProphetNetModel",
        "mindone.transformers.ProphetNetModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
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
        for case in BERT_CASES
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
            # print("===map", pt_key, ms_idx)
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[pt_key]
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
