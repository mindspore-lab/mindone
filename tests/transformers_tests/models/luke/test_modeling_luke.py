# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
from transformers import LukeConfig

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
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 1e-2}
MODES = [1]


class LukeModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        entity_length=2,
        mention_length=5,
        use_attention_mask=True,
        use_token_type_ids=True,
        use_entity_ids=True,
        use_entity_attention_mask=True,
        use_entity_token_type_ids=True,
        use_entity_position_ids=True,
        use_labels=True,
        vocab_size=99,
        entity_vocab_size=10,
        entity_emb_size=6,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=2,
        num_choices=4,
        num_entity_classification_labels=2,
        num_entity_pair_classification_labels=6,
        num_entity_span_classification_labels=2,
        use_entity_aware_attention=True,
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.entity_length = entity_length
        self.mention_length = mention_length
        self.use_attention_mask = use_attention_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_entity_ids = use_entity_ids
        self.use_entity_attention_mask = use_entity_attention_mask
        self.use_entity_token_type_ids = use_entity_token_type_ids
        self.use_entity_position_ids = use_entity_position_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.entity_vocab_size = entity_vocab_size
        self.entity_emb_size = entity_emb_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.num_entity_classification_labels = num_entity_classification_labels
        self.num_entity_pair_classification_labels = num_entity_pair_classification_labels
        self.num_entity_span_classification_labels = num_entity_span_classification_labels
        self.scope = scope
        self.use_entity_aware_attention = use_entity_aware_attention

        self.encoder_seq_length = seq_length
        self.key_length = seq_length
        self.num_hidden_states_types = 2  # hidden_states and entity_hidden_states

    def prepare_config_and_inputs(self):
        # prepare words
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        # prepare entities
        entity_ids = ids_numpy([self.batch_size, self.entity_length], self.entity_vocab_size)

        entity_attention_mask = None
        if self.use_entity_attention_mask:
            entity_attention_mask = random_attention_mask([self.batch_size, self.entity_length])

        entity_token_type_ids = None
        if self.use_token_type_ids:
            entity_token_type_ids = ids_numpy([self.batch_size, self.entity_length], self.type_vocab_size)

        entity_position_ids = None
        if self.use_entity_position_ids:
            entity_position_ids = ids_numpy(
                [self.batch_size, self.entity_length, self.mention_length], self.mention_length
            )

        sequence_labels = None
        token_labels = None
        choice_labels = None
        entity_labels = None
        entity_classification_labels = None
        entity_pair_classification_labels = None
        entity_span_classification_labels = None

        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

            entity_labels = ids_numpy([self.batch_size, self.entity_length], self.entity_vocab_size)

            entity_classification_labels = ids_numpy([self.batch_size], self.num_entity_classification_labels)
            entity_pair_classification_labels = ids_numpy([self.batch_size], self.num_entity_pair_classification_labels)
            entity_span_classification_labels = ids_numpy(
                [self.batch_size, self.entity_length], self.num_entity_span_classification_labels
            )
        entity_start_positions = ids_numpy([self.batch_size, self.entity_length], self.seq_length)
        entity_end_positions = ids_numpy([self.batch_size, self.entity_length], self.seq_length)

        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            token_type_ids,
            entity_ids,
            entity_attention_mask,
            entity_token_type_ids,
            entity_position_ids,
            sequence_labels,
            token_labels,
            choice_labels,
            entity_labels,
            entity_classification_labels,
            entity_pair_classification_labels,
            entity_span_classification_labels,
            entity_start_positions,
            entity_end_positions,
        )

    def get_config(self):
        return LukeConfig(
            vocab_size=self.vocab_size,
            entity_vocab_size=self.entity_vocab_size,
            entity_emb_size=self.entity_emb_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            use_entity_aware_attention=self.use_entity_aware_attention,
        )


model_tester = LukeModelTester()
(
    config,
    input_ids,
    attention_mask,
    token_type_ids,
    entity_ids,
    entity_attention_mask,
    entity_token_type_ids,
    entity_position_ids,
    sequence_labels,
    token_labels,
    choice_labels,
    entity_labels,
    entity_classification_labels,
    entity_pair_classification_labels,
    entity_span_classification_labels,
    entity_start_positions,
    entity_end_positions,
) = model_tester.prepare_config_and_inputs()

BERT_CASES = [
    [
        "LukeForMultipleChoice",
        "transformers.LukeForMultipleChoice",
        "mindone.transformers.LukeForMultipleChoice",
        (config,),
        {},
        (np.repeat(np.expand_dims(input_ids, 1), model_tester.num_choices, 1),),
        {
            "attention_mask": np.repeat(np.expand_dims(attention_mask, 1), model_tester.num_choices, 1),
            "token_type_ids": np.repeat(np.expand_dims(token_type_ids, 1), model_tester.num_choices, 1),
        },
        {
            "logits": 1,
        },
    ],
    [
        "LukeForMaskedLM",
        "transformers.LukeForMaskedLM",
        "mindone.transformers.LukeForMaskedLM",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": token_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "LukeForQuestionAnswering",
        "transformers.LukeForQuestionAnswering",
        "mindone.transformers.LukeForQuestionAnswering",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "start_positions": sequence_labels,
            "end_positions": sequence_labels,
        },
        {
            "loss": 0,
            "start_logits": 1,
            "end_logits": 2,
        },
    ],
    [
        "LukeForTokenClassification",
        "transformers.LukeForTokenClassification",
        "mindone.transformers.LukeForTokenClassification",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": token_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "LukeForSequenceClassification",
        "transformers.LukeForSequenceClassification",
        "mindone.transformers.LukeForSequenceClassification",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": sequence_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "LukeModel",
        "transformers.LukeModel",
        "mindone.transformers.LukeModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
        },
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "LukeForEntityClassification",
        "transformers.LukeForEntityClassification",
        "mindone.transformers.LukeForEntityClassification",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "entity_ids": entity_ids,
            "entity_attention_mask": entity_attention_mask,
            "entity_token_type_ids": entity_token_type_ids,
            "entity_position_ids": entity_position_ids,
            "labels": entity_classification_labels,
        },
        {
            "logits": 0,
        },
    ],
    [
        "LukeForEntitySpanClassification",
        "transformers.LukeForEntitySpanClassification",
        "mindone.transformers.LukeForEntitySpanClassification",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "entity_ids": entity_ids,
            "entity_attention_mask": entity_attention_mask,
            "entity_token_type_ids": entity_token_type_ids,
            "entity_position_ids": entity_position_ids,
            "entity_start_positions": entity_start_positions,
            "entity_end_positions": entity_end_positions,
            "labels": entity_span_classification_labels,
        },
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
