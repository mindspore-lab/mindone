# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import copy
import inspect

import numpy as np
import pytest
import torch
from transformers import BertConfig

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
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3}
MODES = [0, 1]


class BertModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
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
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
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
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

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
        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        """
        Returns a tiny configuration by default.
        """
        return BertConfig(
            vocab_size=self.vocab_size,
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
        )


model_tester = BertModelTester()
(
    config,
    input_ids,
    token_type_ids,
    input_mask,
    sequence_labels,
    token_labels,
    choice_labels,
) = model_tester.prepare_config_and_inputs()
config_has_num_labels = copy.deepcopy(config)
config_has_num_labels.num_labels = model_tester.num_labels

BERT_CASES = [
    [
        "BertForMaskedLM",
        "transformers.BertForMaskedLM",
        "mindone.transformers.BertForMaskedLM",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": token_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "BertForMultipleChoice",
        "transformers.BertForMultipleChoice",
        "mindone.transformers.BertForMultipleChoice",
        (config,),
        {},
        (np.repeat(np.expand_dims(input_ids, 1), model_tester.num_choices, 1),),
        {
            "attention_mask": np.repeat(np.expand_dims(input_mask, 1), model_tester.num_choices, 1),
            "token_type_ids": np.repeat(np.expand_dims(token_type_ids, 1), model_tester.num_choices, 1),
            "labels": choice_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "BertForNextSentencePrediction",
        "transformers.BertForNextSentencePrediction",
        "mindone.transformers.BertForNextSentencePrediction",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": sequence_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "BertForPreTraining",
        "transformers.BertForPreTraining",
        "mindone.transformers.BertForPreTraining",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": token_labels,
            "next_sentence_label": sequence_labels,
        },
        {
            "loss": 0,
            "prediction_logits": 1,
            "seq_relationship_logits": 2,
        },
    ],
    [
        "BertForQuestionAnswering",
        "transformers.BertForQuestionAnswering",
        "mindone.transformers.BertForQuestionAnswering",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
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
        "BertForSequenceClassification",
        "transformers.BertForSequenceClassification",
        "mindone.transformers.BertForSequenceClassification",
        (config_has_num_labels,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": sequence_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "BertForTokenClassification",
        "transformers.BertForTokenClassification",
        "mindone.transformers.BertForTokenClassification",
        (config_has_num_labels,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": token_labels,
        },
        {
            "loss": 0,
            "logits": 1,
        },
    ],
    [
        "BertModel",
        "transformers.BertModel",
        "mindone.transformers.BertModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        },
        {
            "last_hidden_state": 0,
            "pooler_output": 1,
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
