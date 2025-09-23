"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/funnel/test_modeling_funnel.py."""

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
from transformers import FunnelConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4}
MODES = [1]


class FunnelModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        block_sizes=[1, 1, 2],
        num_decoder_layers=1,
        d_model=32,
        n_head=4,
        d_head=8,
        d_inner=37,
        hidden_act="gelu_new",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        max_position_embeddings=512,
        type_vocab_size=3,
        initializer_std=0.02,  # Set to a smaller value, so we can keep the small error threshold (1e-5) in the test
        num_labels=3,
        num_choices=4,
        scope=None,
        base=False,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.block_sizes = block_sizes
        self.num_decoder_layers = num_decoder_layers
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.d_inner = d_inner
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = 2
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.initializer_std = initializer_std

        # Used in the tests to check the size of the first attention layer
        self.num_attention_heads = n_head
        # Used in the tests to check the size of the first hidden state
        self.hidden_size = self.d_model
        # Used in the tests to check the number of output hidden states/attentions
        self.num_hidden_layers = sum(self.block_sizes) + (0 if base else self.num_decoder_layers)
        # FunnelModel adds two hidden layers: input embeddings and the sum of the upsampled encoder hidden state with
        # the last hidden state of the first block (which is the first hidden state of the decoder).
        if not base:
            self.expected_num_hidden_layers = self.num_hidden_layers + 2

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

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
            fake_token_labels = ids_numpy([self.batch_size, self.seq_length], 1)

        config = self.get_config()

        return (
            config,
            input_ids,
            token_type_ids,
            input_mask,
            sequence_labels,
            token_labels,
            choice_labels,
            fake_token_labels,
        )

    def get_config(self):
        return FunnelConfig(
            vocab_size=self.vocab_size,
            block_sizes=self.block_sizes,
            num_decoder_layers=self.num_decoder_layers,
            d_model=self.d_model,
            n_head=self.n_head,
            d_head=self.d_head,
            d_inner=self.d_inner,
            hidden_act=self.hidden_act,
            hidden_dropout=self.hidden_dropout,
            attention_dropout=self.attention_dropout,
            activation_dropout=self.activation_dropout,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_std=self.initializer_std,
        )


model_tester = FunnelModelTester()
(
    config,
    input_ids,
    token_type_ids,
    input_mask,
    sequence_labels,
    token_labels,
    choice_labels,
    fake_token_labels,
) = model_tester.prepare_config_and_inputs()


FUNNEL_CASES = [
    [
        "FunnelBaseModel",
        "transformers.FunnelBaseModel",
        "mindone.transformers.FunnelBaseModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        },
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "FunnelModel",
        "transformers.FunnelModel",
        "mindone.transformers.FunnelModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
        },
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "FunnelForMaskedLM",
        "transformers.FunnelForMaskedLM",
        "mindone.transformers.FunnelForMaskedLM",
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
        "FunnelForSequenceClassification",
        "transformers.FunnelForSequenceClassification",
        "mindone.transformers.FunnelForSequenceClassification",
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
        "FunnelForQuestionAnswering",
        "transformers.FunnelForQuestionAnswering",
        "mindone.transformers.FunnelForQuestionAnswering",
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
        "FunnelForPreTraining",
        "transformers.FunnelForPreTraining",
        "mindone.transformers.FunnelForPreTraining",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "token_type_ids": token_type_ids,
            "labels": fake_token_labels,
        },
        {
            "loss": 0,
            "logits": 1,
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
        for case in FUNNEL_CASES
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
