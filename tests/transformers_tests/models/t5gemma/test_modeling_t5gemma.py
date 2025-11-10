"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/t5gemma/test_modeling_t5gemma.py."""

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
from transformers import T5GemmaConfig, T5GemmaModuleConfig

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
MODES = [1]


class T5GemmaModelTester:
    config_class = T5GemmaConfig
    module_config_class = T5GemmaModuleConfig

    def __init__(
        self,
        batch_size=13,
        is_training=True,
        use_attention_mask=True,
        use_labels=True,
        vocab_size=99,
        # decoder-specific
        seq_length=7,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=37,
        # encoder-specific
        encoder_seq_length=7,
        encoder_hidden_size=32,
        encoder_num_hidden_layers=2,
        encoder_num_attention_heads=4,
        encoder_num_key_value_heads=2,
        encoder_intermediate_size=37,
        # common
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
        # special ids
        eos_token_id=1,
        pad_token_id=0,
        bos_token_id=2,
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        # decoder
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        # encoder
        self.encoder_seq_length = encoder_seq_length
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_hidden_layers = encoder_num_hidden_layers
        self.encoder_num_attention_heads = encoder_num_attention_heads
        self.encoder_num_key_value_heads = encoder_num_key_value_heads
        self.encoder_intermediate_size = encoder_intermediate_size
        # common
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
        self.head_dim = self.hidden_size // self.num_attention_heads
        # assume encoder and decoder have the same head dimension.
        assert self.head_dim == self.encoder_hidden_size // self.encoder_num_attention_heads
        # special ids
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        # assume the number of attention heads are the same across encoder and decoder
        # only used for generation testing purpose.
        assert self.num_attention_heads == self.encoder_num_attention_heads

    def get_encoder_config(self):
        return self.module_config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.encoder_hidden_size,
            num_hidden_layers=self.encoder_num_hidden_layers,
            num_attention_heads=self.encoder_num_attention_heads,
            num_key_value_heads=self.encoder_num_key_value_heads,
            intermediate_size=self.encoder_intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=False,
            initializer_range=self.initializer_range,
            head_dim=self.head_dim,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_decoder_config(self):
        return self.module_config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            cross_attention_hidden_size=self.encoder_hidden_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            is_decoder=True,
            initializer_range=self.initializer_range,
            head_dim=self.head_dim,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_config(self, is_encoder_decoder=True):
        return self.config_class(
            encoder=self.get_encoder_config(),
            decoder=self.get_decoder_config(),
            is_encoder_decoder=is_encoder_decoder,
            # Used for generation test.
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
        )

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.encoder_seq_length], self.vocab_size)
        decoder_input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        # Remove BOS symbols from inputs.
        input_ids = np.where(input_ids == self.bos_token_id, 42, input_ids)
        decoder_input_ids = np.where(decoder_input_ids == self.bos_token_id, 42, decoder_input_ids)

        attention_mask = None
        decoder_attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.encoder_seq_length], vocab_size=2)
            decoder_attention_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

        lm_labels = None
        if self.use_labels:
            lm_labels = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            attention_mask,
            decoder_attention_mask,
            lm_labels,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
        }
        return config, inputs_dict


model_tester = T5GemmaModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


TEST_CASES = [
    [
        "T5GemmaModel",
        "transformers.T5GemmaModel",
        "mindone.transformers.T5GemmaModel",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "T5GemmaForConditionalGeneration",
        "transformers.T5GemmaForConditionalGeneration",
        "mindone.transformers.T5GemmaForConditionalGeneration",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "logits": "logits",
        },
    ],
    [
        "T5GemmaForSequenceClassification",
        "transformers.T5GemmaForSequenceClassification",
        "mindone.transformers.T5GemmaForSequenceClassification",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "logits": "logits",
        },
    ],
    [
        "T5GemmaForTokenClassification",
        "transformers.T5GemmaForTokenClassification",
        "mindone.transformers.T5GemmaForTokenClassification",
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
