"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/bit/test_modeling_bit.py."""

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
from transformers import xLSTMConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

# PadV3 not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-3, "fp16": 5e-2, "bf16": 5e-2}
MODES = [1]


class xLSTMModelTester:
    def __init__(
        self,
        batch_size=13,
        num_heads=2,
        seq_length=7,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=128,
        qk_dim_factor=0.5,
        v_dim_factor=1.0,
        num_hidden_layers=2,
        max_position_embeddings=512,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        scope=None,
        chunkwise_kernel="chunkwise--native_autograd",
        sequence_kernel="native_sequence__native",
        step_kernel="native",
        tie_word_embeddings=False,
    ):
        self.num_heads = num_heads
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.qk_dim_factor = qk_dim_factor
        self.v_dim_factor = v_dim_factor
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1
        self.pad_token_id = vocab_size - 1
        self.chunkwise_kernel = chunkwise_kernel
        self.sequence_kernel = sequence_kernel
        self.step_kernel = step_kernel
        self.tie_word_embeddings = tie_word_embeddings

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        sequence_labels = None
        token_labels = None
        choice_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config()

        return (
            config,
            input_ids,
            None,
            sequence_labels,
            token_labels,
            choice_labels,
        )

    def get_config(self):
        cfg = xLSTMConfig(
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            qk_dim_factor=self.qk_dim_factor,
            v_dim_factor=self.v_dim_factor,
            n_positions=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            chunkwise_kernel=self.chunkwise_kernel,
            sequence_kernel=self.sequence_kernel,
            step_kernel=self.step_kernel,
            tie_word_embeddings=self.tie_word_embeddings,
        )
        # this is needed for compatibility with generic tests
        # cfg.hidden_size = cfg.embedding_dim
        # cfg.num_hidden_layers = cfg.num_blocks
        return cfg

    def get_pipeline_config(self):
        config = self.get_config()
        return config

    def prepare_config_and_inputs_for_common(self):
        (
            config,
            input_ids,
            _,
            sequence_labels,
            token_labels,
            choice_labels,
        ) = self.prepare_config_and_inputs()
        inputs_dict = {"input_ids": input_ids}
        return config, inputs_dict


model_tester = xLSTMModelTester()
(
    config,
    input_ids,
    _,
    sequence_labels,
    token_labels,
    choice_labels,
) = model_tester.prepare_config_and_inputs()


XLSTM_CASES = [
    [
        "xLSTMModel",
        "transformers.xLSTMModel",
        "mindone.transformers.xLSTMModel",
        (config,),
        {},
        (),
        {"input_ids": input_ids},
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "xLSTMForCausalLM",
        "transformers.xLSTMForCausalLM",
        "mindone.transformers.xLSTMForCausalLM",
        (config,),
        {},
        (),
        {"input_ids": input_ids},
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
        for case in XLSTM_CASES
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
