"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/falcon_h1/test_modeling_falcon_h1.py."""

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
from mindone.transformers.models.falcon_h1 import FalconH1Config

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [1]  # 0: graph mode, 1: pynative mode


class FalconH1ModelTester:
    config_class = FalconH1Config

    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        type_vocab_size=16,
        type_sequence_label_size=2,
        num_labels=3,
        num_choices=4,
        # config
        vocab_size=99,
        hidden_size=32,
        intermediate_size=37,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        mamba_d_conv=4,
        mamba_d_ssm=16,
        mamba_expand=2,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_n_heads=4,
        mamba_d_head=8,
        mamba_chunk_size=64,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_rms_norm=True,
        mamba_norm_before_gate=True,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
        key_multiplier=1.0,
        hidden_act="silu",
        max_position_embeddings=512,
        initializer_range=0.02,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        mlp_bias=False,
        output_attentions=False,
        output_hidden_states=False,
        num_logits_to_keep=None,
        mlp_multipliers=1.0,
        ssm_multipliers=1.0,
        ssm_in_multiplier=1.0,
        attention_in_multiplier=1.0,
        ssm_out_multiplier=1.0,
        attention_out_multiplier=1.0,
        embedding_multiplier=1.0,
        lm_head_multiplier=1.0,
        projectors_bias=False,
        rope_scaling=None,
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
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.mamba_d_conv = mamba_d_conv
        self.mamba_d_ssm = mamba_d_ssm
        self.mamba_expand = mamba_expand
        self.mamba_n_groups = mamba_n_groups
        self.mamba_d_state = mamba_d_state
        self.mamba_n_heads = mamba_n_heads
        self.mamba_d_head = mamba_d_head
        self.mamba_chunk_size = mamba_chunk_size
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.mamba_rms_norm = mamba_rms_norm
        self.mamba_norm_before_gate = mamba_norm_before_gate
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.key_multiplier = key_multiplier
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.mlp_bias = mlp_bias
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.num_logits_to_keep = num_logits_to_keep
        self.mlp_multipliers = mlp_multipliers
        self.ssm_multipliers = ssm_multipliers
        self.ssm_in_multiplier = ssm_in_multiplier
        self.attention_in_multiplier = attention_in_multiplier
        self.ssm_out_multiplier = ssm_out_multiplier
        self.attention_out_multiplier = attention_out_multiplier
        self.embedding_multiplier = embedding_multiplier
        self.lm_head_multiplier = lm_head_multiplier
        self.projectors_bias = projectors_bias
        self.rope_scaling = rope_scaling
        self.head_dim = self.hidden_size // self.num_attention_heads

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = np.tril(np.ones_like(input_ids))

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

        return config, input_ids, token_type_ids, input_mask, sequence_labels, token_labels, choice_labels

    def get_config(self):
        return self.config_class(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            mamba_d_conv=self.mamba_d_conv,
            mamba_d_ssm=self.mamba_d_ssm,
            mamba_expand=self.mamba_expand,
            mamba_n_groups=self.mamba_n_groups,
            mamba_d_state=self.mamba_d_state,
            mamba_n_heads=self.mamba_n_heads,
            mamba_d_head=self.mamba_d_head,
            mamba_chunk_size=self.mamba_chunk_size,
            mamba_conv_bias=self.mamba_conv_bias,
            mamba_proj_bias=self.mamba_proj_bias,
            mamba_rms_norm=self.mamba_rms_norm,
            mamba_norm_before_gate=self.mamba_norm_before_gate,
            rms_norm_eps=self.rms_norm_eps,
            attention_dropout=self.attention_dropout,
            attention_bias=self.attention_bias,
            key_multiplier=self.key_multiplier,
            hidden_act=self.hidden_act,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            mlp_bias=self.mlp_bias,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
            num_logits_to_keep=self.num_logits_to_keep,
            mlp_multipliers=self.mlp_multipliers,
            ssm_multipliers=self.ssm_multipliers,
            ssm_in_multiplier=self.ssm_in_multiplier,
            attention_in_multiplier=self.attention_in_multiplier,
            ssm_out_multiplier=self.ssm_out_multiplier,
            attention_out_multiplier=self.attention_out_multiplier,
            embedding_multiplier=self.embedding_multiplier,
            lm_head_multiplier=self.lm_head_multiplier,
            projectors_bias=self.projectors_bias,
            rope_scaling=self.rope_scaling,
        )


model_tester = FalconH1ModelTester()
(
    config,
    input_ids,
    token_type_ids,
    input_mask,
    sequence_labels,
    token_labels,
    choice_labels,
) = model_tester.prepare_config_and_inputs()


FALCON_H1_CASES = [
    [
        "FalconH1Model",
        "transformers.FalconH1Model",
        "mindone.transformers.FalconH1Model",
        (config,),
        {},
        (input_ids,),
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
        for case in FALCON_H1_CASES
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