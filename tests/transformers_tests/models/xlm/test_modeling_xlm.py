# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
from transformers.models.xlm.configuration_xlm import XLMConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class XLMModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_input_lengths=True,
        use_token_type_ids=True,
        use_labels=True,
        gelu_activation=True,
        sinusoidal_embeddings=False,
        causal=False,
        asm=False,
        n_langs=2,
        vocab_size=99,
        n_special=0,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=2,
        num_choices=4,
        summary_type="last",
        use_proj=True,
        scope=None,
        bos_token_id=0,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_lengths = use_input_lengths
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.gelu_activation = gelu_activation
        self.sinusoidal_embeddings = sinusoidal_embeddings
        self.causal = causal
        self.asm = asm
        self.n_langs = n_langs
        self.vocab_size = vocab_size
        self.n_special = n_special
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.summary_type = summary_type
        self.use_proj = use_proj
        self.scope = scope
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length])

        input_lengths = None
        if self.use_input_lengths:
            input_lengths = (
                ids_numpy([self.batch_size], vocab_size=2) + self.seq_length - 2
            )  # small variation of seq_length

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.n_langs)

        sequence_labels = None
        token_labels = None
        is_impossible_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            is_impossible_labels = floats_numpy([self.batch_size], 2)
            choice_labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config()

        return (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            choice_labels,
            input_mask,
        )

    def get_config(self):
        return XLMConfig(
            vocab_size=self.vocab_size,
            n_special=self.n_special,
            emb_dim=self.hidden_size,
            n_layers=self.num_hidden_layers,
            n_heads=self.num_attention_heads,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            gelu_activation=self.gelu_activation,
            sinusoidal_embeddings=self.sinusoidal_embeddings,
            asm=self.asm,
            causal=self.causal,
            n_langs=self.n_langs,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            summary_type=self.summary_type,
            use_proj=self.use_proj,
            num_labels=self.num_labels,
            bos_token_id=self.bos_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            token_type_ids,
            input_lengths,
            sequence_labels,
            token_labels,
            is_impossible_labels,
            choice_labels,
            input_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "token_type_ids": token_type_ids, "lengths": input_lengths}
        return config, inputs_dict


model_tester = XLMModelTester()
(
    config,
    inputs_dict,
) = model_tester.prepare_config_and_inputs_for_common()


XLM_CASES = [
    [
        "XLMModel",
        "transformers.XLMModel",
        "mindone.transformers.XLMModel",
        (config,),
        {},
        (),
        {
            "input_ids": inputs_dict["input_ids"],
            "token_type_ids": inputs_dict["token_type_ids"],
            "lengths": inputs_dict["lengths"],
        },
        {
            "last_hidden_state": 0,
        },
    ],
]


# transformers need >= 4.41.2
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
        for case in XLM_CASES
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
    if mode == 0:
        ms_inputs_kwargs.update({"return_dict": False})
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
