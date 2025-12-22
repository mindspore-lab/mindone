# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the MindSpore Blt model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import BltConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)

from ...causal_lm_tester import CausalLMModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class BltModelTester(CausalLMModelTester):
    config_class = BltConfig

    def __init__(
        self,
        parent,
        ignore_index=-100,
        seq_length=7,
        is_training=True,
    ):
        super().__init__(parent)
        self.parent = parent
        self.ignore_index = ignore_index
        self.seq_length = seq_length
        self.is_training = is_training
        self.batch_size = 1

        # Common parameters for all configs
        self.hidden_size = 16
        self.num_hidden_layers = 1
        self.num_attention_heads = 2
        self.num_key_value_heads = 2
        self.intermediate_size = 32
        self.hidden_act = "silu"
        self.max_position_embeddings = 32
        self.vocab_size = 32
        self.rope_theta = 500000.0
        self.rope_scaling = {"rope_type": "default"}
        self.rms_norm_eps = 1e-5
        self.dropout = 0.0
        self.encoder_hash_byte_group_size = [2, 3]
        self.encoder_hash_byte_group_vocab = 64
        self.encoder_hash_byte_group_nb_functions = 1
        # Common parameters for all configs
        self.patcher_config = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "dropout": self.dropout,
        }

        self.encoder_config = {
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "dropout": self.dropout,
        }

        self.decoder_config = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "hidden_size_global": self.hidden_size * 2,  # Must match global transformer output size
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "dropout": self.dropout,
        }

        self.global_config = {
            "hidden_size": self.hidden_size * 2,  # Double the hidden size for global transformer
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "rope_scaling": self.rope_scaling,
            "hidden_act": self.hidden_act,
            "rms_norm_eps": self.rms_norm_eps,
            "dropout": self.dropout,
        }

        self.num_hidden_layers = self.encoder_config["num_hidden_layers"]

    def get_config(self):
        config = BltConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            patch_in_forward=False,  # Disable patching for tests
            patch_size=4,
            patching_mode="entropy",
            patching_threshold=1.335442066192627,
            patching_batch_size=1,
            max_patch_length=None,
            cross_attn_k=2,
            encoder_hash_byte_group_size=self.encoder_hash_byte_group_size,
            encoder_hash_byte_group_vocab=self.encoder_hash_byte_group_vocab,
            encoder_hash_byte_group_nb_functions=self.encoder_hash_byte_group_nb_functions,
            patcher_config=self.patcher_config,
            encoder_config=self.encoder_config,
            decoder_config=self.decoder_config,
            global_config=self.global_config,
            rope_scaling=self.rope_scaling,
            tie_word_embeddings=False,
        )

        config.num_attention_heads = config.decoder_config.num_attention_heads
        config.num_hidden_layers = config.encoder_config.num_hidden_layers
        config.hidden_size = config.decoder_config.hidden_size

        return config


model_tester = BltModelTester(parent=None)
(
    config,
    input_ids,
    token_type_ids,
    input_mask,
    sequence_labels,
    token_labels,
    choice_labels,
) = model_tester.prepare_config_and_inputs()
BLT_CASES = [
    [
        "BltModel",
        "transformers.BltModel",
        "mindone.transformers.BltModel",
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
        for case in BLT_CASES
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
