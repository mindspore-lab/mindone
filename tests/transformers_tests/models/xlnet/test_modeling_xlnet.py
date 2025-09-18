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
from transformers.models.xlnet.configuration_xlnet import XLNetConfig

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


class XLNetModelTester:
    def __init__(
        self,
        batch_size=14,
        seq_length=7,
        mem_len=10,
        clamp_len=-1,
        reuse_len=15,
        is_training=True,
        use_labels=True,
        vocab_size=99,
        cutoffs=[10, 50, 80],
        hidden_size=32,
        num_attention_heads=4,
        d_inner=128,
        num_hidden_layers=2,
        type_sequence_label_size=2,
        untie_r=True,
        bi_data=False,
        same_length=False,
        initializer_range=0.05,
        seed=1,
        type_vocab_size=2,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=5,
        num_choices=4,
    ):
        self.batch_size = 14
        self.seq_length = 7
        self.mem_len = 10
        # self.key_len = seq_length + mem_len
        self.clamp_len = -1
        self.reuse_len = 15
        self.is_training = True
        self.use_labels = True
        self.vocab_size = 99
        self.cutoffs = [10, 50, 80]
        self.hidden_size = 32
        self.num_attention_heads = 4
        self.d_inner = 128
        self.num_hidden_layers = 5
        self.type_sequence_label_size = 2
        self.untie_r = True
        self.bi_data = False
        self.same_length = False
        self.initializer_range = 0.05
        self.seed = 1
        self.type_vocab_size = 2
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token_id = 5
        self.num_choices = 4

    def prepare_config_and_inputs(self):
        input_ids_1 = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        input_ids_2 = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        segment_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)
        input_mask = random_attention_mask([self.batch_size, self.seq_length])

        input_ids_q = ids_numpy([self.batch_size, self.seq_length + 1], self.vocab_size)
        perm_mask = np.zeros(
            (self.batch_size, self.seq_length + 1, self.seq_length + 1),
        )
        perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        target_mapping = np.zeros(
            (self.batch_size, 1, self.seq_length + 1),
        )
        target_mapping[:, 0, -1] = 1.0  # predict last token

        sequence_labels = None
        lm_labels = None
        is_impossible_labels = None
        token_labels = None
        if self.use_labels:
            lm_labels = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            is_impossible_labels = floats_numpy([self.batch_size], 2)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        config = self.get_config()

        return (
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
            token_labels,
        )

    def get_config(self):
        return XLNetConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            n_head=self.num_attention_heads,
            d_inner=self.d_inner,
            n_layer=self.num_hidden_layers,
            untie_r=self.untie_r,
            mem_len=self.mem_len,
            clamp_len=self.clamp_len,
            same_length=self.same_length,
            reuse_len=self.reuse_len,
            bi_data=self.bi_data,
            initializer_range=self.initializer_range,
            num_labels=self.type_sequence_label_size,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids_1,
            input_ids_2,
            input_ids_q,
            perm_mask,
            input_mask,
            target_mapping,
            segment_ids,
            lm_labels,
            sequence_labels,
            is_impossible_labels,
            token_labels,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids_1}
        return config, inputs_dict


model_tester = XLNetModelTester()
(config, inputs_dict) = model_tester.prepare_config_and_inputs_for_common()


XLNET_CASES = [
    [
        "XLNetModel",
        "transformers.XLNetModel",
        "mindone.transformers.XLNetModel",
        (config,),
        {},
        (),
        {
            "input_ids": inputs_dict["input_ids"],
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
        for case in XLNET_CASES
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
