# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
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
from transformers import RagConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

from ..bart.test_modeling_bart import BartModelTester
from ..dpr.test_modeling_dpr import DPRModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class RagModelTester:
    def __init__(self, retrieval_vector_size=32, n_docs=3, max_combined_length=16):
        self.retrieval_vector_size = retrieval_vector_size
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length

    def config_and_inputs(self):
        question_encoder_tester = DPRModelTester()
        dpr_config_and_inputs = question_encoder_tester.prepare_config_and_inputs()
        generator_tester = BartModelTester()
        bart_config_and_inputs = generator_tester.prepare_config_and_inputs_for_common()

        context_input_ids = ids_numpy(
            [self.n_docs * question_encoder_tester.batch_size, self.max_combined_length],
            question_encoder_tester.vocab_size,
        )
        context_attention_mask = np.ones_like(context_input_ids)
        doc_scores = floats_numpy([question_encoder_tester.batch_size, self.n_docs])

        (question_encoder_config, input_ids, _, input_mask, _, _, _) = dpr_config_and_inputs
        (generator_config, decoder_input_ids, _, decoder_attention_mask, _) = bart_config_and_inputs

        config = RagConfig.from_question_encoder_generator_configs(
            question_encoder_config,
            generator_config,
            n_docs=self.n_docs,
            retrieval_vector_size=self.retrieval_vector_size,
            max_combined_length=self.max_combined_length,
        )

        return (
            config,
            input_ids,
            input_mask,
            decoder_input_ids,
            decoder_attention_mask,
            context_input_ids,
            context_attention_mask,
            doc_scores,
        )


model_tester = RagModelTester()
(
    config,
    input_ids,
    input_mask,
    decoder_input_ids,
    decoder_attention_mask,
    context_input_ids,
    context_attention_mask,
    doc_scores,
) = model_tester.config_and_inputs()
RAG_CASES = [
    [
        "RagModel",
        "transformers.RagModel",
        "mindone.transformers.RagModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "context_input_ids": context_input_ids,
            "context_attention_mask": context_attention_mask,
            "doc_scores": doc_scores,
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
        for case in RAG_CASES
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
