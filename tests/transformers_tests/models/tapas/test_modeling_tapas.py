# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
from transformers import TapasConfig

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
# graph mode is not supported in this model
MODES = [1]


class TapasModelTester:
    """You can also import this e.g from .test_modeling_tapas import TapasModelTester"""

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
        initializer_range=0.02,
        max_position_embeddings=512,
        type_vocab_sizes=[3, 256, 256, 2, 256, 256, 10],
        type_sequence_label_size=2,
        positive_weight=10.0,
        num_aggregation_labels=4,
        num_labels=2,
        aggregation_loss_importance=0.8,
        use_answer_as_supervision=True,
        answer_loss_importance=0.001,
        use_normalized_answer_loss=False,
        huber_loss_delta=25.0,
        temperature=1.0,
        agg_temperature=1.0,
        use_gumbel_for_cells=False,
        use_gumbel_for_agg=False,
        average_approximation_function="ratio",
        cell_selection_preference=0.5,
        answer_loss_cutoff=100,
        max_num_rows=64,
        max_num_columns=32,
        average_logits_per_cell=True,
        select_one_column=True,
        allow_empty_column_selection=False,
        init_cell_selection_weights_to_zero=True,
        reset_position_index_per_cell=True,
        disable_per_token_loss=False,
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
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_sizes = type_vocab_sizes
        self.type_sequence_label_size = type_sequence_label_size
        self.positive_weight = positive_weight
        self.num_aggregation_labels = num_aggregation_labels
        self.num_labels = num_labels
        self.aggregation_loss_importance = aggregation_loss_importance
        self.use_answer_as_supervision = use_answer_as_supervision
        self.answer_loss_importance = answer_loss_importance
        self.use_normalized_answer_loss = use_normalized_answer_loss
        self.huber_loss_delta = huber_loss_delta
        self.temperature = temperature
        self.agg_temperature = agg_temperature
        self.use_gumbel_for_cells = use_gumbel_for_cells
        self.use_gumbel_for_agg = use_gumbel_for_agg
        self.average_approximation_function = average_approximation_function
        self.cell_selection_preference = cell_selection_preference
        self.answer_loss_cutoff = answer_loss_cutoff
        self.max_num_rows = max_num_rows
        self.max_num_columns = max_num_columns
        self.average_logits_per_cell = average_logits_per_cell
        self.select_one_column = select_one_column
        self.allow_empty_column_selection = allow_empty_column_selection
        self.init_cell_selection_weights_to_zero = init_cell_selection_weights_to_zero
        self.reset_position_index_per_cell = reset_position_index_per_cell
        self.disable_per_token_loss = disable_per_token_loss
        self.scope = scope

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        token_type_ids = []
        for type_vocab_size in self.type_vocab_sizes:
            token_type_ids.append(ids_numpy(shape=[self.batch_size, self.seq_length], vocab_size=type_vocab_size))
        token_type_ids = np.stack(token_type_ids, axis=2)

        sequence_labels = None
        token_labels = None
        labels = None
        numeric_values = None
        numeric_values_scale = None
        float_answer = None
        aggregation_labels = None
        if self.use_labels:
            sequence_labels = ids_numpy([self.batch_size], self.type_sequence_label_size)
            token_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)
            labels = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)
            numeric_values = floats_numpy([self.batch_size, self.seq_length])
            numeric_values_scale = floats_numpy([self.batch_size, self.seq_length])
            float_answer = floats_numpy([self.batch_size])
            aggregation_labels = floats_numpy([self.batch_size], self.num_aggregation_labels)

        config = self.get_config()

        return (
            config,
            input_ids,
            input_mask,
            token_type_ids,
            sequence_labels,
            token_labels,
            labels,
            numeric_values,
            numeric_values_scale,
            float_answer,
            aggregation_labels,
        )

    def get_config(self):
        return TapasConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_sizes=self.type_vocab_sizes,
            initializer_range=self.initializer_range,
            positive_weight=self.positive_weight,
            num_aggregation_labels=self.num_aggregation_labels,
            num_labels=self.num_labels,
            aggregation_loss_importance=self.aggregation_loss_importance,
            use_answer_as_supervision=self.use_answer_as_supervision,
            answer_loss_importance=self.answer_loss_importance,
            use_normalized_answer_loss=self.use_normalized_answer_loss,
            huber_loss_delta=self.huber_loss_delta,
            temperature=self.temperature,
            agg_temperature=self.agg_temperature,
            use_gumbel_for_cells=self.use_gumbel_for_cells,
            use_gumbel_for_agg=self.use_gumbel_for_agg,
            average_approximation_function=self.average_approximation_function,
            cell_selection_preference=self.cell_selection_preference,
            answer_loss_cutoff=self.answer_loss_cutoff,
            max_num_rows=self.max_num_rows,
            max_num_columns=self.max_num_columns,
            average_logits_per_cell=self.average_logits_per_cell,
            select_one_column=self.select_one_column,
            allow_empty_column_selection=self.allow_empty_column_selection,
            init_cell_selection_weights_to_zero=self.init_cell_selection_weights_to_zero,
            reset_position_index_per_cell=self.reset_position_index_per_cell,
            disable_per_token_loss=self.disable_per_token_loss,
        )


model_tester = TapasModelTester()
(
    config,
    input_ids,
    input_mask,
    token_type_ids,
    sequence_labels,
    token_labels,
    labels,
    numeric_values,
    numeric_values_scale,
    float_answer,
    aggregation_labels,
) = model_tester.prepare_config_and_inputs()
TAPAS_CASES = [
    [
        "TapasModel",
        "transformers.TapasModel",
        "mindone.transformers.TapasModel",
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
        for case in TAPAS_CASES
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
