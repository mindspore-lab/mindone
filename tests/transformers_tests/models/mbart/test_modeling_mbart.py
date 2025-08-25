# Copyright 2021, The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Mindspore BART model."""
import numpy as np
import pytest
import torch
from transformers import MBartConfig

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules

from ..modeling_common import ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 6e-3}
MODES = [0, 1]


def prepare_mbart_inputs_dict(
    config,
    input_ids,
    decoder_input_ids=None,
    attention_mask=None,
    head_mask=None,
    decoder_head_mask=None,
    cross_attn_head_mask=None,
):
    if attention_mask is None:
        attention_mask = np.not_equal(input_ids, config.pad_token_id)
    if head_mask is None:
        head_mask = random_attention_mask([config.decoder_layers, config.decoder_attention_heads])
    if decoder_head_mask is None:
        decoder_head_mask = random_attention_mask([config.decoder_layers, config.decoder_attention_heads])
    if cross_attn_head_mask is None:
        cross_attn_head_mask = random_attention_mask([config.decoder_layers, config.decoder_attention_heads])
    return {
        "input_ids": input_ids,
        "decoder_input_ids": decoder_input_ids,
        "attention_mask": attention_mask,
        "head_mask": head_mask,
        "decoder_head_mask": decoder_head_mask,
        "cross_attn_head_mask": cross_attn_head_mask,
    }


class MBartModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=True,
        use_labels=False,
        vocab_size=99,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=4,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        eos_token_id=2,
        pad_token_id=1,
        bos_token_id=0,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
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
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        input_ids[:, -1] = self.eos_token_id  # Eos Token

        decoder_input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        config = self.get_config()
        config._attn_implementation = "eager"
        inputs_dict = prepare_mbart_inputs_dict(config, input_ids, decoder_input_ids)
        return config, inputs_dict

    def get_config(self):
        return MBartConfig(
            vocab_size=self.vocab_size,
            d_model=self.hidden_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            encoder_ffn_dim=self.intermediate_size,
            decoder_ffn_dim=self.intermediate_size,
            dropout=self.hidden_dropout_prob,
            attention_dropout=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            eos_token_id=self.eos_token_id,
            bos_token_id=self.bos_token_id,
            pad_token_id=self.pad_token_id,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return (
            config,
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            inputs_dict["head_mask"],
        )


model_tester = MBartModelTester()
(
    config,
    input_ids,
    attention_mask,
    head_mask,
) = model_tester.prepare_config_and_inputs_for_common()


MBart_CASES = [
    [
        "MBartModel",
        "transformers.MBartModel",
        "mindone.transformers.MBartModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "head_mask": head_mask,
        },
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "MBartForConditionalGeneration",
        "transformers.MBartForConditionalGeneration",
        "mindone.transformers.MBartForConditionalGeneration",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
            "head_mask": head_mask,
        },
        {
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "MBartForSequenceClassification",
        "transformers.MBartForSequenceClassification",
        "mindone.transformers.MBartForSequenceClassification",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
        },
        {
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "MBartForQuestionAnswering",
        "transformers.MBartForQuestionAnswering",
        "mindone.transformers.MBartForQuestionAnswering",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
        },
        {
            "encoder_last_hidden_state": 3,
        },
    ],
]


# transformers need >= 4.41.2
@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype",
    [
        case
        + [
            dtype,
        ]
        for case in MBart_CASES
        for dtype in DTYPE_AND_THRESHOLDS.keys()
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
):
    (
        pt_model,
        ms_model,
        pt_dtype,
        ms_dtype,
    ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

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
