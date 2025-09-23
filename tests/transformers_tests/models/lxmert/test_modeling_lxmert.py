# coding=utf-8
# Copyright 2018 LXMERT Authors, The Hugging Face Team.
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


import numpy as np
import pytest
import torch
from transformers import LxmertConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]  # not support graph mode yet


class LxmertModelTester:
    def __init__(
        self,
        vocab_size=300,
        hidden_size=28,
        num_attention_heads=2,
        num_labels=2,
        intermediate_size=64,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        num_qa_labels=30,
        num_object_labels=16,
        num_attr_labels=4,
        num_visual_features=10,
        l_layers=2,
        x_layers=1,
        r_layers=1,
        visual_feat_dim=128,
        visual_pos_dim=4,
        visual_loss_normalizer=6.67,
        seq_length=20,
        batch_size=4,
        is_training=True,
        task_matched=True,
        task_mask_lm=True,
        task_obj_predict=True,
        task_qa=True,
        visual_obj_loss=True,
        visual_attr_loss=True,
        visual_feat_loss=True,
        use_token_type_ids=True,
        use_lang_mask=True,
        output_attentions=False,
        output_hidden_states=False,
        scope=None,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_labels = num_labels
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.num_qa_labels = num_qa_labels
        self.num_object_labels = num_object_labels
        self.num_attr_labels = num_attr_labels
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers
        self.visual_feat_dim = visual_feat_dim
        self.visual_pos_dim = visual_pos_dim
        self.visual_loss_normalizer = visual_loss_normalizer
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.is_training = is_training
        self.use_lang_mask = use_lang_mask
        self.task_matched = task_matched
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_qa = task_qa
        self.visual_obj_loss = visual_obj_loss
        self.visual_attr_loss = visual_attr_loss
        self.visual_feat_loss = visual_feat_loss
        self.num_visual_features = num_visual_features
        self.use_token_type_ids = use_token_type_ids
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.scope = scope
        self.num_hidden_layers = {"vision": r_layers, "cross_encoder": x_layers, "language": l_layers}

    def prepare_config_and_inputs(self):
        output_attentions = self.output_attentions
        input_ids = ids_numpy([self.batch_size, self.seq_length], vocab_size=self.vocab_size)
        visual_feats = np.random.rand(self.batch_size, self.num_visual_features, self.visual_feat_dim)
        bounding_boxes = np.random.rand(self.batch_size, self.num_visual_features, 4)

        input_mask = None
        if self.use_lang_mask:
            input_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)
        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)
        obj_labels = None
        if self.task_obj_predict:
            obj_labels = {}
        if self.visual_attr_loss and self.task_obj_predict:
            obj_labels["attr"] = (
                ids_numpy([self.batch_size, self.num_visual_features], self.num_attr_labels),
                ids_numpy([self.batch_size, self.num_visual_features], self.num_attr_labels),
            )
        if self.visual_feat_loss and self.task_obj_predict:
            obj_labels["feat"] = (
                ids_numpy([self.batch_size, self.num_visual_features, self.visual_feat_dim], self.num_visual_features),
                ids_numpy([self.batch_size, self.num_visual_features], self.num_visual_features),
            )
        if self.visual_obj_loss and self.task_obj_predict:
            obj_labels["obj"] = (
                ids_numpy([self.batch_size, self.num_visual_features], self.num_object_labels),
                ids_numpy([self.batch_size, self.num_visual_features], self.num_object_labels),
            )
        ans = None
        if self.task_qa:
            ans = ids_numpy([self.batch_size], self.num_qa_labels)
        masked_lm_labels = None
        if self.task_mask_lm:
            masked_lm_labels = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        matched_label = None
        if self.task_matched:
            matched_label = ids_numpy([self.batch_size], self.num_labels)

        config = self.get_config()

        return (
            config,
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids,
            input_mask,
            obj_labels,
            masked_lm_labels,
            matched_label,
            ans,
            output_attentions,
        )

    def get_config(self):
        return LxmertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_attention_heads=self.num_attention_heads,
            num_labels=self.num_labels,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            layer_norm_eps=self.layer_norm_eps,
            pad_token_id=self.pad_token_id,
            num_qa_labels=self.num_qa_labels,
            num_object_labels=self.num_object_labels,
            num_attr_labels=self.num_attr_labels,
            l_layers=self.l_layers,
            x_layers=self.x_layers,
            r_layers=self.r_layers,
            visual_feat_dim=self.visual_feat_dim,
            visual_pos_dim=self.visual_pos_dim,
            visual_loss_normalizer=self.visual_loss_normalizer,
            task_matched=self.task_matched,
            task_mask_lm=self.task_mask_lm,
            task_obj_predict=self.task_obj_predict,
            task_qa=self.task_qa,
            visual_obj_loss=self.visual_obj_loss,
            visual_attr_loss=self.visual_attr_loss,
            visual_feat_loss=self.visual_feat_loss,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

    def prepare_config_and_inputs_for_common(self, return_obj_labels=False):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            visual_feats,
            bounding_boxes,
            token_type_ids,
            input_mask,
            obj_labels,
            masked_lm_labels,
            matched_label,
            ans,
            output_attentions,
        ) = config_and_inputs

        inputs_dict = {
            "input_ids": input_ids,
            "visual_feats": visual_feats,
            "visual_pos": bounding_boxes,
            "token_type_ids": token_type_ids,
            "attention_mask": input_mask,
        }

        if return_obj_labels:
            inputs_dict["obj_labels"] = obj_labels
        else:
            config.task_obj_predict = False

        return config, inputs_dict


model_tester = LxmertModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


_CASES = [
    [
        "LxmertModel",
        "transformers.LxmertModel",
        "mindone.transformers.LxmertModel",
        (config,),
        {},
        (),
        inputs_dict,
        {"pooled_output": "pooled_output"},
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
    [case + [dtype] + [mode] for case in _CASES for dtype in DTYPE_AND_THRESHOLDS.keys() for mode in MODES],
)
def test_named_modules(
    name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype, mode
):
    ms.set_context(mode=mode)

    (pt_model, ms_model, pt_dtype, ms_dtype) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
    pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
        pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
    )

    with torch.no_grad():
        pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
    ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)

    if outputs_map:
        pt_outputs_n = []
        ms_outputs_n = []
        for pt_key, ms_idx in outputs_map.items():
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = getattr(ms_outputs, ms_idx)
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
