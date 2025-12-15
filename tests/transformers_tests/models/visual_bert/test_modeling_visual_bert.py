# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
from transformers import VisualBertConfig
from transformers.testing_utils import slow

import mindspore as ms
from mindspore import mint

from mindone.transformers import (
    VisualBertForMultipleChoice,
    VisualBertForPreTraining,
    VisualBertForQuestionAnswering,
    VisualBertForVisualReasoning,
)
from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 1e-4, "fp16": 5e-3, "bf16": 1e-2}


class VisualBertModelTester:
    def __init__(
        self,
        parent=None,
        batch_size=13,
        seq_length=7,
        visual_seq_length=5,
        is_training=True,
        use_attention_mask=True,
        use_visual_attention_mask=True,
        use_token_type_ids=True,
        use_visual_token_type_ids=True,
        use_labels=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=37,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        visual_embedding_dim=20,
        type_vocab_size=16,
        type_sequence_label_size=2,
        initializer_range=0.02,
        num_labels=3,
        num_choices=4,
        scope=None,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.visual_seq_length = visual_seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_visual_attention_mask = use_visual_attention_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_visual_token_type_ids = use_visual_token_type_ids
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
        self.visual_embedding_dim = visual_embedding_dim
        self.type_vocab_size = type_vocab_size
        self.type_sequence_label_size = type_sequence_label_size
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

    def get_config(self):
        return VisualBertConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            max_position_embeddings=self.max_position_embeddings,
            type_vocab_size=self.type_vocab_size,
            visual_embedding_dim=self.visual_embedding_dim,
            num_labels=self.num_labels,
            is_decoder=False,
            initializer_range=self.initializer_range,
        )

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        visual_embeds = floats_numpy([self.batch_size, self.visual_seq_length, self.visual_embedding_dim])

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = np.ones((self.batch_size, self.seq_length), dtype=np.int64)

        visual_attention_mask = None
        if self.use_visual_attention_mask:
            visual_attention_mask = np.ones((self.batch_size, self.visual_seq_length), dtype=np.int64)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.seq_length], self.type_vocab_size)

        visual_token_type_ids = None
        if self.use_visual_token_type_ids:
            visual_token_type_ids = ids_numpy([self.batch_size, self.visual_seq_length], self.type_vocab_size)

        config = self.get_config()
        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
        }

    def prepare_config_and_inputs_for_pretraining(self):
        masked_lm_labels = None
        sentence_image_labels = None

        if self.use_labels:
            masked_lm_labels = ids_numpy([self.batch_size, self.seq_length + self.visual_seq_length], self.vocab_size)
            sentence_image_labels = ids_numpy(
                [self.batch_size],
                self.type_sequence_label_size,
            )

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": masked_lm_labels, "sentence_image_labels": sentence_image_labels})

        return config, input_dict

    def prepare_config_and_inputs_for_multiple_choice(self):
        input_ids = ids_numpy([self.batch_size, self.num_choices, self.seq_length], self.vocab_size)
        visual_embeds = floats_numpy(
            [self.batch_size, self.num_choices, self.visual_seq_length, self.visual_embedding_dim]
        )

        attention_mask = None
        if self.use_attention_mask:
            attention_mask = np.ones((self.batch_size, self.num_choices, self.seq_length), dtype=np.int64)

        visual_attention_mask = None
        if self.use_visual_attention_mask:
            visual_attention_mask = np.ones((self.batch_size, self.num_choices, self.visual_seq_length), dtype=np.int64)

        token_type_ids = None
        if self.use_token_type_ids:
            token_type_ids = ids_numpy([self.batch_size, self.num_choices, self.seq_length], self.type_vocab_size)

        visual_token_type_ids = None
        if self.use_visual_token_type_ids:
            visual_token_type_ids = ids_numpy(
                [self.batch_size, self.num_choices, self.visual_seq_length], self.type_vocab_size
            )

        labels = None

        if self.use_labels:
            labels = ids_numpy([self.batch_size], self.num_choices)

        config = self.get_config()
        return config, {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
            "visual_embeds": visual_embeds,
            "visual_token_type_ids": visual_token_type_ids,
            "visual_attention_mask": visual_attention_mask,
            "labels": labels,
        }

    def prepare_config_and_inputs_for_vqa(self):
        vqa_labels = None

        if self.use_labels:
            vqa_labels = floats_numpy([self.batch_size, self.num_labels])

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": vqa_labels})
        return config, input_dict

    def prepare_config_and_inputs_for_nlvr(self):
        nlvr_labels = None

        if self.use_labels:
            nlvr_labels = ids_numpy([self.batch_size], self.num_labels)

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"labels": nlvr_labels})
        return config, input_dict

    def prepare_config_and_inputs_for_flickr(self):
        region_to_phrase_position = np.concatenate(
            (
                ids_numpy([self.batch_size, self.seq_length], self.visual_seq_length),
                np.ones((self.batch_size, self.visual_seq_length), dtype=np.int64) * -1,
            ),
            axis=-1,
        )
        flickr_labels = None
        if self.use_labels:
            flickr_labels = floats_numpy(
                [self.batch_size, self.seq_length + self.visual_seq_length, self.visual_seq_length]
            )

        config, input_dict = self.prepare_config_and_inputs_for_common()

        input_dict.update({"region_to_phrase_position": region_to_phrase_position, "labels": flickr_labels})
        return config, input_dict


_CASES = [
    [
        "VisualBertModel",
        "transformers.VisualBertModel",
        "mindone.transformers.VisualBertModel",
        VisualBertModelTester().prepare_config_and_inputs_for_common(),
        {"last_hidden_state": "last_hidden_state"},
    ],
    [
        "VisualBertForPreTraining",
        "transformers.VisualBertForPreTraining",
        "mindone.transformers.VisualBertForPreTraining",
        VisualBertModelTester().prepare_config_and_inputs_for_pretraining(),
        {"prediction_logits": "prediction_logits"},
    ],
    [
        "VisualBertForMultipleChoice",
        "transformers.VisualBertForMultipleChoice",
        "mindone.transformers.VisualBertForMultipleChoice",
        VisualBertModelTester().prepare_config_and_inputs_for_multiple_choice(),
        {"logits": "logits"},
    ],
    [
        "VisualBertForQuestionAnswering",
        "transformers.VisualBertForQuestionAnswering",
        "mindone.transformers.VisualBertForQuestionAnswering",
        VisualBertModelTester().prepare_config_and_inputs_for_vqa(),
        {"logits": "logits"},
    ],
    [
        "VisualBertForVisualReasoning",
        "transformers.VisualBertForVisualReasoning",
        "mindone.transformers.VisualBertForVisualReasoning",
        VisualBertModelTester().prepare_config_and_inputs_for_nlvr(),
        {"logits": "logits"},
    ],
    [
        "VisualBertForRegionToPhraseAlignment",
        "transformers.VisualBertForRegionToPhraseAlignment",
        "mindone.transformers.VisualBertForRegionToPhraseAlignment",
        VisualBertModelTester().prepare_config_and_inputs_for_flickr(),
        {"logits": "logits"},
    ],
]

_CASES = [
    [module, pt_module, ms_module, (config,), {}, (), inputs_dict, outputs]
    for module, pt_module, ms_module, (config, inputs_dict), outputs in _CASES
]


@pytest.mark.parametrize("dtype", DTYPE_AND_THRESHOLDS.keys())
@pytest.mark.parametrize("name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map", _CASES)
def test_named_modules(
    name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
):
    pt_model, ms_model, pt_dtype, ms_dtype = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
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


@slow
def test_inference_vqa_coco_pre():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model = VisualBertForPreTraining.from_pretrained("uclanlp/visualbert-vqa-coco-pre", revision="refs/pr/1")

    input_ids = ms.tensor([1, 2, 3, 4, 5, 6], dtype=ms.int64).reshape(1, -1)
    token_type_ids = ms.tensor([0, 0, 0, 1, 1, 1], dtype=ms.int64).reshape(1, -1)
    visual_embeds = mint.ones(size=(1, 10, 2048), dtype=ms.float32) * 0.5
    visual_token_type_ids = mint.ones(size=(1, 10), dtype=ms.int64)
    attention_mask = ms.tensor([1] * 6).reshape(1, -1)
    visual_attention_mask = ms.tensor([1] * 10).reshape(1, -1)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
    )

    vocab_size = 30522
    assert output.prediction_logits.shape == (1, 16, vocab_size)
    expected_slice = np.array([[[-5.1858, -5.1903, -4.9142], [-6.2214, -5.9238, -5.8381], [-6.3027, -5.9939, -5.9297]]])
    diffs = compute_diffs(expected_slice, output.prediction_logits[:, :3, :3])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"

    assert output.seq_relationship_logits.shape == (1, 2)
    expected_slice_2 = np.array([[0.7393, 0.1754]])
    diffs = compute_diffs(expected_slice_2, output.seq_relationship_logits)
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"


@slow
def test_inference_vqa():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model = VisualBertForQuestionAnswering.from_pretrained("uclanlp/visualbert-vqa", revision="refs/pr/3")

    input_ids = ms.tensor([1, 2, 3, 4, 5, 6], dtype=ms.int64).reshape(1, -1)
    token_type_ids = ms.tensor([0, 0, 0, 1, 1, 1], dtype=ms.int64).reshape(1, -1)
    visual_embeds = mint.ones(size=(1, 10, 2048), dtype=ms.float32) * 0.5
    visual_token_type_ids = mint.ones(size=(1, 10), dtype=ms.int64)
    attention_mask = ms.tensor([1] * 6).reshape(1, -1)
    visual_attention_mask = ms.tensor([1] * 10).reshape(1, -1)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
    )

    assert output.logits.shape == (1, 3129)

    expected_slice = np.array(
        [[-8.9898, 3.0803, -1.8016, 2.4542, -8.3420, -2.0224, -3.3124, -4.4139, -3.1491, -3.8997]]
    )
    diffs = compute_diffs(expected_slice, output.logits[:, :10])
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"


@slow
def test_inference_nlvr():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model = VisualBertForVisualReasoning.from_pretrained("uclanlp/visualbert-nlvr2", revision="refs/pr/1")

    input_ids = ms.tensor([1, 2, 3, 4, 5, 6], dtype=ms.int64).reshape(1, -1)
    token_type_ids = ms.tensor([0, 0, 0, 1, 1, 1], dtype=ms.int64).reshape(1, -1)
    visual_embeds = mint.ones(size=(1, 10, 1024), dtype=ms.float32) * 0.5
    visual_token_type_ids = mint.ones(size=(1, 10), dtype=ms.int64)
    attention_mask = ms.tensor([1] * 6).reshape(1, -1)
    visual_attention_mask = ms.tensor([1] * 10).reshape(1, -1)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
    )

    assert output.logits.shape == (1, 2)

    expected_slice = np.array([[-1.1436, 0.8900]])
    diffs = compute_diffs(expected_slice, output.logits)
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"


@slow
def test_inference_vcr():
    THRESHOLD = DTYPE_AND_THRESHOLDS["fp32"]

    model = VisualBertForMultipleChoice.from_pretrained("uclanlp/visualbert-vcr", revision="refs/pr/2")

    input_ids = ms.tensor([[[1, 2, 3, 4, 5, 6] for i in range(4)]], dtype=ms.int64)
    attention_mask = mint.ones_like(input_ids)
    token_type_ids = mint.ones_like(input_ids)

    visual_embeds = mint.ones(size=(1, 4, 10, 512), dtype=ms.float32) * 0.5
    visual_token_type_ids = mint.ones(size=(1, 4, 10), dtype=ms.int64)
    visual_attention_mask = mint.ones_like(visual_token_type_ids)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        visual_embeds=visual_embeds,
        visual_attention_mask=visual_attention_mask,
        visual_token_type_ids=visual_token_type_ids,
    )

    assert output.logits.shape == (1, 4)

    expected_slice = np.array([[-7.7697, -7.7697, -7.7697, -7.7697]])
    diffs = compute_diffs(expected_slice, output.logits)
    assert (np.array(diffs) < THRESHOLD).all(), f"Output difference exceeds the threshold: {diffs} > {THRESHOLD}"
