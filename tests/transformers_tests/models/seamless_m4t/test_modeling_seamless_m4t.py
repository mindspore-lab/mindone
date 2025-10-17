"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/albert/test_modeling_albert.py."""

# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Testing suite for the Mindspore SeamlessM4T model."""

import inspect

import numpy as np
import pytest
import torch
from transformers import SeamlessM4TConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class SeamlessM4TModelTester:
    def __init__(
        self,
        input_modality="speech",
        batch_size=2,
        seq_length=4,
        is_training=True,
        use_input_mask=True,
        use_token_type_ids=True,
        use_labels=True,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        max_new_tokens=None,
        num_labels=3,
        num_choices=4,
        scope=None,
        vocab_size=20,
        t2u_vocab_size=20,
        hidden_size=6,
        num_hidden_layers=2,
        intermediate_size=6,
        max_position_embeddings=256,
        encoder_layers=2,
        decoder_layers=2,
        encoder_ffn_dim=6,
        decoder_ffn_dim=6,
        t2u_encoder_layers=2,
        t2u_decoder_layers=2,
        t2u_encoder_ffn_dim=6,
        t2u_decoder_ffn_dim=6,
        num_heads=2,
        vocoder_num_spkrs=5,
        vocoder_num_langs=5,
        upsample_initial_channel=32,
        unit_embed_dim=25,
        spkr_embed_dim=6,
        lang_embed_dim=6,
        num_conv_pos_embeddings=8,
        unit_hifi_gan_vocab_size=20,
        t2u_num_langs=0,
        t2u_max_new_tokens=25,
        t2u_offset_tgt_lang=0,
        vocoder_offset=0,
    ):
        self.input_modality = input_modality

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.num_labels = num_labels
        self.num_choices = num_choices
        self.scope = scope

        self.vocab_size = vocab_size
        self.t2u_vocab_size = t2u_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim
        self.t2u_encoder_layers = t2u_encoder_layers
        self.t2u_decoder_layers = t2u_decoder_layers
        self.t2u_encoder_ffn_dim = t2u_encoder_ffn_dim
        self.t2u_decoder_ffn_dim = t2u_decoder_ffn_dim
        self.num_heads = num_heads
        self.num_attention_heads = num_heads

        self.vocoder_num_spkrs = vocoder_num_spkrs
        self.vocoder_num_langs = vocoder_num_langs
        self.upsample_initial_channel = upsample_initial_channel
        self.unit_embed_dim = unit_embed_dim
        self.spkr_embed_dim = spkr_embed_dim
        self.num_conv_pos_embeddings = num_conv_pos_embeddings
        self.lang_embed_dim = lang_embed_dim

        self.max_new_tokens = max_new_tokens

        self.unit_hifi_gan_vocab_size = unit_hifi_gan_vocab_size
        self.t2u_num_langs = t2u_num_langs
        self.t2u_max_new_tokens = t2u_max_new_tokens
        self.t2u_offset_tgt_lang = t2u_offset_tgt_lang
        self.vocoder_offset = vocoder_offset

    def prepare_config_and_inputs(self):
        if self.input_modality == "text":
            inputs = ids_numpy([self.batch_size, self.seq_length], self.vocab_size - 1)
        else:
            inputs = ids_numpy([self.batch_size, self.seq_length, 160], self.vocab_size - 1)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        decoder_input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size - 1)

        lm_labels = ids_numpy([self.batch_size, self.seq_length], self.num_labels)

        config = self.get_config()

        return config, inputs, decoder_input_ids, input_mask, lm_labels

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            decoder_input_ids,
            input_mask,
            lm_labels,
        ) = config_and_inputs

        input_name = "input_ids" if self.input_modality == "text" else "input_features"

        inputs_dict = {
            input_name: input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
            "labels": lm_labels,
        }
        return config, inputs_dict

    def get_config(self):
        return SeamlessM4TConfig(
            hidden_act=self.hidden_act,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attention_probs_dropout_prob=self.attention_probs_dropout_prob,
            initializer_range=self.initializer_range,
            vocab_size=self.vocab_size,
            t2u_vocab_size=self.t2u_vocab_size,
            hidden_size=self.hidden_size,
            speech_encoder_layers=self.num_heads,
            speech_encoder_intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            decoder_ffn_dim=self.decoder_ffn_dim,
            t2u_encoder_layers=self.t2u_encoder_layers,
            t2u_decoder_layers=self.t2u_decoder_layers,
            t2u_encoder_ffn_dim=self.t2u_encoder_ffn_dim,
            t2u_decoder_ffn_dim=self.t2u_decoder_ffn_dim,
            num_attention_heads=self.num_heads,
            encoder_attention_heads=self.num_heads,
            decoder_attention_heads=self.num_heads,
            t2u_encoder_attention_heads=self.num_heads,
            t2u_decoder_attention_heads=self.num_heads,
            speech_encoder_attention_heads=self.num_heads,
            vocoder_num_spkrs=self.vocoder_num_spkrs,
            vocoder_num_langs=self.vocoder_num_langs,
            upsample_initial_channel=self.upsample_initial_channel,
            unit_embed_dim=self.unit_embed_dim,
            spkr_embed_dim=self.spkr_embed_dim,
            num_conv_pos_embeddings=self.num_conv_pos_embeddings,
            lang_embed_dim=self.lang_embed_dim,
            max_new_tokens=self.max_new_tokens,
            unit_hifi_gan_vocab_size=self.unit_hifi_gan_vocab_size,
            t2u_num_langs=self.t2u_num_langs,
            t2u_max_new_tokens=self.t2u_max_new_tokens,
            t2u_offset_tgt_lang=self.t2u_offset_tgt_lang,
            vocoder_offset=self.vocoder_offset,
        )


model_tester_text = SeamlessM4TModelTester(input_modality="text")
config_text, inputs_dict_text = model_tester_text.prepare_config_and_inputs_for_common()
model_tester_speech = SeamlessM4TModelTester(input_modality="speech")
config_speech, inputs_dict_speech = model_tester_speech.prepare_config_and_inputs_for_common()

SEAMLESS_M4T_CASES = [
    [
        "SeamlessM4TForSpeechToSpeech",
        "transformers.SeamlessM4TForSpeechToSpeech",
        "mindone.transformers.SeamlessM4TForSpeechToSpeech",
        (config_speech,),
        {},
        (
            inputs_dict_speech["input_features"],
            inputs_dict_speech["attention_mask"],
            inputs_dict_speech["decoder_input_ids"],
            inputs_dict_speech["labels"],
        ),
        {},
        [1, 2, 3],
        {
            "logits": 0,
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "SeamlessM4TForSpeechToText",
        "transformers.SeamlessM4TForSpeechToText",
        "mindone.transformers.SeamlessM4TForSpeechToText",
        (config_speech,),
        {},
        (
            inputs_dict_speech["input_features"],
            inputs_dict_speech["attention_mask"],
            inputs_dict_speech["decoder_input_ids"],
            inputs_dict_speech["labels"],
        ),
        {},
        [1, 2, 3],
        {
            "logits": 0,
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "SeamlessM4TForTextToSpeech",
        "transformers.SeamlessM4TForTextToSpeech",
        "mindone.transformers.SeamlessM4TForTextToSpeech",
        (config_text,),
        {},
        (
            inputs_dict_text["input_ids"],
            inputs_dict_text["attention_mask"],
            inputs_dict_text["decoder_input_ids"],
            inputs_dict_text["labels"],
        ),
        {},
        [0, 1, 2, 3],
        {
            "logits": 0,
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "SeamlessM4TForTextToText",
        "transformers.SeamlessM4TForTextToText",
        "mindone.transformers.SeamlessM4TForTextToText",
        (config_text,),
        {},
        (
            inputs_dict_text["input_ids"],
            inputs_dict_text["attention_mask"],
            inputs_dict_text["decoder_input_ids"],
            inputs_dict_text["labels"],
        ),
        {},
        [0, 1, 2, 3],
        {
            "logits": 0,
            "encoder_last_hidden_state": 2,
        },
    ],
    [
        "SeamlessM4TModel",
        "transformers.SeamlessM4TModel",
        "mindone.transformers.SeamlessM4TModel",
        (config_text,),
        {},
        (
            inputs_dict_text["input_ids"],
            inputs_dict_speech["input_features"],
            inputs_dict_text["attention_mask"],
            inputs_dict_text["decoder_input_ids"],
            inputs_dict_text["labels"],
        ),
        {},
        [0, 2, 3, 4],
        {
            "logits": 0,
            "encoder_last_hidden_state": 2,
        },
    ],
]


@pytest.mark.parametrize(
    "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,inputs_type_idx,outputs_map,dtype,mode",
    [
        case
        + [
            dtype,
        ]
        + [
            mode,
        ]
        for case in SEAMLESS_M4T_CASES
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
    inputs_type_idx,
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
    ms_inputs_kwargs["return_dict"] = False

    pt_inputs_args = tuple(
        tensor.long() if i in inputs_type_idx else tensor.to(PT_DTYPE_MAPPING[pt_dtype])
        for i, tensor in enumerate(pt_inputs_args)
    )

    ms_inputs_args = tuple(
        tensor.to(ms.int64) if i in inputs_type_idx else tensor.to(MS_DTYPE_MAPPING[ms_dtype])
        for i, tensor in enumerate(ms_inputs_args)
    )

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
