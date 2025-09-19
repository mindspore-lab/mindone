"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/seamless_m4t_v2/test_modeling_seamless_m4t_v2.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import inspect

import numpy as np
import pytest
import torch
from transformers import SeamlessM4Tv2Config

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


class SeamlessM4Tv2ModelTester:
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
        encoder_layerdrop=0.0,
        speech_encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
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
        t2u_offset_tgt_lang=0,
        vocoder_offset=0,
        t2u_variance_predictor_hidden_dim=4,
        char_vocab_size=4,
        left_max_position_embeddings=2,
        right_max_position_embeddings=1,
        speech_encoder_chunk_size=2,
        speech_encoder_left_chunk_num=1,
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
        self.t2u_offset_tgt_lang = t2u_offset_tgt_lang
        self.vocoder_offset = vocoder_offset

        self.t2u_variance_predictor_hidden_dim = t2u_variance_predictor_hidden_dim
        self.char_vocab_size = char_vocab_size
        self.left_max_position_embeddings = left_max_position_embeddings
        self.right_max_position_embeddings = right_max_position_embeddings
        self.speech_encoder_chunk_size = speech_encoder_chunk_size
        self.speech_encoder_left_chunk_num = speech_encoder_left_chunk_num

        self.encoder_layerdrop = encoder_layerdrop
        self.speech_encoder_layerdrop = speech_encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop

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

    def get_config(self):
        return SeamlessM4Tv2Config(
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
            unit_hifigan_vocab_vise=self.t2u_vocab_size,
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
            t2u_offset_tgt_lang=self.t2u_offset_tgt_lang,
            vocoder_offset=self.vocoder_offset,
            t2u_variance_predictor_embed_dim=self.hidden_size,
            t2u_variance_predictor_hidden_dim=self.t2u_variance_predictor_hidden_dim,
            char_vocab_size=self.char_vocab_size,
            left_max_position_embeddings=self.left_max_position_embeddings,
            right_max_position_embeddings=self.right_max_position_embeddings,
            speech_encoder_chunk_size=self.speech_encoder_chunk_size,
            speech_encoder_left_chunk_num=self.speech_encoder_left_chunk_num,
            encoder_layerdrop=self.encoder_layerdrop,
            speech_encoder_layerdrop=self.speech_encoder_layerdrop,
            decoder_layerdrop=self.decoder_layerdrop,
        )


model_tester = SeamlessM4Tv2ModelTester()
(
    config,
    input_ids,
    decoder_input_ids,
    input_mask,
    lm_labels,
) = model_tester.prepare_config_and_inputs()


SEAMLESS_M4T_V2_CASES = [
    [
        "SeamlessM4Tv2Model",
        "transformers.SeamlessM4Tv2Model",
        "mindone.transformers.SeamlessM4Tv2Model",
        (config,),
        {},
        (),
        {
            "input_features": input_ids,
            "attention_mask": input_mask,
            "decoder_input_ids": decoder_input_ids,
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
        for case in SEAMLESS_M4T_V2_CASES
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
