"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/kyutai_speech_to_text/test_modeling_kyutai_speech_to_text.py."""

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
from transformers import KyutaiSpeechToTextConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [1]


class KyutaiSpeechToTextModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        text_seq_length=1,
        input_values_length=192,  # gives 3 audio tokens, corresponding to the default in GenerationTesterMixin
        is_training=False,
        use_input_mask=True,
        use_token_type_ids=False,
        use_labels=True,
        codebook_vocab_size=2049,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=None,
        max_position_embeddings=512,
        rope_theta=10000.0,
        hidden_act="silu",
        head_dim=None,
        initializer_range=0.02,
        use_cache=True,
        sliding_window=512,
        attention_dropout=0.1,
        ffn_dim=38,
        rms_norm_eps=1e-6,
        num_codebooks=8,
        frame_size=64,
        delay_in_tokens=5,
        audio_bos_token_id=2048,
        audio_pad_token_id=2048,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        codec_config={
            "model_type": "mimi",
            "num_quantizers": 8,
            "audio_channels": 1,
            "chunk_in_sec": None,
            "hidden_size": 16,
            "num_filters": 8,
            "num_residual_layers": 1,
            "upsampling_ratios": [8, 4],
            "codebook_size": 16,
            "vector_quantization_hidden_dimension": 16,
            "upsample_groups": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "sliding_window": 4,
            "codebook_dim": 16,
            "use_cache": False,
        },
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.text_seq_length = text_seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_token_type_ids = use_token_type_ids
        self.use_labels = use_labels
        self.codebook_vocab_size = codebook_vocab_size
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.hidden_act = hidden_act
        self.head_dim = head_dim
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        self.ffn_dim = ffn_dim
        self.rms_norm_eps = rms_norm_eps
        self.num_codebooks = num_codebooks
        self.frame_size = frame_size
        self.delay_in_tokens = delay_in_tokens
        self.audio_bos_token_id = audio_bos_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.codec_config = codec_config
        self.scope = scope
        self.input_values_length = input_values_length

    def get_config(self):
        return KyutaiSpeechToTextConfig(
            codebook_vocab_size=self.codebook_vocab_size,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            hidden_act=self.hidden_act,
            head_dim=self.head_dim,
            initializer_range=self.initializer_range,
            use_cache=self.use_cache,
            sliding_window=self.sliding_window,
            attention_dropout=self.attention_dropout,
            ffn_dim=self.ffn_dim,
            rms_norm_eps=self.rms_norm_eps,
            num_codebooks=self.num_codebooks,
            frame_size=self.frame_size,
            delay_in_tokens=self.delay_in_tokens,
            audio_bos_token_id=self.audio_bos_token_id,
            audio_pad_token_id=self.audio_pad_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            codec_config=self.codec_config,
        )

    def prepare_config_and_inputs(self):
        config = self.get_config()

        text_input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size - 1) + 1
        codebook_input_ids = (
            ids_numpy([self.batch_size, self.seq_length, self.num_codebooks], self.codebook_vocab_size - 1) + 1
        )

        input_ids = np.concatenate([np.expand_dims(text_input_ids, axis=2), codebook_input_ids], axis=2)
        attention_mask = np.not_equal(text_input_ids, 1)

        return config, input_ids, attention_mask


model_tester = KyutaiSpeechToTextModelTester()
config, input_ids, attention_mask = model_tester.prepare_config_and_inputs()


Moshi_CASES = [
    [
        "KyutaiSpeechToTextModel",
        "transformers.KyutaiSpeechToTextModel",
        "mindone.transformers.KyutaiSpeechToTextModel",
        (config,),
        {},
        (input_ids, attention_mask),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "KyutaiSpeechToTextForConditionalGeneration",
        "transformers.KyutaiSpeechToTextForConditionalGeneration",
        "mindone.transformers.KyutaiSpeechToTextForConditionalGeneration",
        (config,),
        {},
        (input_ids, attention_mask),
        {},
        {
            "logits": "logits",
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
        for case in Moshi_CASES
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
