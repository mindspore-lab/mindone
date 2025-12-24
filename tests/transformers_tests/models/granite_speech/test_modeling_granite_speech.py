"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/granite_speech/test_modeling_granite_speech.py."""

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
from transformers import GraniteSpeechConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 6e-3}
MODES = [1]


class GraniteSpeechForConditionalGenerationModelTester:
    def __init__(
        self,
        seq_length=7,
        encoder_config={
            "model_type": "granite_speech_encoder",
            "context_size": 200,
            "conv_expansion_factor": 2,
            "conv_kernel_size": 15,
            "dim_head": 32,
            "dropout": 0.1,
            "feedforward_mult": 4,
            "hidden_dim": 32,
            "input_dim": 160,
            "num_heads": 4,
            "num_layers": 2,
            "output_dim": 42,
        },
        text_config={
            "model_type": "granite",
            "is_training": True,
            "seq_length": 7,
            "use_token_type_ids": False,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 37,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "max_position_embeddings": 580,
            "type_vocab_size": 16,
            "type_sequence_label_size": 2,
            "initializer_range": 0.02,
            "num_labels": 3,
            "num_choices": 4,
            "pad_token_id": 1,
        },
        projector_config={
            "attention_probs_dropout_prob": 0.1,
            "cross_attention_frequency": 1,
            "encoder_hidden_size": 32,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 32,
            "initializer_range": 0.02,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 2048,
            "model_type": "blip_2_qformer",
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "position_embedding_type": "absolute",
            "use_qformer_text_input": False,
            "vocab_size": 30522,
        },
        audio_token_index=0,
        tie_word_embeddings=True,
        initializer_range=0.02,
        has_lora_adapter=True,
        downsample_rate=5,
        window_size=15,
        is_training=True,
    ):
        self.encoder_config = encoder_config
        self.text_config = text_config
        self.projector_config = projector_config
        self.audio_token_index = audio_token_index
        self.tie_word_embeddings = tie_word_embeddings
        self.initializer_range = initializer_range
        self.has_lora_adapater = has_lora_adapter
        self.downsample_rate = downsample_rate
        self.window_size = window_size
        self.is_training = is_training

        # Dims for audio features
        self.sequence_dim = 844
        self.feature_dim = 160
        self.num_attention_heads = text_config["num_attention_heads"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.hidden_size = text_config["hidden_size"]
        self.batch_size = 3
        self.pad_token_id = text_config["pad_token_id"]
        self.seq_len = 7
        self.num_audio_tokens = 2
        self.seq_length = seq_length + self.num_audio_tokens

    def get_config(self):
        return GraniteSpeechConfig(
            encoder_config=self.encoder_config,
            text_config=self.text_config,
            projector_config=self.projector_config,
            audio_token_index=self.audio_token_index,
            tie_word_embeddings=self.tie_word_embeddings,
            initializer_range=self.initializer_range,
            has_lora_adapter=self.has_lora_adapater,
        )

    def prepare_config_and_inputs(self):
        input_features = floats_numpy(
            [self.batch_size, self.sequence_dim, self.feature_dim],
        )
        config = self.get_config()
        return config, input_features

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features = config_and_inputs
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 2
        attention_mask = np.ones(input_ids.shape, dtype=np.int64)
        input_ids[input_ids == config.audio_token_index] = self.pad_token_id

        input_ids[:, : self.num_audio_tokens] = config.audio_token_index

        inputs_dict = {
            "input_features": input_features,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


model_tester = GraniteSpeechForConditionalGenerationModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


TEST_CASES = [
    [
        "GraniteSpeechForConditionalGeneration",
        "transformers.GraniteSpeechForConditionalGeneration",
        "mindone.transformers.GraniteSpeechForConditionalGeneration",
        (config,),
        {},
        (),
        {
            "input_features": inputs_dict["input_features"],
            "input_ids": inputs_dict["input_ids"],
            "attention_mask": inputs_dict["attention_mask"],
        },
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
        for case in TEST_CASES
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
