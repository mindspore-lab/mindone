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
from transformers import Qwen2AudioConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class Qwen2AudioModelTester:
    def __init__(
        self,
        ignore_index=-100,
        audio_token_index=0,
        seq_length=25,
        feat_seq_length=60,
        text_config={
            "model_type": "qwen2",
            "intermediate_size": 36,
            "initializer_range": 0.02,
            "hidden_size": 32,
            "max_position_embeddings": 52,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "use_labels": True,
            "use_mrope": False,
            "vocab_size": 99,
        },
        is_training=False,
        audio_config={
            "model_type": "qwen2_audio_encoder",
            "d_model": 16,
            "encoder_attention_heads": 4,
            "encoder_ffn_dim": 16,
            "encoder_layers": 2,
            "num_mel_bins": 80,
            "max_source_positions": 30,
            "initializer_range": 0.02,
        },
    ):
        self.ignore_index = ignore_index
        self.audio_token_index = audio_token_index
        self.text_config = text_config
        self.audio_config = audio_config
        self.seq_length = seq_length
        self.feat_seq_length = feat_seq_length

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.vocab_size = text_config["vocab_size"]
        self.hidden_size = text_config["hidden_size"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.is_training = is_training

        self.batch_size = 3
        self.encoder_seq_length = seq_length

    def get_config(self):
        return Qwen2AudioConfig(
            attn_implementation="eager",
            text_config=self.text_config,
            audio_config=self.audio_config,
            ignore_index=self.ignore_index,
            audio_token_index=self.audio_token_index,
        )

    def prepare_config_and_inputs(self):
        input_features_values = floats_numpy(
            (
                self.batch_size,
                self.audio_config["num_mel_bins"],
                self.feat_seq_length,
            )
        )
        config = self.get_config()
        feature_attention_mask = np.ones((self.batch_size, self.feat_seq_length), dtype=np.int32)
        return config, input_features_values, feature_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_features_values, feature_attention_mask = config_and_inputs
        input_length = (input_features_values.shape[-1] - 1) // 2 + 1
        num_audio_tokens = (input_length - 2) // 2 + 1
        input_ids = ids_numpy((self.batch_size, self.seq_length), config.text_config.vocab_size - 1) + 1
        attention_mask = np.ones(input_ids.shape, dtype=np.int32)
        attention_mask[:, :1] = 0
        # we are giving 3 audios let's make sure we pass in 3 audios tokens
        input_ids[:, 1 : 1 + num_audio_tokens] = config.audio_token_index

        return config, input_features_values, feature_attention_mask, input_ids, attention_mask


model_tester = Qwen2AudioModelTester()

(
    config,
    input_features,
    feature_attention_mask,
    input_ids,
    attention_mask,
) = model_tester.prepare_config_and_inputs_for_common()

Qwen2Audio_CASES = [
    [
        "qwen2_audio",
        "transformers.Qwen2AudioForConditionalGeneration",
        "mindone.transformers.Qwen2AudioForConditionalGeneration",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "input_features": input_features,
            "feature_attention_mask": feature_attention_mask,
            "attention_mask": attention_mask,
        },
        {
            "logits": 0,
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
        for case in Qwen2Audio_CASES
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
