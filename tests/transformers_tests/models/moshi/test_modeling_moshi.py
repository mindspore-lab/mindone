"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/moshi/test_modeling_moshi.py."""

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
from transformers import MoshiConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 7e-3}
MODES = [1]


class MoshiTester:
    def __init__(
        self,
        batch_size=4,  # need batch_size != num_hidden_layers
        seq_length=7,
        is_training=True,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=4,
        hidden_act="silu",
        rms_norm_eps=0.001,
        ffn_dim=32,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=100,
        pad_token_id=25,
        bos_token_id=25,
        num_codebooks=4,
        audio_encoder_type="mimi",
        attn_implementation="eager",
        depth_hidden_size=16,
        depth_num_hidden_layers=2,
        depth_max_position_embeddings=5,
        depth_num_attention_heads=8,
        depth_ffn_dim=16,
        depth_sliding_window=4,
        mimi_intermediate_size=40,
        mimi_hidden_size=32,
        mimi_num_filters=8,
        mimi_num_residual_layers=1,
        mimi_upsampling_ratios=[8, 4],
        mimi_codebook_size=64,
        mimi_vector_quantization_hidden_dimension=64,
        mimi_codebook_dim=64,
        mimi_upsample_groups=32,
        mimi_num_hidden_layers=2,
        mimi_num_attention_heads=2,
        mimi_num_key_value_heads=2,
        mimi_sliding_window=3,
        sampling_rate=800,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.ffn_dim = ffn_dim
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.num_codebooks = num_codebooks
        self.attn_implementation = attn_implementation
        self.depth_hidden_size = depth_hidden_size
        self.depth_num_hidden_layers = depth_num_hidden_layers
        self.depth_max_position_embeddings = depth_max_position_embeddings
        self.depth_num_attention_heads = depth_num_attention_heads
        self.depth_ffn_dim = depth_ffn_dim
        self.depth_sliding_window = depth_sliding_window

        self.audio_encoder_type = audio_encoder_type
        self.mimi_intermediate_size = mimi_intermediate_size
        self.mimi_hidden_size = mimi_hidden_size
        self.mimi_num_filters = mimi_num_filters
        self.mimi_num_residual_layers = mimi_num_residual_layers
        self.mimi_upsampling_ratios = mimi_upsampling_ratios
        self.mimi_codebook_size = mimi_codebook_size
        self.mimi_vector_quantization_hidden_dimension = mimi_vector_quantization_hidden_dimension
        self.mimi_codebook_dim = mimi_codebook_dim
        self.mimi_upsample_groups = mimi_upsample_groups
        self.mimi_num_hidden_layers = mimi_num_hidden_layers
        self.mimi_num_attention_heads = mimi_num_attention_heads
        self.mimi_num_key_value_heads = mimi_num_key_value_heads
        self.mimi_sliding_window = mimi_sliding_window
        self.sampling_rate = sampling_rate

        self.num_hidden_states_types = 2

    def prepare_config_and_inputs(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size

        input_ids = ids_numpy([batch_size, self.seq_length], self.vocab_size)

        moshi_audio_codes = ids_numpy([batch_size, self.num_codebooks, self.seq_length], self.mimi_codebook_size)
        user_audio_codes = ids_numpy([batch_size, self.num_codebooks, self.seq_length], self.mimi_codebook_size)
        # attention_mask = input_ids.ne(self.pad_token_id)
        attention_mask = np.not_equal(input_ids, self.pad_token_id)

        config = self.get_config()
        # set config._attn_implementation
        config._attn_implementation = "eager"
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "moshi_audio_codes": moshi_audio_codes,
            "user_audio_codes": user_audio_codes,
        }
        return config, inputs_dict

    def get_config(self):
        mimi_dict_config = {
            "model_type": self.audio_encoder_type,
            "audio_channels": 1,
            "hidden_size": self.mimi_hidden_size,
            "num_filters": self.mimi_num_filters,
            "num_residual_layers": self.mimi_num_residual_layers,
            "upsampling_ratios": self.mimi_upsampling_ratios,
            "codebook_size": self.mimi_codebook_size,
            "vector_quantization_hidden_dimension": self.mimi_vector_quantization_hidden_dimension,
            "upsample_groups": self.mimi_upsample_groups,
            "num_hidden_layers": self.mimi_num_hidden_layers,
            "num_attention_heads": self.mimi_num_attention_heads,
            "num_key_value_heads": self.mimi_num_key_value_heads,
            "sliding_window": self.mimi_sliding_window,
            "codebook_dim": self.mimi_codebook_dim,
            "use_cache": False,
            "sampling_rate": self.sampling_rate,
        }

        depth_dict_config = {
            "hidden_size": self.depth_hidden_size,
            "num_hidden_layers": self.depth_num_hidden_layers,
            "max_position_embeddings": self.depth_max_position_embeddings,
            "num_attention_heads": self.depth_num_attention_heads,
            "ffn_dim": self.depth_ffn_dim,
            "sliding_window": self.depth_sliding_window,
        }

        config = MoshiConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            d_ff=self.intermediate_size,
            num_codebooks=self.num_codebooks,
            rms_norm_eps=self.rms_norm_eps,
            tie_word_embeddings=False,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            ffn_dim=self.ffn_dim,
            audio_encoder_config=mimi_dict_config,
            depth_decoder_config=depth_dict_config,
            attn_implementation=self.attn_implementation,
        )
        return config

    def prepare_config_and_inputs_for_common(self, batch_size=None):
        config, inputs_dict = self.prepare_config_and_inputs(batch_size)
        return config, inputs_dict


model_tester = MoshiTester()
config, inputs_dict = model_tester.prepare_config_and_inputs()


Moshi_CASES = [
    [
        "MoshiModel",
        "transformers.MoshiModel",
        "mindone.transformers.MoshiModel",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "MoshiForCausalLM",
        "transformers.MoshiForCausalLM",
        "mindone.transformers.MoshiForCausalLM",
        (config,),
        {},
        (inputs_dict["input_ids"], inputs_dict["attention_mask"]),
        {},
        {
            "logits": "logits",
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "MoshiForConditionalGeneration",
        "transformers.MoshiForConditionalGeneration",
        "mindone.transformers.MoshiForConditionalGeneration",
        (config,),
        {},
        (
            inputs_dict["input_ids"],
            inputs_dict["attention_mask"],
            None,
            inputs_dict["user_audio_codes"],
            None,
            inputs_dict["moshi_audio_codes"],
        ),
        {},
        {
            "logits": "logits",
            "last_hidden_state": "last_hidden_state",
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
