"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/qwen2_vl/test_modeling_qwen2_vl.py."""

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
from transformers import Qwen2VLConfig

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
MODES = [1]  # not support graph mode yet


class Qwen2VLModelTester:
    def __init__(
        self,
        batch_size=1,
        seq_length=7,
        num_channels=3,
        # For common tests
        is_training=False,
        use_attention_mask=True,
        use_labels=False,
        use_cache=False,
        output_attentions=False,
        # For net config
        vocab_size=99,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=32,
        max_position_embeddings=512,
        use_sliding_window=False,
        attn_implementation="eager",
        image_size=14,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=2,
        vision_start_token_id=3,
        image_token_id=4,
        video_token_id=5,
        vision_config={
            "depth": 2,
            "in_chans": 3,
            "hidden_act": "silu",
            "intermediate_size": 32,
            "out_hidden_size": 128,
            "hidden_size": 128,
            "num_heads": 8,
            "patch_size": 14,
            "spatial_patch_size": 14,
            "spatial_merge_size": 1,
            "temporal_patch_size": 2,
        },
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_channels = num_channels
        # For common tests
        self.seq_length = self.seq_length
        self.is_training = is_training
        self.use_attention_mask = use_attention_mask
        self.use_labels = use_labels
        self.use_cache = use_cache
        self.output_attentions = output_attentions
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.use_sliding_window = use_sliding_window
        self.attn_implementation = attn_implementation
        self.rope_scaling = {
            "mrope_section": [2, 3, 3],
            "type": "mrope",
        }  # sum*2=16 = head_dim = hidden_size//num_attention_heads = 128//8=16
        self.image_size = image_size
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.vision_start_token_id = vision_start_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = vision_config
        self.num_image_tokens = 32
        self.seq_length = self.seq_length + self.num_image_tokens

    def get_large_model_config(self):
        return Qwen2VLConfig.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = None
        if self.use_attention_mask:
            attention_mask = ids_numpy([self.batch_size, self.seq_length], vocab_size=2)

        config = self.get_config()
        # config = self.get_large_model_config()
        # config._attn_implementation = self.attn_implementation # default use sdpa

        patch_size = config.vision_config.patch_size
        temporal_patch_size = config.vision_config.temporal_patch_size
        pixel_values = floats_numpy(
            [
                self.batch_size * (self.image_size**2) // (patch_size**2),
                self.num_channels * (patch_size**2) * temporal_patch_size,
            ]
        )
        image_grid_thw = np.array([[1, 1, 1]] * self.batch_size)

        input_ids[:, -1] = self.pad_token_id
        input_ids[input_ids == self.video_token_id] = self.pad_token_id
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
        input_ids[:, self.num_image_tokens] = self.image_token_id
        input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id

        return config, input_ids, attention_mask, pixel_values, image_grid_thw

    def get_config(self):
        config = Qwen2VLConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=8,
            max_position_embeddings=self.max_position_embeddings,
            rope_scaling=self.rope_scaling,
            use_cache=self.use_cache,
            output_attentions=self.output_attentions,
            attn_implementation=self.attn_implementation,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_start_token_id=self.vision_start_token_id,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            vision_config=self.vision_config,
        )

        return config


model_tester = Qwen2VLModelTester()
(config, input_ids, attention_mask, pixel_values, image_grid_thw) = model_tester.prepare_config_and_inputs()


T5_CASES = [
    [  # LM
        "Qwen2VLModel",
        "transformers.Qwen2VLModel",
        "mindone.transformers.Qwen2VLModel",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        {
            "last_hidden_state": 0,
        },
    ],
    [  # VQA
        "Qwen2VLForConditionalGeneration",
        "transformers.Qwen2VLForConditionalGeneration",
        "mindone.transformers.Qwen2VLForConditionalGeneration",
        (config,),
        {},
        (),
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
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
        for case in T5_CASES
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
