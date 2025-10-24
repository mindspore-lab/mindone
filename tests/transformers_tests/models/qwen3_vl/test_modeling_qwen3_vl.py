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
import transformers

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


if transformers.__version__ >= "4.57.0":
    from transformers import Qwen3VLConfig

    class Qwen3VLModelTester:
        def __init__(
            self,
            batch_size=3,
            seq_length=7,
            num_channels=3,
            ignore_index=-100,
            image_size=16,
            text_config={
                "bos_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 2,
                "hidden_act": "silu",
                "head_dim": 8,
                "hidden_size": 32,
                "vocab_size": 99,
                "intermediate_size": 37,
                "max_position_embeddings": 512,
                "model_type": "qwen3_vl",
                "num_attention_heads": 4,
                "num_hidden_layers": 4,
                "num_key_value_heads": 2,
                "rope_theta": 10000,
                "tie_word_embeddings": True,
                "rope_scaling": {"rope_type": "default", "mrope_section": [16, 8, 8], "mrope_interleaved": True},
            },
            vision_config={
                "depth": 2,
                "in_chans": 3,
                "hidden_act": "gelu_pytorch_tanh",
                "intermediate_size": 32,
                "out_hidden_size": 32,
                "hidden_size": 32,
                "num_heads": 4,
                "patch_size": 16,
                "spatial_merge_size": 1,
                "temporal_patch_size": 2,
                "num_position_embeddings": 16,
                "deepstack_visual_indexes": [0, 1],
            },
            image_token_id=3,
            video_token_id=4,
            vision_start_token_id=5,
            vision_end_token_id=6,
            tie_word_embeddings=True,
            is_training=False,
        ):
            self.ignore_index = ignore_index
            self.is_training = is_training

            self.vision_config = vision_config
            self.text_config = text_config

            self.vocab_size = text_config["vocab_size"]
            self.bos_token_id = text_config["bos_token_id"]
            self.eos_token_id = text_config["eos_token_id"]
            self.pad_token_id = text_config["pad_token_id"]
            self.head_dim = text_config["head_dim"]
            self.hidden_size = text_config["hidden_size"]
            self.intermediate_size = text_config["intermediate_size"]
            self.num_hidden_layers = text_config["num_hidden_layers"]
            self.num_attention_heads = text_config["num_attention_heads"]
            self.num_key_value_heads = text_config["num_key_value_heads"]
            self.rope_theta = text_config["rope_theta"]
            self.rope_scaling = text_config["rope_scaling"]
            self.hidden_act = text_config["hidden_act"]
            self.max_position_embeddings = text_config["max_position_embeddings"]
            self.model_type = text_config["model_type"]

            self.vision_start_token_id = vision_start_token_id
            self.vision_end_token_id = vision_end_token_id
            self.image_token_id = image_token_id
            self.video_token_id = video_token_id
            self.tie_word_embeddings = tie_word_embeddings

            self.batch_size = batch_size
            self.num_channels = num_channels
            self.image_size = image_size
            self.num_image_tokens = 32
            self.seq_length = seq_length + self.num_image_tokens

        def get_config(self):
            return Qwen3VLConfig(
                text_config=self.text_config,
                vision_config=self.vision_config,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                vision_start_token_id=self.vision_start_token_id,
                vision_end_token_id=self.vision_end_token_id,
                tie_word_embeddings=self.tie_word_embeddings,
            )

        def prepare_config_and_inputs(self):
            config = self.get_config()
            patch_size = config.vision_config.patch_size
            temporal_patch_size = config.vision_config.temporal_patch_size
            pixel_values = floats_numpy(
                [
                    self.batch_size * (self.image_size**2) // (patch_size**2),
                    self.num_channels * (patch_size**2) * temporal_patch_size,
                ]
            )

            return config, pixel_values

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            config, pixel_values = config_and_inputs
            input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
            attention_mask = np.ones(input_ids.shape, dtype=np.int64)

            input_ids[:, -1] = self.pad_token_id
            input_ids[input_ids == self.video_token_id] = self.pad_token_id
            input_ids[input_ids == self.image_token_id] = self.pad_token_id
            input_ids[input_ids == self.vision_start_token_id] = self.pad_token_id
            input_ids[:, self.num_image_tokens] = self.image_token_id
            input_ids[:, self.num_image_tokens - 1] = self.vision_start_token_id
            image_grid_thw = np.array([[1, 1, 1]] * self.batch_size)
            return config, input_ids, attention_mask, pixel_values, image_grid_thw

    model_tester = Qwen3VLModelTester()
    (
        config,
        input_ids,
        attention_mask,
        pixel_values,
        image_grid_thw,
    ) = model_tester.prepare_config_and_inputs_for_common()

    QWEN3VL_CASES = [
        [  # LM
            "Qwen3VLModel",
            "transformers.Qwen3VLModel",
            "mindone.transformers.Qwen3VLModel",
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
            "Qwen3VLForConditionalGeneration",
            "transformers.Qwen3VLForConditionalGeneration",
            "mindone.transformers.Qwen3VLForConditionalGeneration",
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
            for case in QWEN3VL_CASES
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

        ms_inputs_kwargs.update({"use_cache": False})

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
