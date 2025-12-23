"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/bigbird_pegasus/test_modeling_bigbird_pegasus.py."""

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
import math

import numpy as np
import pytest
import torch
from transformers import Lfm2VlConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

from ...causal_lm_tester import CausalLMModelTester

# CrossEntropyLoss not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class Lfm2VlModelTester(CausalLMModelTester):
    config_class = Lfm2VlConfig

    def __init__(
        self,
        parent=None,
        is_training=True,
        batch_size=2,
        scale_factor=2,
        num_images=2,
        vision_config={
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_channels": 3,
            "num_patches": 16,
            "patch_size": 4,
            "hidden_act": "gelu_pytorch_tanh",
            "layer_norm_eps": 1e-6,
            "attention_dropout": 0.0,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 32,
            "intermediate_size": 37,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 100,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "tie_word_embeddings": True,
            "rope_theta": 1000000.0,
            "conv_bias": False,
            "conv_L_cache": 3,
            "block_multiple_of": 2,
            "full_attn_idxs": [0],
        },
        image_token_id=4,
        downsample_factor=4,
        projector_hidden_size=32,
    ):
        super().__init__(parent)
        self.vision_config = vision_config
        self.text_config = text_config
        self.image_token_id = image_token_id
        self.is_training = is_training
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.num_images = num_images
        self.downsample_factor = downsample_factor
        self.projector_hidden_size = projector_hidden_size
        self.image_seq_length = 4

    def get_config(self):
        return Lfm2VlConfig(
            vision_config=self.vision_config,
            text_config=self.text_config,
            image_token_id=self.image_token_id,
            downsample_factor=self.downsample_factor,
            projector_hidden_size=self.projector_hidden_size,
        )

    def prepare_config_and_inputs(self):
        # Create dummy pixel values: [num_images, num_patches, channels * patch_size^2]
        patch_size = self.vision_config["patch_size"]
        pixel_values = floats_numpy([self.num_images, 64, 3 * patch_size * patch_size])

        # Spatial shapes: one (height_patches, width_patches) per image
        patches = int(math.sqrt(64))
        spatial_shapes = np.array([[patches, patches]] * self.num_images, dtype=np.int64)

        # Pixel attention mask: mark all patches as valid (no padding)
        pixel_attention_mask = np.ones((self.num_images, 64), dtype=np.int64)
        config = self.get_config()
        return config, pixel_values, spatial_shapes, pixel_attention_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values, spatial_shapes, pixel_attention_mask = config_and_inputs
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1

        # For simplicity just set the last n tokens to the image token
        input_ids[input_ids == self.image_token_id] = self.text_config["pad_token_id"]
        input_ids[:, -self.image_seq_length :] = self.image_token_id

        attention_mask = np.not_equal(input_ids, 1)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spatial_shapes": spatial_shapes,
            "pixel_attention_mask": pixel_attention_mask,
        }
        return config, inputs_dict


model_tester = Lfm2VlModelTester()
(config, inputs_dict) = model_tester.prepare_config_and_inputs_for_common()


BERT_CASES = [
    [
        "Lfm2VlModel",
        "transformers.Lfm2VlModel",
        "mindone.transformers.Lfm2VlModel",
        (config,),
        {},
        (),
        {**inputs_dict},
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "Lfm2VlForConditionalGeneration",
        "transformers.Lfm2VlForConditionalGeneration",
        "mindone.transformers.Lfm2VlForConditionalGeneration",
        (config,),
        {},
        (),
        {**inputs_dict},
        {
            "logits": 1,
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
        for case in BERT_CASES
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
            # print(": ==map", pt_key, ms_idx)
            pt_output = getattr(pt_outputs, pt_key)
            ms_output = ms_outputs[pt_key]
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
