"""
Adapted from https://github.com/huggingface/transformers/blob/a6393e7d28e652c598ced79f0107f1eff370df1b/tests/models/smolvlm/test_modeling_smolvlm.py
"""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import numpy as np
import pytest
import torch
from transformers import SmolVLMConfig

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

# FIXME in fp16/bf16 case, torch test cases do not run in expected precision.
# and the comparable cases has error > 0.1 through the `bucketize` operater in pos emb computation.
DTYPE_AND_THRESHOLDS = {"fp32": 5e-4}  # , "fp16": 1e-3, "bf16": 1e-2}


class SmolVLMVisionText2TextModelTester:
    def __init__(
        self,
        is_training=True,
        batch_size=2,
        scale_factor=2,
        num_images=2,
        vision_config={
            "image_size": 16,
            "patch_size": 4,
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "dropout": 0.1,
            "attention_dropout": 0.1,
            "initializer_range": 0.02,
        },
        text_config={
            "vocab_size": 100,
            "hidden_size": 64,
            "intermediate_size": 56,
            "num_hidden_layers": 3,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": 256,
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-6,
            "pad_token_id": 2,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "image_token_id": 57,
            "tie_word_embeddings": False,
            "rope_theta": 10000.0,
            "sliding_window": 32,
            "attention_dropout": 0.0,
        },
        use_cache=False,
        tie_word_embeddings=False,
        image_token_id=57,
    ):
        self.is_training = is_training
        self.batch_size = batch_size
        self.num_images = num_images
        self.scale_factor = scale_factor
        self.seq_length = (
            int(((vision_config["image_size"] // vision_config["patch_size"]) ** 2) / (self.scale_factor**2))
            * self.num_images
        )
        self.use_cache = use_cache
        self.image_token_id = image_token_id
        self.tie_word_embeddings = tie_word_embeddings
        # Hack - add properties here so use common tests
        self.vocab_size = text_config["vocab_size"]
        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.num_attention_heads = text_config["num_attention_heads"]
        self.hidden_size = text_config["hidden_size"]

        self.vision_config = vision_config
        self.text_config = text_config

    def get_config(self):
        return SmolVLMConfig(
            use_cache=self.use_cache,
            image_token_id=self.image_token_id,
            tie_word_embeddings=self.tie_word_embeddings,
            vision_config=self.vision_config,
            text_config=self.text_config,
            vocab_size=self.vocab_size,
            scale_factor=self.scale_factor,
        )

    def prepare_config_and_inputs(self) -> tuple[SmolVLMConfig, np.ndarray]:
        pixel_values = floats_numpy(
            [
                self.batch_size,
                self.num_images,
                3,  # SmolVLMImageProcessor always generates RGB pixel values
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self) -> tuple[SmolVLMConfig, dict[str, np.ndarray]]:
        config, pixel_values = self.prepare_config_and_inputs()
        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size - 2) + 1

        # For simplicity just set the last n tokens to the image token
        n_image_tokens_per_batch = self.seq_length
        input_ids[:, -n_image_tokens_per_batch:] = self.image_token_id
        attention_mask = np.not_equal(input_ids, 1)
        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


model_tester = SmolVLMVisionText2TextModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()

_CASES = [
    [
        "SmolVLMForConditionalGeneration",
        "transformers.SmolVLMForConditionalGeneration",
        "mindone.transformers.SmolVLMForConditionalGeneration",
        (config,),
        {},
        (),
        inputs_dict,
        {"logits": "logits"},
    ],
]


@pytest.mark.parametrize("name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map", _CASES)
@pytest.mark.parametrize("dtype", DTYPE_AND_THRESHOLDS.keys())
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
