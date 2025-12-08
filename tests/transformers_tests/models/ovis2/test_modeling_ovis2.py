"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/ovis2/test_modeling_ovis2.py."""

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
from transformers import Ovis2Config

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


class Ovis2VisionText2TextModelTester:
    def __init__(
        self,
        seq_length=7,
        text_config={
            "model_type": "qwen2",
            "seq_length": 7,
            "is_training": True,
            "use_labels": True,
            "vocab_size": 99,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 54,
            "hidden_act": "silu",
            "max_position_embeddings": 580,
            "initializer_range": 0.02,
            "num_labels": 3,
            "pad_token_id": 0,
        },
        is_training=True,
        vision_config={
            "image_size": 32,
            "patch_size": 8,
            "num_channels": 3,
            "hidden_size": 64,
            "vocab_size": 99,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 54,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "qkv_bias": False,
            "hidden_stride": 2,
            "tokenize_function": "softmax",
        },
        image_token_id=1,
        visual_indicator_token_ids=[],
        vocab_size=99,
        hidden_size=64,
        ignore_id=-100,
    ):
        self.text_config = text_config
        self.vision_config = vision_config
        self.image_token_id = image_token_id
        self.visual_indicator_token_ids = visual_indicator_token_ids
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.image_seq_length = (
            vision_config["image_size"] // (vision_config["patch_size"] * vision_config["hidden_stride"])
        ) ** 2
        self.seq_length = seq_length + self.image_seq_length
        self.is_training = is_training
        self.num_attention_heads = text_config["num_attention_heads"]

        self.num_hidden_layers = text_config["num_hidden_layers"]
        self.pad_token_id = text_config["pad_token_id"]
        self.ignore_id = ignore_id

        self.batch_size = 3
        self.num_channels = 3

    def get_config(self):
        return Ovis2Config(
            text_config=self.text_config,
            vision_config=self.vision_config,
            image_token_id=self.image_token_id,
            visual_indicator_token_ids=self.visual_indicator_token_ids,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
        )

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy(
            [
                self.batch_size,
                self.vision_config["num_channels"],
                self.vision_config["image_size"],
                self.vision_config["image_size"],
            ]
        )
        config = self.get_config()

        return config, pixel_values

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, pixel_values = config_and_inputs

        vocab_range = self.vocab_size - 2
        input_ids = ids_numpy([self.batch_size, self.seq_length], vocab_range) + 2
        input_ids[:, : self.image_seq_length] = config.image_token_id

        attention_mask = np.ones(input_ids.shape, dtype=np.int64)

        labels = np.zeros((self.batch_size, self.seq_length), dtype=np.int64)
        labels[:, : self.image_seq_length] = self.ignore_id

        inputs_dict = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return config, inputs_dict


model_tester = Ovis2VisionText2TextModelTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


TEST_CASES = [
    [
        "Ovis2Model",
        "transformers.Ovis2Model",
        "mindone.transformers.Ovis2Model",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "last_hidden_state": "last_hidden_state",
        },
    ],
    [
        "Ovis2ForConditionalGeneration",
        "transformers.Ovis2ForConditionalGeneration",
        "mindone.transformers.Ovis2ForConditionalGeneration",
        (config,),
        {},
        (),
        inputs_dict,
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
