"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/phi4_multimodal/test_modeling_phi4_multimodal.py."""

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
import logging

import numpy as np
import pytest
import torch
from transformers import Phi4MultimodalAudioConfig, Phi4MultimodalConfig, Phi4MultimodalVisionConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-3}
MODES = [1]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Phi4MultimodalModelTester:
    def __init__(
        self,
        batch_size=2,
        seq_length=12,
        image_seq_length=275,
        audio_seq_length=8,
        is_training=True,
        num_hidden_layers=2,
        vocab_size=49,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=8,
        num_key_value_heads=4,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=0,
        image_token_id=1,
        audio_token_id=2,
        image_size=16,
        audio_size=12,
        audio_config=Phi4MultimodalAudioConfig(
            num_blocks=2,
            hidden_size=32,
            num_attention_heads=8,
            intermediate_size=48,
            depthwise_separable_out_channel=128,
            nemo_conv_channels=128,
            initializer_range=1e-5,
        ),
        vision_config=Phi4MultimodalVisionConfig(
            num_hidden_layers=2,
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=8,
            crop_size=16,
            initializer_range=1e-5,
        ),
    ):
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.bos_token_id = bos_token_id
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.audio_config = audio_config
        self.vision_config = vision_config

        self.is_training = is_training
        self.batch_size = batch_size
        self.seq_length = seq_length + image_seq_length + audio_seq_length
        self.image_seq_length = image_seq_length
        self.audio_seq_length = audio_seq_length
        self.image_size = image_size
        self.audio_size = audio_size
        self.num_channels = 3

    def get_config(self):
        return Phi4MultimodalConfig(
            num_hidden_layers=self.num_hidden_layers,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            vision_config=self.vision_config,
            audio_config=self.audio_config,
        )

    # Copied from tests.models.mistral.test_modeling_mistral.MistralModelTester.prepare_config_and_inputs
    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        # The shapes corresponds to the inputs for image of size 16x16
        image_pixel_values = floats_numpy([self.batch_size, 2, self.num_channels, self.image_size, self.image_size])
        image_attention_mask = np.ones((self.batch_size, 2, 1, 1))
        image_sizes = ms.tensor([[self.image_size, self.image_size]] * self.batch_size, dtype=ms.int64)

        # Feature sizes returned by an audio of size 10000
        audio_input_features = floats_numpy([self.batch_size, 61, 80])
        audio_embed_sizes = ms.tensor([self.audio_seq_length] * self.batch_size, dtype=ms.int64)

        input_ids[input_ids == self.pad_token_id] = self.pad_token_id + 1  # random value but not pad token
        input_ids[-1, 0] = self.pad_token_id  # mask the last text token
        input_ids[:, -self.image_seq_length - self.audio_seq_length : -self.audio_seq_length] = self.image_token_id
        input_ids[:, -self.audio_seq_length :] = self.audio_token_id

        attention_mask = np.ones_like(input_ids)
        attention_mask[-1, 0] = 0  # mask the last text token
        config = self.get_config()

        return (
            config,
            input_ids,
            attention_mask,
            image_pixel_values,
            image_attention_mask,
            image_sizes,
            audio_input_features,
            audio_embed_sizes,
        )


model_tester = Phi4MultimodalModelTester()
(
    config,
    input_ids,
    attention_mask,
    image_pixel_values,
    image_attention_mask,
    image_sizes,
    audio_input_features,
    audio_embed_sizes,
) = model_tester.prepare_config_and_inputs()


PHI4_CASES = [
    [
        "Phi4MultimodalModel",
        "transformers.Phi4MultimodalModel",
        "mindone.transformers.Phi4MultimodalModel",
        (config,),
        {},
        (input_ids,),
        {
            "attention_mask": attention_mask,
        },
        {
            "last_hidden_state": 0,
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
        for case in PHI4_CASES
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
    # logger.info(f"ms:{ms_outputs}")
    # logger.info(f"pt:{pt_outputs}" )
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
    logger.info(f"Differences: {diffs}")
    THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
    assert (np.array(diffs) < THRESHOLD).all(), (
        f"ms_dtype: {ms_dtype}, pt_type: {pt_dtype}, "
        f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
    )
