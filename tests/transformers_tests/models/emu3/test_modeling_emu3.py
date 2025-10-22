"""Adapted from https://github.com/huggingface/transformers/blob/main/tests/models/emu3/test_modeling_emu3.py."""
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
from transformers import Emu3Config, Emu3TextConfig

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-2, "fp16": 5e-2, "bf16": 5e-1}
MODES = [1]


class Emu3Text2TextModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        use_cache=False,
        attn_implementation="eager",
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.use_cache = use_cache
        self.attn_implementation = attn_implementation

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)
        attention_mask = np.not_equal(input_ids, 1)

        config = self.get_config()

        return config, input_ids, attention_mask

    def get_config(self):
        return Emu3TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            is_decoder=False,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=self.use_cache,
            attn_implementation=self.attn_implementation,
        )

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
        ) = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


class Emu3Vision2TextModelTester:
    def __init__(
        self,
        batch_size=13,
        seq_length=7,
        is_training=False,
        vocab_size=99,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=37,
        max_position_embeddings=512,
        initializer_range=0.02,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        image_token_id=3,
        image_size=30,
        codebook_size=20,
        temporal_downsample_factor=1,
        base_channels=32,
        vq_channel_multiplier=[1, 1],
        image_seq_length=100,
        vq_img_token_start_id=3,
        use_cache=False,
        attn_implementation="eager",
    ):
        self.batch_size = batch_size
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.image_token_id = image_token_id
        self.image_size = image_size
        self.codebook_size = codebook_size
        self.temporal_downsample_factor = temporal_downsample_factor
        self.vq_channel_multiplier = vq_channel_multiplier
        self.vq_img_token_start_id = vq_img_token_start_id
        self.base_channels = base_channels
        self.seq_length = seq_length + image_seq_length
        self.image_seq_length = image_seq_length
        self.use_cache = use_cache
        self.attn_implementation = attn_implementation

    def prepare_config_and_inputs(self):
        config = self.get_config()

        input_ids = ids_numpy([self.batch_size, self.seq_length], config.text_config.vocab_size)
        attention_mask = np.not_equal(input_ids, 1)
        input_ids[input_ids == self.image_token_id] = self.pad_token_id
        input_ids[:, : self.image_seq_length] = self.image_token_id

        pixel_values = floats_numpy(
            [
                self.batch_size,
                3,
                self.image_size,
                self.image_size,
            ]
        )
        image_sizes = [[self.image_size, self.image_size]] * self.batch_size
        image_sizes = np.array(image_sizes)

        return config, input_ids, attention_mask, pixel_values, image_sizes

    def get_config(self):
        # create dummy vocab map for image2bpe mapping if it needs remapping
        # we assume that vocab size is big enough to account for `codebook_size` amount of
        # image tokens somewhere at the beginning of total vocab size

        vocab_map = {i: chr(i) for i in range(self.vocab_size)}
        start = self.vq_img_token_start_id
        end = self.vq_img_token_start_id + self.codebook_size
        for i in range(start, end):
            # dummy str for each token, anything that fits pattern "<|visual token XXXXXX|>"
            vocab_map[i] = f"<|visual token{i:06d}|>"

        # add tokens that have to be in the vocab, we'll retrieve their ids later in modeling code
        vocab_map[self.image_token_id] = "<image>"
        vocab_map[self.image_token_id + 1] = "<|extra_200|>"
        vocab_map = {v: k for k, v in vocab_map.items()}

        text_config = Emu3TextConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            intermediate_size=self.intermediate_size,
            max_position_embeddings=self.max_position_embeddings,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            use_cache=self.use_cache,
            attn_implementation=self.attn_implementation,
        )

        vq_config = {
            "codebook_size": self.codebook_size,
            "temporal_downsample_factor": self.temporal_downsample_factor,
            "base_channels": self.base_channels,
            "channel_multiplier": self.vq_channel_multiplier,
            "hidden_size": self.base_channels,
        }
        return Emu3Config(text_config=text_config, vq_config=vq_config, vocabulary_map=vocab_map)

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        (
            config,
            input_ids,
            attention_mask,
            pixel_values,
            image_sizes,
        ) = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_sizes": image_sizes,
            "logits_to_keep": 1,
        }
        return config, inputs_dict


text2text_model_tester = Emu3Text2TextModelTester()
(
    text_config,
    text2text_input_dict,
) = text2text_model_tester.prepare_config_and_inputs_for_common()
vision2text_model_tester = Emu3Vision2TextModelTester()
(
    config,
    vision2text_input_dict,
) = vision2text_model_tester.prepare_config_and_inputs_for_common()


EMU3_CASES = [
    [
        "Emu3TextModel",
        "transformers.Emu3TextModel",
        "mindone.transformers.Emu3TextModel",
        (text_config,),
        {},
        (),
        text2text_input_dict,
        {
            "last_hidden_state": 0,
        },
    ],
    [
        "Emu3ForConditionalGeneration",
        "transformers.Emu3ForConditionalGeneration",
        "mindone.transformers.Emu3ForConditionalGeneration",
        (config,),
        {},
        (),
        vision2text_input_dict,
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
        for case in EMU3_CASES
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
    (
        pt_inputs_args,
        pt_inputs_kwargs,
        ms_inputs_args,
        ms_inputs_kwargs,
    ) = generalized_parse_args(pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs)

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
