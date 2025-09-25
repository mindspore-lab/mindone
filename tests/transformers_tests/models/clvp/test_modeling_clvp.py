"""Adapted from https://github.com/huggingface/transformers/tree/main/tests//models/clvp/test_modeling_clvp.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

# import datasets
import numpy as np
import pytest
import torch
from transformers import ClvpConfig, ClvpDecoderConfig, ClvpEncoderConfig  # ClvpFeatureExtractor

import mindspore as ms

from tests.modeling_test_utils import compute_diffs, generalized_parse_args, get_modules
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy, random_attention_mask

DTYPE_AND_THRESHOLDS = {"fp32": 5e-6, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class ClvpEncoderTester:
    def __init__(
        self,
        batch_size=2,
        seq_length=7,
        is_training=False,
        use_input_mask=True,
        use_labels=True,
        vocab_size=50,
        hidden_size=128,
        projection_dim=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=32,
        dropout=0.1,
        attention_dropout=0.1,
        initializer_range=0.02,
        scope=None,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.projection_dim = projection_dim
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.scope = scope
        self.bos_token_id = vocab_size - 1
        self.eos_token_id = vocab_size - 1

    def get_config(self):
        encoder_config = ClvpEncoderConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            projection_dim=self.projection_dim,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            intermediate_size=self.intermediate_size,
            dropout=self.dropout,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

        return encoder_config

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        encoder_config = self.get_config()

        return encoder_config, input_ids, input_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        speech_config, input_ids, input_mask = config_and_inputs
        inputs_dict = {"input_ids": input_ids, "attention_mask": input_mask}
        return speech_config, inputs_dict


class ClvpDecoderTester:
    def __init__(
        self,
        batch_size=2,
        seq_length=3,
        is_training=False,
        vocab_size=300,
        max_position_embeddings=256,
        max_text_tokens=256,
        use_input_mask=True,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        bos_token_id=97,
        eos_token_id=98,
        relative_attention_num_buckets=4,
        relative_attention_max_distance=16,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.max_text_tokens = max_text_tokens
        self.use_input_mask = use_input_mask
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance

    def get_config(self):
        decoder_config = ClvpDecoderConfig(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            max_text_tokens=self.max_text_tokens,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            relative_attention_num_buckets=self.relative_attention_num_buckets,
            relative_attention_max_distance=self.relative_attention_max_distance,
        )

        return decoder_config

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        if input_mask is not None:
            batch_size, seq_length = input_mask.shape
            rnd_start_indices = np.random.randint(1, seq_length - 1, size=(batch_size,))
            for batch_idx, start_index in enumerate(rnd_start_indices):
                input_mask[batch_idx, :start_index] = 1
                input_mask[batch_idx, start_index:] = 0

        decoder_config = self.get_config()

        return decoder_config, input_ids, input_mask

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        return config, inputs_dict


class ClvpModelForConditionalGenerationTester:
    def __init__(self, is_training=False):
        self.clvp_encoder_tester = ClvpEncoderTester()
        self.is_training = is_training
        self.batch_size = self.clvp_encoder_tester.batch_size  # need bs for batching_equivalence test

    def get_config(self):
        decoder_config = ClvpDecoderConfig(
            vocab_size=50,
            max_position_embeddings=30,
            max_text_tokens=30,
            hidden_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            bos_token_id=97,
            eos_token_id=98,
            relative_attention_num_buckets=4,
            relative_attention_max_distance=16,
        )
        text_config = self.clvp_encoder_tester.get_config()
        speech_config = self.clvp_encoder_tester.get_config()
        speech_config.vocab_size = 300

        return ClvpConfig.from_sub_model_configs(
            text_config,
            speech_config,
            decoder_config,
            projection_dim=16,
        )

    def prepare_config_and_inputs(self):
        _, input_ids, attention_mask = self.clvp_encoder_tester.prepare_config_and_inputs()

        # ds = datasets.load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        # ds = ds.cast_column("audio", datasets.Audio(sampling_rate=22050))
        # _, audio, sr = ds.sort("id").select(range(1))[:1]["audio"][0].values()
        # sr = 22050
        # feature_extractor = ClvpFeatureExtractor()
        # input_features = feature_extractor(raw_speech=audio, sampling_rate=sr, return_tensors="np")["input_features"]
        input_features = floats_numpy([1, 80, 517])

        config = self.get_config()

        return config, input_ids, attention_mask, input_features

    def prepare_config_and_inputs_for_common(self):
        config_and_inputs = self.prepare_config_and_inputs()
        config, input_ids, attention_mask, input_features = config_and_inputs
        inputs_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "return_loss": False,
        }
        return config, inputs_dict


model_tester = ClvpEncoderTester()
encoder_config, encoder_inputs_dict = model_tester.prepare_config_and_inputs_for_common()
model_tester = ClvpDecoderTester()
decoder_config, decoder_inputs_dict = model_tester.prepare_config_and_inputs_for_common()
model_tester = ClvpModelForConditionalGenerationTester()
config, inputs_dict = model_tester.prepare_config_and_inputs_for_common()


CLVP_CASES = [
    [
        "ClvpEncoder",
        "transformers.ClvpEncoder",
        "mindone.transformers.ClvpEncoder",
        (encoder_config,),
        {},
        (),
        encoder_inputs_dict,
        {
            "embeds": 0,
        },
    ],
    [
        "ClvpForCausalLM",
        "transformers.ClvpForCausalLM",
        "mindone.transformers.ClvpForCausalLM",
        (decoder_config,),
        {},
        (),
        decoder_inputs_dict,
        {
            "logits": 0,
        },
    ],
    [
        "ClvpModelForConditionalGeneration",
        "transformers.ClvpModelForConditionalGeneration",
        "mindone.transformers.ClvpModelForConditionalGeneration",
        (config,),
        {},
        (),
        inputs_dict,
        {
            "logits_per_speech": 0,
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
        for case in CLVP_CASES
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
