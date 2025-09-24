"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/hubert/test_modeling_hubert.py."""

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
from transformers import BarkConfig, BarkSemanticConfig, BarkCoarseConfig, BarkFineConfig

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import ids_numpy, random_attention_mask
from ..encodec.test_modeling_encodec import EncodecModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]


class BarkSemanticModelTester:
    def __init__(
        self,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=4,
        is_training=False,  # for now training is not supported
        use_input_mask=True,
        use_labels=True,
        vocab_size=33,
        output_vocab_size=33,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=15,
        dropout=0.1,
        window_size=256,
        initializer_range=0.02,
        n_codes_total=8,  # for BarkFineModel
        n_codes_given=1,  # for BarkFineModel
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.window_size = window_size
        self.initializer_range = initializer_range
        self.bos_token_id = output_vocab_size - 1
        self.eos_token_id = output_vocab_size - 1
        self.pad_token_id = output_vocab_size - 1

        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        head_mask = ids_numpy([self.num_hidden_layers, self.num_attention_heads], 2)

        inputs_dict = {
            "input_ids": input_ids,
            "head_mask": head_mask,
            "attention_mask": input_mask,
        }

        return config, inputs_dict

    def get_config(self):
        return BarkSemanticConfig(
            vocab_size=self.vocab_size,
            output_vocab_size=self.output_vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            window_size=self.window_size,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        config.output_vocab_size = 300
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


class BarkCoarseModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=4,
        is_training=False,  # for now training is not supported
        use_input_mask=True,
        use_labels=True,
        vocab_size=33,
        output_vocab_size=33,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=15,
        dropout=0.1,
        window_size=256,
        initializer_range=0.02,
        n_codes_total=8,  # for BarkFineModel
        n_codes_given=1,  # for BarkFineModel
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.window_size = window_size
        self.initializer_range = initializer_range
        self.bos_token_id = output_vocab_size - 1
        self.eos_token_id = output_vocab_size - 1
        self.pad_token_id = output_vocab_size - 1

        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        head_mask = ids_numpy([self.num_hidden_layers, self.num_attention_heads], 2)

        inputs_dict = {
            "input_ids": input_ids,
            "head_mask": head_mask,
            "attention_mask": input_mask,
        }

        return config, inputs_dict

    def get_config(self):
        return BarkCoarseConfig(
            vocab_size=self.vocab_size,
            output_vocab_size=self.output_vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            window_size=self.window_size,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        config.output_vocab_size = 300
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


class BarkFineModelTester:
    def __init__(
        self,
        parent,
        batch_size=3,  # need batch_size != num_hidden_layers
        seq_length=4,
        is_training=False,  # for now training is not supported
        use_input_mask=True,
        use_labels=True,
        vocab_size=33,
        output_vocab_size=33,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=15,
        dropout=0.1,
        window_size=256,
        initializer_range=0.02,
        n_codes_total=8,  # for BarkFineModel
        n_codes_given=1,  # for BarkFineModel
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.is_training = is_training
        self.use_input_mask = use_input_mask
        self.use_labels = use_labels
        self.vocab_size = vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.window_size = window_size
        self.initializer_range = initializer_range
        self.bos_token_id = output_vocab_size - 1
        self.eos_token_id = output_vocab_size - 1
        self.pad_token_id = output_vocab_size - 1

        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        self.is_encoder_decoder = False

    def prepare_config_and_inputs(self):
        input_ids = ids_numpy([self.batch_size, self.seq_length, self.n_codes_total], self.vocab_size)

        input_mask = None
        if self.use_input_mask:
            input_mask = random_attention_mask([self.batch_size, self.seq_length])

        config = self.get_config()

        head_mask = ids_numpy([self.num_hidden_layers, self.num_attention_heads], 2)

        # randint between self.n_codes_given - 1 and self.n_codes_total - 1
        codebook_idx = ids_numpy((1,), self.n_codes_total - self.n_codes_given).item() + self.n_codes_given

        inputs_dict = {
            "codebook_idx": codebook_idx,
            "input_ids": input_ids,
            "head_mask": head_mask,
            "attention_mask": input_mask,
        }

        return config, inputs_dict

    def get_config(self):
        return BarkFineConfig(
            vocab_size=self.vocab_size,
            output_vocab_size=self.output_vocab_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_hidden_layers,
            num_heads=self.num_attention_heads,
            use_cache=True,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            window_size=self.window_size,
        )

    def get_pipeline_config(self):
        config = self.get_config()
        config.vocab_size = 300
        config.output_vocab_size = 300
        return config

    def prepare_config_and_inputs_for_common(self):
        config, inputs_dict = self.prepare_config_and_inputs()
        return config, inputs_dict


class BarkModelTester:
    def __init__(
        self,
        parent,
        semantic_kwargs=None,
        coarse_acoustics_kwargs=None,
        fine_acoustics_kwargs=None,
        codec_kwargs=None,
        is_training=False,  # for now training is not supported
    ):
        if semantic_kwargs is None:
            semantic_kwargs = {}
        if coarse_acoustics_kwargs is None:
            coarse_acoustics_kwargs = {}
        if fine_acoustics_kwargs is None:
            fine_acoustics_kwargs = {}
        if codec_kwargs is None:
            codec_kwargs = {}

        self.parent = parent
        self.semantic_model_tester = BarkSemanticModelTester(parent, **semantic_kwargs)
        self.coarse_acoustics_model_tester = BarkCoarseModelTester(parent, **coarse_acoustics_kwargs)
        self.fine_acoustics_model_tester = BarkFineModelTester(parent, **fine_acoustics_kwargs)
        self.codec_model_tester = EncodecModelTester(parent, **codec_kwargs)

        self.is_training = is_training

    def get_config(self):
        return BarkConfig.from_sub_model_configs(
            self.semantic_model_tester.get_config(),
            self.coarse_acoustics_model_tester.get_config(),
            self.fine_acoustics_model_tester.get_config(),
            self.codec_model_tester.get_config(),
        )

    def get_pipeline_config(self):
        config = self.get_config()

        # follow the `get_pipeline_config` of the sub component models
        config.semantic_config.vocab_size = 300
        config.coarse_acoustics_config.vocab_size = 300
        config.fine_acoustics_config.vocab_size = 300

        config.semantic_config.output_vocab_size = 300
        config.coarse_acoustics_config.output_vocab_size = 300
        config.fine_acoustics_config.output_vocab_size = 300

        return config


model_tester = BarkModelTester()
(
    config,
    input_values,
    attention_mask,
) = model_tester.prepare_config_and_inputs()


BARK_CASES = [
    [
        "BarkModel",
        "transformers.BarkModel",
        "mindone.transformers.BarkModel",
        (config,),
        {},
        (input_values,),
        {
            "attention_mask": attention_mask,
        },
        {
            "last_hidden_state": 0,  # key: torch attribute, value: mindspore idx
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
        for case in BARK_CASES
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
