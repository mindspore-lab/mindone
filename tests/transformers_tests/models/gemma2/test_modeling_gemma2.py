# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import Gemma2Config

import mindspore as ms

from tests.modeling_test_utils import forward_compare

from ..gemma.test_modeling_gemma import GemmaModelTester

DTYPE_AND_THRESHOLDS = {"fp32": 5e-4, "fp16": 5e-3, "bf16": 1e-2}
MODES = [0, 1]


class Gemma2ModelTester(GemmaModelTester):
    config_class = Gemma2Config


class Gemma2ModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = Gemma2ModelTester()

    @parameterized.expand(
        [(dtype,) + (mode,) for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        pt_module = "transformers.Gemma2Model"
        ms_module = "mindone.transformers.Gemma2Model"
        config, input_ids, _, input_mask = self.model_tester.prepare_config_and_inputs()[:4]
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (input_ids,)
        inputs_kwargs = {"attention_mask": input_mask}
        outputs_map = {"last_hidden_state": 0}

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For Gemma2Model forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )
