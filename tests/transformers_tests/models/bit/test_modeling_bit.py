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
from transformers import BitConfig

import mindspore as ms

from tests.modeling_test_utils import forward_compare
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

# PadV3 not support bf16
DTYPE_AND_THRESHOLDS = {"fp32": 5e-3, "fp16": 5e-2}
MODES = [0, 1]


class BitModelTester:
    def __init__(
        self,
        batch_size=3,
        image_size=32,
        num_channels=3,
        embeddings_size=10,
        hidden_sizes=[8, 16, 32, 64],
        depths=[1, 1, 2, 1],
        is_training=True,
        use_labels=True,
        hidden_act="relu",
        num_labels=3,
        scope=None,
        out_features=["stage2", "stage3", "stage4"],
        out_indices=[2, 3, 4],
        num_groups=1,
    ):
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = num_channels
        self.embeddings_size = embeddings_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.is_training = is_training
        self.use_labels = use_labels
        self.hidden_act = hidden_act
        self.num_labels = num_labels
        self.scope = scope
        self.num_stages = len(hidden_sizes)
        self.out_features = out_features
        self.out_indices = out_indices
        self.num_groups = num_groups

    def prepare_config_and_inputs(self):
        pixel_values = floats_numpy([self.batch_size, self.num_channels, self.image_size, self.image_size])

        labels = None
        if self.use_labels:
            labels = ids_numpy([self.batch_size], self.num_labels)

        config = self.get_config()

        return config, pixel_values, labels

    def get_config(self):
        return BitConfig(
            num_channels=self.num_channels,
            embeddings_size=self.embeddings_size,
            hidden_sizes=self.hidden_sizes,
            depths=self.depths,
            hidden_act=self.hidden_act,
            num_labels=self.num_labels,
            out_features=self.out_features,
            out_indices=self.out_indices,
            num_groups=self.num_groups,
        )


class BitModelTest(unittest.TestCase):
    def setUp(self):
        self.model_tester = BitModelTester()

    @parameterized.expand(
        [[dtype,] + [mode,] for dtype in DTYPE_AND_THRESHOLDS for mode in MODES]
    )
    def test_model_forward(self, dtype, mode):
        ms.set_context(mode=mode)
        name = "BitBackbone"
        pt_module = f"transformers.{name}"
        ms_module = f"mindone.transformers.{name}"
        config, pixel_values, labels = self.model_tester.prepare_config_and_inputs()
        init_args = (config,)
        init_kwargs = {}
        inputs_args = (pixel_values,)
        inputs_kwargs = {}
        outputs_map = {"feature_maps": 0}

        diffs, pt_dtype, ms_dtype = forward_compare(
            pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map, dtype
        )

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        self.assertTrue(
            (np.array(diffs) < THRESHOLD).all(),
            f"For {name} forward test, mode: {mode}, ms_dtype: {ms_dtype}, pt_type:{pt_dtype},"
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )
