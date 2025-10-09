# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import pytest
import transformers
from transformers import BertConfig
from transformers.testing_utils import DUMMY_UNKNOWN_IDENTIFIER, RequestCounter, slow

from mindone.transformers.testing_utils import require_mindspore
from mindone.transformers.utils import is_mindspore_available

sys.path.append(str(Path(__file__).parent.parent.parent.parent / "utils"))


if is_mindspore_available():
    from transformers import AutoConfig

    from mindone.transformers import AutoModel, AutoModelForMaskedLM, BertForMaskedLM, BertModel


@require_mindspore
class AutoModelTest(unittest.TestCase):
    def setUp(self):
        transformers.dynamic_module_utils.TIME_OUT_REMOTE_CODE = 0

    @slow
    def test_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModel.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertModel)

    @slow
    def test_model_for_masked_lm(self):
        model_name = "google-bert/bert-base-uncased"
        config = AutoConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, BertConfig)

        model = AutoModelForMaskedLM.from_pretrained(model_name)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, BertForMaskedLM)

    def test_repo_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, "bert-base is not a local folder and is not a valid model identifier"
        ):
            _ = AutoModel.from_pretrained("bert-base")

    def test_revision_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError, r"aaaaaa is not a valid git identifier \(branch name, tag name or commit id\)"
        ):
            _ = AutoModel.from_pretrained(DUMMY_UNKNOWN_IDENTIFIER, revision="aaaaaa")

    def test_model_file_not_found(self):
        with self.assertRaisesRegex(
            EnvironmentError,
            "hf-internal-testing/config-no-model does not appear to have a file named pytorch_model.bin",
        ):
            _ = AutoModel.from_pretrained("hf-internal-testing/config-no-model")

    # TODO due to unstable network connection with hf, this test need to be skipped right now
    @pytest.mark.skip
    def test_cached_model_has_minimum_calls_to_head(self):
        # Make sure we have cached the model.
        _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        with RequestCounter() as counter:
            _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        self.assertEqual(counter["GET"], 0)
        self.assertEqual(counter["HEAD"], 1)
        self.assertEqual(counter.total_calls, 1)

        # With a sharded checkpoint
        _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
        with RequestCounter() as counter:
            _ = AutoModel.from_pretrained("hf-internal-testing/tiny-random-bert-sharded")
        self.assertEqual(counter["GET"], 0)
        self.assertEqual(counter["HEAD"], 1)
        self.assertEqual(counter.total_calls, 1)

    def test_attr_not_existing(self):
        from mindone.transformers.models.auto.auto_factory import _LazyAutoMapping

        _CONFIG_MAPPING_NAMES = OrderedDict([("bert", "BertConfig")])
        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "GhostModel")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)

        with pytest.raises(ValueError, match=r"Could not find GhostModel neither in .* nor in .*!"):
            _MODEL_MAPPING[BertConfig]

        _MODEL_MAPPING_NAMES = OrderedDict([("bert", "BertModel")])
        _MODEL_MAPPING = _LazyAutoMapping(_CONFIG_MAPPING_NAMES, _MODEL_MAPPING_NAMES)
        self.assertEqual(_MODEL_MAPPING[BertConfig], BertModel)
