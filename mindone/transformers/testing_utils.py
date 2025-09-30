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

import copy
import functools
import os
import time
import unittest
from typing import Optional

from transformers import logging as transformers_logging
from transformers.utils import strtobool

import mindspore as ms

from .utils import is_mindspore_available

logger = transformers_logging.get_logger(__name__)


def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)


def slow(test_case):
    """
    Decorator marking a test as slow.

    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.

    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)


def set_config_for_less_flaky_test(config):
    target_attrs = [
        "rms_norm_eps",
        "layer_norm_eps",
        "norm_eps",
        "norm_epsilon",
        "layer_norm_epsilon",
        "batch_norm_eps",
    ]
    for target_attr in target_attrs:
        setattr(config, target_attr, 1.0)

    # norm layers (layer/group norm, etc.) could cause flaky tests when the tensors have very small variance.
    # (We don't need the original epsilon values to check eager/sdpa matches)
    attrs = ["text_config", "vision_config", "text_encoder", "audio_encoder", "decoder"]
    for attr in attrs:
        if hasattr(config, attr):
            for target_attr in target_attrs:
                setattr(getattr(config, attr), target_attr, 1.0)


def set_model_for_less_flaky_test(model):
    # Another way to make sure norm layers have desired epsilon. (Some models don't set it from its config.)
    target_names = (
        "LayerNorm",
        "GroupNorm",
        "BatchNorm",
        "RMSNorm",
        "BatchNorm2d",
        "BatchNorm1d",
        "BitGroupNormActivation",
        "WeightStandardizedConv2d",
    )
    target_attrs = ["eps", "epsilon", "variance_epsilon"]
    if is_mindspore_available() and isinstance(model, ms.nn.Module):
        for module in model.modules():
            if type(module).__name__.endswith(target_names):
                for attr in target_attrs:
                    if hasattr(module, attr):
                        setattr(module, attr, 1.0)


def set_model_tester_for_less_flaky_test(test_case):
    target_num_hidden_layers = 1
    # TODO (if possible): Avoid exceptional cases
    exceptional_classes = [
        "ZambaModelTester",
        "Zamba2ModelTester",
        "RwkvModelTester",
        "AriaVisionText2TextModelTester",
        "GPTNeoModelTester",
        "DPTModelTester",
    ]
    if test_case.model_tester.__class__.__name__ in exceptional_classes:
        target_num_hidden_layers = None
    if hasattr(test_case.model_tester, "out_features") or hasattr(test_case.model_tester, "out_indices"):
        target_num_hidden_layers = None

    if hasattr(test_case.model_tester, "num_hidden_layers") and target_num_hidden_layers is not None:
        test_case.model_tester.num_hidden_layers = target_num_hidden_layers
    if (
        hasattr(test_case.model_tester, "vision_config")
        and "num_hidden_layers" in test_case.model_tester.vision_config
        and target_num_hidden_layers is not None
    ):
        test_case.model_tester.vision_config = copy.deepcopy(test_case.model_tester.vision_config)
        if isinstance(test_case.model_tester.vision_config, dict):
            test_case.model_tester.vision_config["num_hidden_layers"] = 1
        else:
            test_case.model_tester.vision_config.num_hidden_layers = 1
    if (
        hasattr(test_case.model_tester, "text_config")
        and "num_hidden_layers" in test_case.model_tester.text_config
        and target_num_hidden_layers is not None
    ):
        test_case.model_tester.text_config = copy.deepcopy(test_case.model_tester.text_config)
        if isinstance(test_case.model_tester.text_config, dict):
            test_case.model_tester.text_config["num_hidden_layers"] = 1
        else:
            test_case.model_tester.text_config.num_hidden_layers = 1

    # A few model class specific handling

    # For Albert
    if hasattr(test_case.model_tester, "num_hidden_groups"):
        test_case.model_tester.num_hidden_groups = test_case.model_tester.num_hidden_layers


def is_flaky(max_attempts: int = 5, wait_before_retry: Optional[float] = None, description: Optional[str] = None):
    """
    To decorate flaky tests. They will be retried on failures.

    Please note that our push tests use `pytest-rerunfailures`, which prompts the CI to rerun certain types of
    failed tests. More specifically, if the test exception contains any substring in `FLAKY_TEST_FAILURE_PATTERNS`
    (in `.circleci/create_circleci_config.py`), it will be rerun. If you find a recurrent pattern of failures,
    expand `FLAKY_TEST_FAILURE_PATTERNS` in our CI configuration instead of using `is_flaky`.

    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
        description (`str`, *optional*):
            A string to describe the situation (what / where / why is flaky, link to GH issue/PR comments, errors,
            etc.)
    """

    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    logger.error(f"Test failed with {err} at try {retry_count}/{max_attempts}.")
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator


def require_mindspore(test_case):
    """
    Decorator marking a test that requires MindSpore.

    These tests are skipped when MindSpore isn't installed.

    """
    return unittest.skipUnless(is_mindspore_available(), "test requires MindSpore")(test_case)
