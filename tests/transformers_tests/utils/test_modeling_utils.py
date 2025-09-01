# coding=utf-8
# Copyright 2019 HuggingFace Inc.
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
import os
import os.path
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest
import unittest.mock as mock
import uuid
from pathlib import Path

import requests
from huggingface_hub import HfApi, HfFolder
from parameterized import parameterized
from pytest import mark
from requests.exceptions import HTTPError
from transformers import PretrainedConfig, logging
from transformers.testing_utils import (
    TOKEN,
    CaptureLogger,
    LoggingLevel,
    TemporaryHubRepo,
    TestCasePlus,
    hub_retry,
    is_staging_test,
    require_accelerate,
    require_read_token,
    require_safetensors,
    require_usr_bin_time,
    slow,
)

from mindone.transformers import AutoConfig, AutoModelForImageClassification, AutoModelForSequenceClassification
from mindone.transformers.testing_utils import require_mindspore
from mindone.transformers.utils import is_mindspore_available

sys.path.append(str(Path(__file__).parent.parent.parent / "utils"))

if is_mindspore_available():
    from transformers import BertConfig

    import mindspore as ms
    from mindspore import Parameter, mint, nn

    from mindone.transformers import (
        AutoModelForCausalLM,
        BertModel,
        T5ForConditionalGeneration,
    )
    from mindone.transformers.modeling_attn_mask_utils import (
        AttentionMaskConverter,
        _create_4d_causal_attention_mask,
        _prepare_4d_attention_mask,
        _prepare_4d_causal_attention_mask,
        dtype_to_min,
    )
    from mindone.transformers.modeling_utils import PreTrainedModel, dtype_byte_size

    # Fake pretrained models for tests
    class BaseModel(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.linear = mint.nn.Linear(5, 5)
            self.linear_2 = mint.nn.Linear(5, 5)

        def construct(self, x):
            return self.linear_2(self.linear(x))

    class BaseModelWithTiedWeights(PreTrainedModel):
        config_class = PretrainedConfig

        def __init__(self, config):
            super().__init__(config)
            self.linear = mint.nn.Linear(5, 5)
            self.linear_2 = mint.nn.Linear(5, 5)

        def construct(self, x):
            return self.linear_2(self.linear(x))

        def tie_weights(self):
            self.linear_2.weight = self.linear.weight

    class ModelWithHead(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def _init_weights(self, module):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.base = BaseModel(config)
            # linear is a common name between Base and Head on purpose.
            self.linear = mint.nn.Linear(5, 5)
            self.linear2 = mint.nn.Linear(5, 5)

        def construct(self, x):
            return self.linear2(self.linear(self.base(x)))

    class ModelWithHeadAndTiedWeights(PreTrainedModel):
        base_model_prefix = "base"
        config_class = PretrainedConfig

        def _init_weights(self, module):
            pass

        def __init__(self, config):
            super().__init__(config)
            self.base = BaseModel(config)
            self.decoder = mint.nn.Linear(5, 5)

        def construct(self, x):
            return self.decoder(self.base(x))

        def tie_weights(self):
            self.decoder.weight = self.base.linear.weight

    class Prepare4dCausalAttentionMaskModel(nn.Cell):
        def construct(self, inputs_embeds):
            batch_size, seq_length, _ = inputs_embeds.shape
            past_key_values_length = 4
            attention_mask = _prepare_4d_causal_attention_mask(
                None, (batch_size, seq_length), inputs_embeds, past_key_values_length
            )
            return attention_mask

    class Create4dCausalAttentionMaskModel(nn.Cell):
        def construct(self, inputs_embeds):
            batch_size, seq_length, _ = inputs_embeds.shape
            past_key_values_length = 4
            attention_mask = _create_4d_causal_attention_mask(
                (batch_size, seq_length),
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
            return attention_mask

    class Prepare4dAttentionMaskModel(nn.Cell):
        def construct(self, mask, inputs_embeds):
            attention_mask = _prepare_4d_attention_mask(mask, dtype=inputs_embeds.dtype)
            return attention_mask

    class TestOffline(unittest.TestCase):
        def test_offline(self):
            # Ugly setup with monkeypatches, amending env vars here is too late as libs have already been imported
            from huggingface_hub import constants
            from transformers.utils import hub

            offlfine_env = hub._is_offline_mode
            hub_cache_env = constants.HF_HUB_CACHE
            hub_cache_env1 = constants.HUGGINGFACE_HUB_CACHE
            default_cache = constants.default_cache_path
            transformers_cache = hub.TRANSFORMERS_CACHE

            try:
                hub._is_offline_mode = True
                with tempfile.TemporaryDirectory() as tmpdir:
                    LOG.info("Temporary cache dir %s", tmpdir)
                    constants.HF_HUB_CACHE = tmpdir
                    constants.HUGGINGFACE_HUB_CACHE = tmpdir
                    constants.default_cache_path = tmpdir
                    hub.TRANSFORMERS_CACHE = tmpdir
                    # First offline load should fail
                    try:
                        AutoModelForImageClassification.from_pretrained(
                            TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None
                        )
                    except OSError:
                        LOG.info("Loading model %s in offline mode failed as expected", TINY_IMAGE_CLASSIF)
                    else:
                        self.fail("Loading model {} in offline mode should fail".format(TINY_IMAGE_CLASSIF))

                    # Download model -> Huggingface Hub not concerned by our offline mode
                    LOG.info("Downloading %s for offline tests", TINY_IMAGE_CLASSIF)
                    hub_api = HfApi()
                    local_dir = hub_api.snapshot_download(TINY_IMAGE_CLASSIF, cache_dir=tmpdir)

                    LOG.info("Model %s downloaded in %s", TINY_IMAGE_CLASSIF, local_dir)

                    AutoModelForImageClassification.from_pretrained(
                        TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None
                    )
            finally:
                # Tear down: reset env as it was before calling this test
                hub._is_offline_mode = offlfine_env
                constants.HF_HUB_CACHE = hub_cache_env
                constants.HUGGINGFACE_HUB_CACHE = hub_cache_env1
                constants.default_cache_path = default_cache
                hub.TRANSFORMERS_CACHE = transformers_cache

        def test_local_files_only(self):
            # Ugly setup with monkeypatches, amending env vars here is too late as libs have already been imported
            from huggingface_hub import constants
            from transformers.utils import hub

            hub_cache_env = constants.HF_HUB_CACHE
            hub_cache_env1 = constants.HUGGINGFACE_HUB_CACHE
            default_cache = constants.default_cache_path
            transformers_cache = hub.TRANSFORMERS_CACHE
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    LOG.info("Temporary cache dir %s", tmpdir)
                    constants.HF_HUB_CACHE = tmpdir
                    constants.HUGGINGFACE_HUB_CACHE = tmpdir
                    constants.default_cache_path = tmpdir
                    hub.TRANSFORMERS_CACHE = tmpdir
                    try:
                        AutoModelForImageClassification.from_pretrained(
                            TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None, local_files_only=True
                        )
                    except OSError:
                        LOG.info("Loading model %s in offline mode failed as expected", TINY_IMAGE_CLASSIF)
                    else:
                        self.fail("Loading model {} in offline mode should fail".format(TINY_IMAGE_CLASSIF))

                    LOG.info("Downloading %s for offline tests", TINY_IMAGE_CLASSIF)
                    hub_api = HfApi()
                    local_dir = hub_api.snapshot_download(TINY_IMAGE_CLASSIF, cache_dir=tmpdir)

                    LOG.info("Model %s downloaded in %s", TINY_IMAGE_CLASSIF, local_dir)

                    AutoModelForImageClassification.from_pretrained(
                        TINY_IMAGE_CLASSIF, revision="main", use_auth_token=None, local_files_only=True
                    )
            finally:
                # Tear down: reset env as it was before calling this test
                constants.HF_HUB_CACHE = hub_cache_env
                constants.HUGGINGFACE_HUB_CACHE = hub_cache_env1
                constants.default_cache_path = default_cache
                hub.TRANSFORMERS_CACHE = transformers_cache


TINY_T5 = "patrickvonplaten/t5-tiny-random"
TINY_BERT_FOR_TOKEN_CLASSIFICATION = "hf-internal-testing/tiny-bert-for-token-classification"
TINY_MISTRAL = "hf-internal-testing/tiny-random-MistralForCausalLM"
TINY_IMAGE_CLASSIF = "hf-internal-testing/tiny-random-SiglipForImageClassification"
TINY_LLAVA = "hf-internal-testing/tiny-random-LlavaForConditionalGeneration"

LOG = logging.get_logger(__name__)


def check_models_equal(model1, model2):
    models_are_equal = True
    for model1_p, model2_p in zip(model1.get_parameters(), model2.get_parameters()):
        if model1_p.data.ne(model2_p.data).sum() > 0:
            models_are_equal = False

    return models_are_equal


@require_mindspore
class ModelUtilsTest(TestCasePlus):
    def setUp(self):
        self.old_dtype = ms.float32
        super().setUp()

    def tearDown(self):
        # fixme how to align with transformers on "torch.set_default_dtype"
        super().tearDown()

    def test_hub_retry(self):
        @hub_retry(max_attempts=2)
        def test_func():
            # First attempt will fail with a connection error
            if not hasattr(test_func, "attempt"):
                test_func.attempt = 1
                raise requests.exceptions.ConnectionError("Connection failed")
            # Second attempt will succeed
            return True

        self.assertTrue(test_func())

    @slow
    def test_model_from_pretrained(self):
        model_name = "google-bert/bert-base-uncased"
        config = BertConfig.from_pretrained(model_name)
        self.assertIsNotNone(config)
        self.assertIsInstance(config, PretrainedConfig)

        model = BertModel.from_pretrained(model_name)
        model, loading_info = BertModel.from_pretrained(model_name, output_loading_info=True)
        self.assertIsNotNone(model)
        self.assertIsInstance(model, PreTrainedModel)

        self.assertEqual(len(loading_info["missing_keys"]), 0)
        self.assertEqual(len(loading_info["unexpected_keys"]), 8)
        self.assertEqual(len(loading_info["mismatched_keys"]), 0)
        self.assertEqual(len(loading_info["error_msgs"]), 0)

        config = BertConfig.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)

        # Not sure this is the intended behavior. TODO fix Lysandre & Thom
        config.name_or_path = model_name

        model = BertModel.from_pretrained(model_name, output_attentions=True, output_hidden_states=True)
        self.assertEqual(model.config.output_hidden_states, True)
        self.assertEqual(model.config, config)

    def test_model_from_pretrained_subfolder(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder))

            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)

            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_manually_shared_disjointed_tensors_optimum(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        # Let's fuse qkv
        attn = model.encoder.layer[0].attention.self
        q = attn.query.weight
        k = attn.key.weight
        v = attn.value.weight
        # Force some shared storage
        qkv = mint.stack([q, k, v], dim=0)
        attn.query.weight = Parameter(qkv[0])
        attn.key.weight = Parameter(qkv[1])
        attn.value.weight = Parameter(qkv[2])
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir)
            model_loaded = BertModel.from_pretrained(tmp_dir)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_subfolder_sharded(self):
        config = BertConfig.from_pretrained("hf-internal-testing/tiny-random-bert")
        model = BertModel(config)

        subfolder = "bert"
        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(os.path.join(tmp_dir, subfolder), max_shard_size="10KB")

            with self.assertRaises(OSError):
                _ = BertModel.from_pretrained(tmp_dir)

            model_loaded = BertModel.from_pretrained(tmp_dir, subfolder=subfolder)

        self.assertTrue(check_models_equal(model, model_loaded))

    def test_model_from_pretrained_with_different_pretrained_model_name(self):
        model = T5ForConditionalGeneration.from_pretrained(TINY_T5, revision="refs/pr/4")
        self.assertIsNotNone(model)

        logger = logging.get_logger("transformers.configuration_utils")
        with LoggingLevel(logging.WARNING):
            with CaptureLogger(logger) as cl:
                BertModel.from_pretrained(TINY_T5, revision="refs/pr/4")
        self.assertTrue("You are using a model of type t5 to instantiate a model of type bert" in cl.out)

    @require_accelerate
    def test_model_from_pretrained_with_none_quantization_config(self):
        # Needs a device_map for to enter the low_cpu_mem branch. We also load AutoModelForSequenceClassification
        # deliberately to enter the missing keys branch.
        model = AutoModelForSequenceClassification.from_pretrained(
            TINY_MISTRAL, quantization_config=None, revision="refs/pr/1"
        )
        self.assertIsNotNone(model)

    def test_model_from_config_attn_implementation(self):
        # test that the model can be instantiated with attn_implementation of either
        # 1. config created with explicit attn_implementatation and from_config
        # 2. explicit from_config's attn_implementation argument with a config argument
        # 3. config created with explicit attn_implementatation and from_config overriding with explicit attn_implementation argument
        attn_implementation_available = ["eager"]
        # fixme there is not the same implementaion for sdpa in mindspore
        # if is_torch_sdpa_available():
        #     attn_implementation_available.append("sdpa")

        if is_mindspore_available():
            attn_implementation_available.append("flash_attention_2")

        for requested_attn_implementation in attn_implementation_available:
            config = AutoConfig.from_pretrained(TINY_MISTRAL, attn_implementation=requested_attn_implementation)
            # Ensure the config was set correctly
            self.assertEqual(config._attn_implementation, requested_attn_implementation)
            self.assertEqual(config._attn_implementation_internal, requested_attn_implementation)
            model = AutoModelForCausalLM.from_config(config)
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

            config = AutoConfig.from_pretrained(TINY_MISTRAL)
            # When the config is not set, the default is "eager"
            self.assertEqual(config._attn_implementation, "eager")
            self.assertEqual(config._attn_implementation_internal, None)
            model = AutoModelForCausalLM.from_config(config=config, attn_implementation=requested_attn_implementation)
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

            # Set a nonsense attn_implementation in the config, which should be overridden by the explicit argument
            config = AutoConfig.from_pretrained(TINY_MISTRAL, attn_implementation="foo-bar-baz")
            self.assertEqual(config._attn_implementation, "foo-bar-baz")
            self.assertEqual(config._attn_implementation_internal, "foo-bar-baz")
            model = AutoModelForCausalLM.from_config(config=config, attn_implementation=requested_attn_implementation)
            self.assertEqual(model.config._attn_implementation, requested_attn_implementation)

    def test_mindspore_dtype_byte_sizes(self):
        mindspore_dtypes_and_bytes = [
            (ms.float64, 8),
            (ms.float32, 4),
            (ms.float16, 2),
            (ms.bfloat16, 2),
            (ms.int64, 8),
            (ms.int32, 4),
            (ms.int16, 2),
            (ms.uint8, 1),
            (ms.int8, 1),
            (ms.bool_, 0.125),
        ]

        for mindspore_dtype, bytes_per_element in mindspore_dtypes_and_bytes:
            self.assertEqual(dtype_byte_size(mindspore_dtype), bytes_per_element)

    @require_safetensors
    def test_checkpoint_variant_hub_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir)
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_safetensors
    def test_checkpoint_variant_hub_sharded_safe(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises(EnvironmentError):
                _ = BertModel.from_pretrained(
                    "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir
                )
            model = BertModel.from_pretrained(
                "hf-internal-testing/tiny-random-bert-variant-sharded-safe", cache_dir=tmp_dir, variant="v2"
            )
        self.assertIsNotNone(model)

    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_functional(self):
        # test that we can use `from_pretrained(..., low_cpu_mem_usage=True)` with normal and
        # sharded models

        mnames = [
            "hf-internal-testing/tiny-random-bert-sharded",
            "hf-internal-testing/tiny-random-bert",
        ]
        for mname in mnames:
            _ = BertModel.from_pretrained(mname, low_cpu_mem_usage=True)

    @slow
    @require_usr_bin_time
    @require_accelerate
    @mark.accelerate_tests
    def test_from_pretrained_low_cpu_mem_usage_equal(self):
        # Before this would test that `from_pretrained(..., low_cpu_mem_usage=True)` uses less cpu memory than default
        # Now though these should be around the same.
        # TODO: Look for good bounds to check that their timings are near the same

        mname = "HuggingFaceTB/SmolLM-135M"

        preamble = "from transformers import AutoModel"
        one_liner_str = f'{preamble}; AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=False)'
        # Save this output as `max_rss_normal` if testing memory results
        max_rss_normal = self.python_one_liner_max_rss(one_liner_str)

        one_liner_str = f'{preamble};  AutoModel.from_pretrained("{mname}", low_cpu_mem_usage=True)'
        # Save this output as `max_rss_low_mem` if testing memory results
        max_rss_low_mem = self.python_one_liner_max_rss(one_liner_str)

        # Should be within 5MBs of each other (overhead)
        self.assertAlmostEqual(
            max_rss_normal / 1024 / 1024,
            max_rss_low_mem / 1024 / 1024,
            delta=5,
            msg="using `low_cpu_mem_usage` should incur the same memory usage in both cases.",
        )

        # if you want to compare things manually, let's first look at the size of the model in bytes
        # model = AutoModel.from_pretrained(mname, low_cpu_mem_usage=False)
        # total_numel = sum(dict((p.data_ptr(), p.numel()) for p in model.get_parameters()).values())
        # total_bytes = total_numel * 4
        # Now the diff_bytes should be very close to total_bytes, but the reports are inconsistent.
        # The easiest way to test this is to switch the model and torch.load to do all the work on
        # gpu - that way one can measure exactly the total and peak memory used. Perhaps once we add
        # functionality to load models directly on gpu, this test can be rewritten to use torch's
        # cuda memory tracking and then we should be able to do a much more precise test.

    def test_cached_files_are_used_when_internet_is_down(self):
        # A mock response for an HTTP head request to emulate server down
        response_mock = mock.Mock()
        response_mock.status_code = 500
        response_mock.headers = {}
        response_mock.raise_for_status.side_effect = HTTPError
        response_mock.json.return_value = {}

        # Download this model to make sure it's in the cache.
        _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")

        # Under the mock environment we get a 500 error when trying to reach the model.
        with mock.patch("requests.Session.request", return_value=response_mock) as mock_head:
            _ = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
            # This check we did call the fake head request
            mock_head.assert_called()

    @require_safetensors
    def test_safetensors_torch_from_torch(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only", revision="refs/pr/1")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True)
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.get_parameters(), new_model.get_parameters()):
            self.assertTrue(mint.equal(p1, p2))

    @require_safetensors
    def test_safetensors_torch_from_torch_sharded(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-bert-pt-only", revision="refs/pr/1")

        with tempfile.TemporaryDirectory() as tmp_dir:
            model.save_pretrained(tmp_dir, safe_serialization=True, max_shard_size="100kB")
            new_model = BertModel.from_pretrained(tmp_dir)

        for p1, p2 in zip(model.get_parameters(), new_model.get_parameters()):
            self.assertTrue(mint.equal(p1, p2))

    def test_load_model_with_state_dict_only(self):
        model = BertModel.from_pretrained("hf-internal-testing/tiny-random-bert")
        state_dict = model.state_dict()
        config = model.config

        model_loaded = BertModel.from_pretrained(
            pretrained_model_name_or_path=None, config=config, state_dict=state_dict
        )
        self.assertTrue(check_models_equal(model, model_loaded))

    @parameterized.expand([("Qwen/Qwen2.5-3B-Instruct", 10), ("meta-llama/Llama-2-7b-chat-hf", 10)])
    @slow
    @require_read_token
    def test_loading_is_fast_on_gpu(self, model_id: str, max_loading_time: float):
        """
        This test is used to avoid regression on https://github.com/huggingface/transformers/pull/36380.
        10s should be more than enough for both models, and allows for some margin as loading time are quite
        unstable. Before #36380, it used to take more than 40s, so 10s is still reasonable.
        Note that we run this test in a subprocess, to ensure that cuda is not already initialized/warmed-up.
        """
        # First download the weights if not already on disk
        _ = AutoModelForCausalLM.from_pretrained(model_id, mindspore_dtype=ms.float16)

        script_to_run = textwrap.dedent(
            """
            import torch
            import time
            import argparse
            from transformers import AutoModelForCausalLM

            parser = argparse.ArgumentParser()
            parser.add_argument("model_id", type=str)
            parser.add_argument("max_loading_time", type=float)
            args = parser.parse_args()

            device = torch.device("cuda:0")

            torch.cuda.synchronize(device)
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(args.model_id, mindspore_dtype=torch.float16, device_map=device)
            torch.cuda.synchronize(device)
            dt = time.time() - t0

            # Assert loading is faster (it should be more than enough in both cases)
            if dt > args.max_loading_time:
                raise ValueError(f"Loading took {dt:.2f}s! It should not take more than {args.max_loading_time}s")
            # Ensure everything is correctly loaded on gpu
            bad_device_params = {k for k, v in model.get_parameters() if v.device != device}
            if len(bad_device_params) > 0:
                raise ValueError(f"The following parameters are not on GPU: {bad_device_params}")
            """
        )

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".py") as tmp:
            tmp.write(script_to_run)
            tmp.flush()
            tmp.seek(0)
            cmd = f"python {tmp.name} {model_id} {max_loading_time}".split()
            try:
                # We cannot use a timeout of `max_loading_time` as cuda initialization can take up to 15-20s
                _ = subprocess.run(cmd, capture_output=True, env=self.get_env(), text=True, check=True, timeout=60)
            except subprocess.CalledProcessError as e:
                raise Exception(f"The following error was captured: {e.stderr}")


@slow
@require_mindspore
class ModelOnTheFlyConversionTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.user = "huggingface-hub-ci"
        cls.token = os.getenv("HUGGINGFACE_PRODUCTION_USER_TOKEN", None)

        if cls.token is None:
            raise ValueError("Cannot run tests as secret isn't setup.")

        cls.api = HfApi(token=cls.token)

    def setUp(self) -> None:
        self.repo_name = f"{self.user}/test-model-on-the-fly-{uuid.uuid4()}"

    def tearDown(self) -> None:
        self.api.delete_repo(self.repo_name)

    def test_safetensors_on_the_fly_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.get_parameters(), converted_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_conversion_private(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, private=True)
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.get_parameters(), converted_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name, token=self.token)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_conversion_gated(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)
        headers = {"Authorization": f"Bearer {self.token}"}
        requests.put(
            f"https://huggingface.co/api/models/{self.repo_name}/settings", json={"gated": "auto"}, headers=headers
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.get_parameters(), converted_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, max_shard_size="200kb")
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.get_parameters(), converted_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion_private(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(
            self.repo_name, token=self.token, safe_serialization=False, max_shard_size="200kb", private=True
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.get_parameters(), converted_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_sharded_conversion_gated(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, max_shard_size="200kb", safe_serialization=False)
        headers = {"Authorization": f"Bearer {self.token}"}
        requests.put(
            f"https://huggingface.co/api/models/{self.repo_name}/settings", json={"gated": "auto"}, headers=headers
        )
        converted_model = BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("Initial and converted models are equal"):
            for p1, p2 in zip(initial_model.get_parameters(), converted_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, "SFconvertbot")
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

    @unittest.skip(reason="Edge case, should work once the Space is updated`")
    def test_safetensors_on_the_fly_wrong_user_opened_pr(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, private=True)
        BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        # This should have opened a PR with the user's account
        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)
            discussion = next(discussions)
            self.assertEqual(discussion.author, self.user)
            self.assertEqual(discussion.title, "Adding `safetensors` variant of this model")

        # We now switch the repo visibility to public
        self.api.update_repo_settings(self.repo_name, private=False)

        # We once again call from_pretrained, which should call the bot to open a PR
        BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token)

        with self.subTest("PR was open with the safetensors account"):
            discussions = self.api.get_repo_discussions(self.repo_name)

            bot_opened_pr = None
            bot_opened_pr_title = None

            for discussion in discussions:
                if discussion.author == "SFconvertbot":
                    bot_opened_pr = True
                    bot_opened_pr_title = discussion.title

            self.assertTrue(bot_opened_pr)
            self.assertEqual(bot_opened_pr_title, "Adding `safetensors` variant of this model")

    def test_safetensors_on_the_fly_specific_revision(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # Push a model on a given revision
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False, revision="new-branch")

        # Try to convert the model on that revision should raise
        with self.assertRaises(EnvironmentError):
            BertModel.from_pretrained(self.repo_name, use_safetensors=True, token=self.token, revision="new-branch")

    def test_absence_of_safetensors_triggers_conversion(self):
        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # Download the model that doesn't have safetensors
        BertModel.from_pretrained(self.repo_name, token=self.token)

        for thread in threading.enumerate():
            if thread.name == "Thread-autoconversion":
                thread.join(timeout=10)

        discussions = self.api.get_repo_discussions(self.repo_name)

        bot_opened_pr = None
        bot_opened_pr_title = None

        for discussion in discussions:
            if discussion.author == "SFconvertbot":
                bot_opened_pr = True
                bot_opened_pr_title = discussion.title

        self.assertTrue(bot_opened_pr)
        self.assertEqual(bot_opened_pr_title, "Adding `safetensors` variant of this model")

    @mock.patch("transformers.safetensors_conversion.spawn_conversion")
    def test_absence_of_safetensors_triggers_conversion_failed(self, spawn_conversion_mock):
        spawn_conversion_mock.side_effect = HTTPError()

        config = BertConfig(
            vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
        )
        initial_model = BertModel(config)

        # Push a model on `main`
        initial_model.push_to_hub(self.repo_name, token=self.token, safe_serialization=False)

        # The auto conversion is mocked to always raise; ensure that it doesn't raise in the main thread
        BertModel.from_pretrained(self.repo_name, token=self.token)


@require_mindspore
@is_staging_test
class ModelPushToHubTester(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._token = TOKEN
        HfFolder.save_token(TOKEN)

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            model.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.get_parameters(), new_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub_via_save_pretrained(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir, repo_id=tmp_repo.repo_id, push_to_hub=True, token=self._token)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.get_parameters(), new_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

    def test_push_to_hub_with_description(self):
        with TemporaryHubRepo(token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            COMMIT_DESCRIPTION = """
The commit description supports markdown synthax see:
```python
>>> form transformers import AutoConfig
>>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")
```
"""
            commit_details = model.push_to_hub(
                tmp_repo.repo_id, use_auth_token=self._token, create_pr=True, commit_description=COMMIT_DESCRIPTION
            )
            self.assertEqual(commit_details.commit_description, COMMIT_DESCRIPTION)

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub_in_organization(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            model.push_to_hub(tmp_repo.repo_id, token=self._token)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.get_parameters(), new_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))

    @unittest.skip(reason="This test is flaky")
    def test_push_to_hub_in_organization_via_save_pretrained(self):
        with TemporaryHubRepo(namespace="valid_org", token=self._token) as tmp_repo:
            config = BertConfig(
                vocab_size=99, hidden_size=32, num_hidden_layers=5, num_attention_heads=4, intermediate_size=37
            )
            model = BertModel(config)
            # Push to hub via save_pretrained
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir, push_to_hub=True, token=self._token, repo_id=tmp_repo.repo_id)

            new_model = BertModel.from_pretrained(tmp_repo.repo_id)
            for p1, p2 in zip(model.get_parameters(), new_model.get_parameters()):
                self.assertTrue(mint.equal(p1, p2))


@require_mindspore
class AttentionMaskTester(unittest.TestCase):
    def check_non_causal(self, bsz, q_len, kv_len, mask_2d, mask_4d):
        mask_indices = (mask_2d != 1)[:, None].broadcast_to((bsz, q_len, kv_len))
        mask_4d_values = mask_4d[:, 0][mask_indices]
        is_inf = mask_4d_values == -float("inf")
        is_min = mask_4d_values == dtype_to_min(mask_4d.dtype)
        assert mint.logical_or(is_inf, is_min).all()

    def check_to_4d(self, mask_converter, q_len, kv_len, additional_mask=None, bsz=3):
        mask_2d = mint.ones((bsz, kv_len), dtype=ms.int64)

        if additional_mask is not None:
            for bsz_idx, seq_idx in additional_mask:
                mask_2d[bsz_idx, seq_idx] = 0

        mask_4d = mask_converter.to_4d(mask_2d, query_length=q_len, key_value_length=kv_len, dtype=ms.float32)

        assert mask_4d.shape == (bsz, 1, q_len, kv_len)

        # make sure there are no overflows
        assert mask_4d.min() != float("-inf")

        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)

            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().item() == num_tokens_masked
            if 0 in mask_2d:
                # at least causal mask + maybe more
                assert (mask_4d != 0).sum().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif not mask_converter.is_causal and context is None:
            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().item() == 0
            if 0 in mask_2d:
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)
        elif mask_converter.is_causal and context is not None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = (q_len * (q_len - 1) // 2) + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked

            if 0 not in mask_2d:
                assert (mask_4d != 0).sum().item() == num_tokens_masked
            if 0 in mask_2d:
                # at least causal mask + maybe more
                assert (mask_4d != 0).sum().item() >= num_tokens_masked
                self.check_non_causal(bsz, q_len, kv_len, mask_2d, mask_4d)

    def check_to_causal(self, mask_converter, q_len, kv_len, bsz=3):
        mask_4d = mask_converter.to_causal_4d(bsz, query_length=q_len, key_value_length=kv_len, dtype=ms.float32)

        if q_len == 1 and mask_converter.sliding_window is None:
            # no causal mask if q_len is 1
            assert mask_4d is None
            return

        context = mask_converter.sliding_window
        if mask_converter.is_causal and context is None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = bsz * (q_len * (q_len - 1) // 2)

            assert (mask_4d != 0).sum().item() == num_tokens_masked
        elif not mask_converter.is_causal and context is None:
            assert (mask_4d != 0).sum().item() == 0
        elif mask_converter.is_causal and context is not None:
            # k * (k+1) / 2 tokens are masked in triangualar masks
            num_tokens_masked = (q_len * (q_len - 1) // 2) + self.compute_num_context_mask(kv_len, context, q_len)
            num_tokens_masked = bsz * num_tokens_masked

            assert (mask_4d != 0).sum().item() == num_tokens_masked

    def compute_num_context_mask(self, kv_len, context, q_len):
        # This function computes the # of attention tokens that are added for
        # the sliding window
        c_mask_len = kv_len - context - 1
        num_mask_triangle = c_mask_len * (c_mask_len + 1) // 2
        cut_mask_len = max(c_mask_len - q_len, 0)
        num_cut_mask = cut_mask_len * (cut_mask_len + 1) // 2
        return num_mask_triangle - num_cut_mask

    def test_2d_to_4d(self):
        mask_converter = AttentionMaskConverter(is_causal=False)

        # non auto-regressive case
        self.check_to_4d(mask_converter, q_len=7, kv_len=7)

        # same with extra attention masks
        self.check_to_4d(mask_converter, q_len=7, kv_len=7, additional_mask=[(0, 2), (1, 3), (2, 0)])

    def test_causal_mask(self):
        mask_converter = AttentionMaskConverter(is_causal=True)

        # auto-regressive use case
        self.check_to_causal(mask_converter, q_len=1, kv_len=7)
        # special auto-regressive case
        self.check_to_causal(mask_converter, q_len=3, kv_len=7)
        # non auto-regressive case
        self.check_to_causal(mask_converter, q_len=7, kv_len=7)

    @require_mindspore
    @slow
    def test_unmask_unattended_left_padding(self):
        attention_mask = ms.tensor([[0, 0, 1], [1, 1, 1], [0, 1, 1]]).to(ms.int64)

        expanded_mask = ms.tensor(
            [
                [[[0, 0, 0], [0, 0, 0], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[0, 0, 0], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(ms.int64)

        reference_output = ms.tensor(
            [
                [[[1, 1, 1], [1, 1, 1], [0, 0, 1]]],
                [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]],
                [[[1, 1, 1], [0, 1, 0], [0, 1, 1]]],
            ]
        ).to(ms.int64)

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=1)

        self.assertTrue(mint.equal(result, reference_output))

        attention_mask = ms.tensor([[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]]).to(ms.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=ms.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)
        min_inf = dtype_to_min(ms.float32)
        reference_output = ms.tensor(
            [
                [
                    [
                        [0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0],
                        [min_inf, min_inf, 0, min_inf, min_inf],
                        [min_inf, min_inf, 0, 0, min_inf],
                        [min_inf, min_inf, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, min_inf, min_inf, min_inf, min_inf],
                        [0, 0, min_inf, min_inf, min_inf],
                        [0, 0, 0, min_inf, min_inf],
                        [0, 0, 0, 0, min_inf],
                        [0, 0, 0, 0, 0],
                    ]
                ],
                [
                    [
                        [0, 0, 0, 0, 0],
                        [min_inf, 0, min_inf, min_inf, min_inf],
                        [min_inf, 0, 0, min_inf, min_inf],
                        [min_inf, 0, 0, 0, min_inf],
                        [min_inf, 0, 0, 0, 0],
                    ]
                ],
            ]
        )

        self.assertTrue(mint.equal(reference_output, result))

    @require_mindspore
    @slow
    def test_unmask_unattended_right_padding(self):
        attention_mask = ms.tensor([[1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]]).to(ms.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=ms.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(mint.equal(expanded_mask, result))

    @require_mindspore
    @slow
    def test_unmask_unattended_random_mask(self):
        attention_mask = ms.tensor([[1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]]).to(ms.int64)

        attn_mask_converter = AttentionMaskConverter(is_causal=True)
        past_key_values_length = 0
        key_value_length = attention_mask.shape[-1] + past_key_values_length

        expanded_mask = attn_mask_converter.to_4d(
            attention_mask, attention_mask.shape[-1], key_value_length=key_value_length, dtype=ms.float32
        )

        result = AttentionMaskConverter._unmask_unattended(expanded_mask, attention_mask, unmasked_value=0)

        self.assertTrue(mint.equal(expanded_mask, result))
