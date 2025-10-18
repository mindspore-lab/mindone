# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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

import unittest

from mindone.transformers.testing_utils import is_mindspore_available, require_mindspore

if is_mindspore_available():
    from transformers import LlamaConfig

    from mindspore import mint

    from mindone.transformers import GPT2LMHeadModel
    from mindone.transformers.cache_utils import DynamicCache, StaticCache


@require_mindspore
class CacheTest(unittest.TestCase):
    def test_dynamic_cache_retrocompatibility(self):
        """Tests that we can convert back and forth between the legacy cache format and DynamicCache"""
        legacy_cache = ()
        new_cache = DynamicCache()

        # Creates a new cache with 10 layers in both formats
        for layer_idx in range(10):
            new_key = mint.rand((2, 4, 8, 16))
            new_value = mint.rand((2, 4, 8, 16))
            new_cache.update(new_key, new_value, layer_idx)
            legacy_cache += ((new_key, new_value),)

        # Sanity check 1: they must have the same shapes
        self.assertTrue(len(legacy_cache), len(new_cache))
        for layer_idx in range(10):
            self.assertTrue(len(legacy_cache[layer_idx]), len(legacy_cache[layer_idx]))
            for key_value_idx in range(2):
                self.assertTrue(
                    legacy_cache[layer_idx][key_value_idx].shape == new_cache[layer_idx][key_value_idx].shape
                )

        # Sanity check 2: we can get the sequence length in multiple ways with DynamicCache, and they return the
        # expected value
        self.assertTrue(legacy_cache[0][0].shape[-2] == new_cache[0][0].shape[-2] == new_cache.get_seq_length() == 8)

        # Sanity check 3: they must be equal, and both support indexing
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    mint.allclose(new_cache[layer_idx][key_value_idx], legacy_cache[layer_idx][key_value_idx])
                )

        # Test 1: We can convert from legacy to new with no changes
        from_legacy = DynamicCache.from_legacy_cache(legacy_cache)
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    mint.allclose(from_legacy[layer_idx][key_value_idx], legacy_cache[layer_idx][key_value_idx])
                )

        # Test 2: We can convert from new to legacy with no changes
        to_legacy = new_cache.to_legacy_cache()
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(mint.allclose(to_legacy[layer_idx][key_value_idx], new_cache[layer_idx][key_value_idx]))

    def test_reorder_cache_retrocompatibility(self):
        """Tests that Cache.reorder_cache is retrocompatible with the legacy code path"""
        legacy_reorder_fn = GPT2LMHeadModel._reorder_cache  # An example of a legacy `_reorder_cache` function

        legacy_cache = ()
        new_cache = DynamicCache()

        # Creates a new cache with 10 layers in both formats
        for layer_idx in range(10):
            new_key = mint.rand((4, 4, 8, 16))
            new_value = mint.rand((4, 4, 8, 16))
            new_cache.update(new_key, new_value, layer_idx)
            legacy_cache += ((new_key, new_value),)

        # Let's create some dummy beam indices. From the shape above, it is equivalent to the case where num_beams=4
        # and batch_size=1
        beam_idx = mint.randint(0, 4, (4,))

        legacy_cache_reordered = legacy_reorder_fn(legacy_cache, beam_idx)
        new_cache.reorder_cache(beam_idx)

        # Let's check that the results are the same
        for layer_idx in range(10):
            for key_value_idx in range(2):
                self.assertTrue(
                    mint.allclose(new_cache[layer_idx][key_value_idx], legacy_cache_reordered[layer_idx][key_value_idx])
                )

    def test_static_cache_mha_mqa_gqa(self):
        """
        Tests that static cache works with multi-head attention (MHA), grouped query attention (GQA), and multi-query
        attention (MQA)
        """

        def _random_kvs(config):
            # shape for key and values: (batch_size, num_heads, seq_len, head_dim)
            random_keys = mint.rand(
                (1, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads)
            )
            random_values = mint.rand(
                (1, config.num_key_value_heads, 1, config.hidden_size // config.num_attention_heads),
            )
            return random_keys, random_values

        mha_config = LlamaConfig(num_attention_heads=32)
        mha_static_cache = StaticCache(config=mha_config, batch_size=1, max_cache_len=10)
        cached_keys, cached_values = mha_static_cache.update(
            *_random_kvs(mha_config), 0, cache_kwargs={"cache_position": mint.arange(1)}
        )
        self.assertTrue(cached_keys.shape == (1, 32, 10, 128))
        self.assertTrue(cached_values.shape == (1, 32, 10, 128))

        gqa_config = LlamaConfig(num_attention_heads=32, num_key_value_heads=4)
        gqa_static_cache = StaticCache(config=gqa_config, batch_size=1, max_cache_len=10)
        cached_keys, cached_values = gqa_static_cache.update(
            *_random_kvs(gqa_config), 0, cache_kwargs={"cache_position": mint.arange(1)}
        )
        self.assertTrue(cached_keys.shape == (1, 4, 10, 128))
        self.assertTrue(cached_values.shape == (1, 4, 10, 128))

        mqa_config = LlamaConfig(num_attention_heads=32, num_key_value_heads=1)
        mqa_static_cache = StaticCache(config=mqa_config, batch_size=1, max_cache_len=10)
        cached_keys, cached_values = mqa_static_cache.update(
            *_random_kvs(mqa_config), 0, cache_kwargs={"cache_position": mint.arange(1)}
        )
        self.assertTrue(cached_keys.shape == (1, 1, 10, 128))
        self.assertTrue(cached_values.shape == (1, 1, 10, 128))
