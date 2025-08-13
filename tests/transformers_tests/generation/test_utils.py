# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run pytest on mindspore
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a clone of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import gc
import inspect
import random
import tempfile
import unittest

import numpy as np
import pytest
from parameterized import parameterized
from transformers import AutoTokenizer

from mindone.transformers.testing_utils import (
    is_flaky,
    require_mindspore,
    set_config_for_less_flaky_test,
    set_model_for_less_flaky_test,
    set_model_tester_for_less_flaky_test,
    slow,
)
from mindone.transformers.utils import is_mindspore_available

if is_mindspore_available():
    from transformers.generation import PhrasalConstraint

    import mindspore as ms
    from mindspore import mint, ops

    from mindone.transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
    from mindone.transformers.cache_utils import Cache, EncoderDecoderCache, HybridCache, StaticCache
    from mindone.transformers.generation import (
        BeamSampleDecoderOnlyOutput,
        BeamSampleEncoderDecoderOutput,
        BeamSearchDecoderOnlyOutput,
        BeamSearchEncoderDecoderOutput,
        GenerateBeamDecoderOnlyOutput,
        GenerateBeamEncoderDecoderOutput,
        GenerateDecoderOnlyOutput,
        GenerateEncoderDecoderOutput,
        GenerationConfig,
        GreedySearchDecoderOnlyOutput,
        GreedySearchEncoderDecoderOutput,
        PromptLookupCandidateGenerator,
        SampleDecoderOnlyOutput,
        SampleEncoderDecoderOutput,
    )
    from mindone.transformers.generation.candidate_generator import AssistedCandidateGeneratorDifferentTokenizers
    from mindone.transformers.generation.utils import _speculative_sampling


# TODO: raushan remove this when VLMs start accepting input embeds
VLM_CLASS_NAMES = [
    "llava",
    "idefics2",
    "idefics3",
    "mllama",
    "paligemma",
    "emu3",
    "qwen2vl",
    "qwen2_5_vl",
    "ayavision",
    "gemma3",
    "mistral3",
    "chameleon",
]


class GenerationTesterMixin:
    input_name = "input_ids"
    model_tester = None
    max_new_tokens = 3

    def prepare_config_and_inputs_for_generate(self, batch_size=2):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()

        # We don't want a few model inputs in our model input dictionary for generation tests
        input_keys_to_ignore = [
            # we don't want to mask attention heads
            "head_mask",
            "decoder_head_mask",
            "cross_attn_head_mask",
            # we don't want encoder-decoder models to start from filled decoder ids
            "decoder_input_ids",
            "decoder_attention_mask",
            # we'll set cache use in each test differently
            "use_cache",
            # Ignore labels if it is in the input dict
            "labels",
            # model-specific exceptions should overload/overwrite this function
        ]
        filtered_inputs_dict = {
            k: v[:batch_size, ...] if isinstance(v, ms.tensor) else v
            for k, v in inputs_dict.items()
            if k not in input_keys_to_ignore
        }

        # It is important set `eos_token_id` to `None` to avoid early stopping (would break for length-based checks)
        text_gen_config = config.get_text_config(decoder=True)
        if text_gen_config.eos_token_id is not None and text_gen_config.pad_token_id is None:
            text_gen_config.pad_token_id = (
                text_gen_config.eos_token_id
                if isinstance(text_gen_config.eos_token_id, int)
                else text_gen_config.eos_token_id[0]
            )
        text_gen_config.eos_token_id = None
        text_gen_config.forced_eos_token_id = None

        return config, filtered_inputs_dict

    def _check_similar_generate_outputs(self, output_1, output_2, atol=1e-5, rtol=1e-5):
        """
        Checks whether a pair of generate outputs are similar. Two `generate` call outputs are considered similar in
        the following situations:
        1. The sequences are the same
        2. The sequences are different, but the scores up to (and including) the first mismatch are nearly identical
        """
        # scores doesn't include data regarding decoder input tokens
        decoder_input_length = output_1.sequences.shape[1] - len(output_1.scores)
        output_matches = output_1.sequences == output_2.sequences
        has_matching_outputs = output_matches.all()
        has_matching_scores = None
        if not has_matching_outputs:
            for batch_idx in range(output_1.sequences.shape[0]):
                batch_matches = output_matches[batch_idx]
                if batch_matches.all():
                    continue
                first_mismatch_idx = batch_matches.int().argmin()  # gets the index of the first False
                first_mismatch_idx -= decoder_input_length
                output_1_first_mismatch_scores = output_1.scores[first_mismatch_idx][batch_idx]
                output_2_first_mismatch_scores = output_2.scores[first_mismatch_idx][batch_idx]
                has_matching_scores = mint.allclose(
                    output_1_first_mismatch_scores, output_2_first_mismatch_scores, rtol=atol, atol=rtol
                )
                if not has_matching_scores:
                    break
        self.assertTrue(has_matching_outputs or has_matching_scores)

    def _get_logits_processor_kwargs(self, do_sample=False, config=None):
        logits_processor_kwargs = {
            "bad_words_ids": [[1, 0]],
            "repetition_penalty": 1.2,
            "remove_invalid_values": True,
        }
        if do_sample:
            logits_processor_kwargs.update(
                {
                    "top_k": 10,
                    "top_p": 0.7,
                    "temperature": 0.7,
                }
            )
        # TODO (joao, raushan): see this comment for a long-term fix
        # https://github.com/huggingface/transformers/pull/33593#issuecomment-2361824264)
        # This is a band-aid for VLM models, to ensure they don't generate image/video tokens which would cause them
        # to crash. On pretrained models this isn't a risk, as they are trained to not generate these tokens.
        if config is not None:
            for key in [
                "image_token_index",
                "image_token_id",
                "video_token_index",
                "video_token_id",
                "vision_start_token_id",
            ]:
                token_index = getattr(config, key, None)
                if token_index is None and hasattr(self, "model_tester"):
                    token_index = getattr(self.model_tester, key, None)
                if token_index is not None and token_index < config.get_text_config().vocab_size:
                    logits_processor_kwargs["bad_words_ids"].append([token_index])

        return logits_processor_kwargs

    def _get_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    def _get_diverse_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": 2,
            "num_return_sequences": num_return_sequences,
            "num_beam_groups": 2,  # one beam per group
            "diversity_penalty": 2.0,
        }
        return beam_kwargs

    def _get_constrained_beam_kwargs(self, num_return_sequences=1):
        beam_kwargs = {
            "early_stopping": False,
            "length_penalty": 2.0,
            "num_beams": num_return_sequences * 4,
            "num_return_sequences": num_return_sequences,
        }
        return beam_kwargs

    def _greedy_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _sample_generate(
        self,
        model,
        inputs_dict,
        num_return_sequences,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        # fixme set seed in mindspore and get the same performance as torch
        # torch.manual_seed(0)
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        output_generate = model.generate(
            do_sample=True,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            num_return_sequences=num_return_sequences,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _beam_sample_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        # fixme set seed in mindspore and get the same performance as torch
        # torch.manual_seed(0)
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)
        output_generate = model.generate(
            do_sample=True,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _group_beam_search_generate(
        self,
        model,
        inputs_dict,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _constrained_beam_search_generate(
        self,
        model,
        inputs_dict,
        constraints,
        beam_kwargs,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            output_scores=output_scores,
            output_logits=output_logits,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict_in_generate=return_dict_in_generate,
            constraints=constraints,
            use_cache=use_cache,
            **beam_kwargs,
            **logits_processor_kwargs,
            **inputs_dict,
        )

        return output_generate

    def _contrastive_generate(
        self,
        model,
        inputs_dict,
        output_scores=False,
        output_logits=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict_in_generate=False,
        use_cache=True,
    ):
        contrastive_search_kwargs = {
            "penalty_alpha": 0.6,
            "top_k": 5,
        }

        logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=False, config=model.config)
        output_generate = model.generate(
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
            output_logits=output_logits,
            return_dict_in_generate=return_dict_in_generate,
            use_cache=use_cache,
            **logits_processor_kwargs,
            **contrastive_search_kwargs,
            **inputs_dict,
        )

        return output_generate

    @pytest.mark.generate
    def test_greedy_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).set_train(False)
            output_generate = self._greedy_generate(model=model, inputs_dict=inputs_dict)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_greedy_generate_dict_outputs(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).set_train(False)
            output_generate = self._greedy_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, GreedySearchEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, GreedySearchDecoderOnlyOutput)

            self._check_generate_outputs(output_generate, model.config)

    @pytest.mark.generate
    def test_greedy_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            if any(model_name in model_class.__name__.lower() for model_name in ["rwkv"]):
                self.skipTest(reason="Won't fix: model with non-standard dictionary output shapes")

            config.is_decoder = True
            model = model_class(config).set_train(False)
            output_generate = self._greedy_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=True,  # Enable cache
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )

            self._check_generate_outputs(output_generate, model.config, use_cache=True)

    @pytest.mark.generate
    def test_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).set_train(False)
            output_generate = self._sample_generate(model=model, inputs_dict=inputs_dict, num_return_sequences=1)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).set_train(False)
            output_generate = self._sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                num_return_sequences=2,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, SampleEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, SampleDecoderOnlyOutput)

            self._check_generate_outputs(output_generate, model.config, num_return_sequences=2)

    @pytest.mark.generate
    def test_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).set_train(False)

            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_search_generate(model=model, inputs_dict=inputs_dict, beam_kwargs=beam_kwargs)

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).set_train(False)
            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_generate_outputs(
                output_generate,
                model.config,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    @pytest.mark.generate
    def test_beam_search_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            if any(model_name in model_class.__name__.lower() for model_name in ["rwkv"]):
                self.skipTest(reason="Won't fix: model with non-standard dictionary output shapes")

            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise
            model = model_class(config).set_train(False)
            beam_kwargs = self._get_beam_kwargs()

            config.is_decoder = True
            model = model_class(config).set_train(False)
            output_generate = self._beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=True,  # Enable cache
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )

            self._check_generate_outputs(
                output_generate,
                model.config,
                use_cache=True,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    # using zero3 parallel
    @pytest.mark.generate
    def test_model_parallel_beam_search(self):
        for model_class in self.all_generative_model_classes:
            if model_class._no_split_modules is None:
                continue

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).eval()
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)
                new_model = model_class.from_pretrained(tmp_dir)

                new_model.generate(
                    max_new_tokens=self.max_new_tokens,
                    num_beams=2,
                    **inputs_dict,
                )

    @pytest.mark.generate
    def test_beam_sample_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).set_train(False)
            beam_kwargs = self._get_beam_kwargs()
            output_generate = self._beam_sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_beam_sample_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).set_train(False)
            beam_kwargs = self._get_beam_kwargs()

            output_generate = self._beam_sample_generate(
                model=model,
                inputs_dict=inputs_dict,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSampleEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSampleDecoderOnlyOutput)

            self._check_generate_outputs(
                output_generate,
                model.config,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    @pytest.mark.generate
    def test_generate_without_input_ids(self):
        config, _ = self.prepare_config_and_inputs_for_generate()

        # if no bos token id => cannot generate from None
        if config.bos_token_id is None:
            self.skipTest(reason="bos_token_id is None")

        # hack in case they are equal, otherwise the attn mask will be [0]
        if config.bos_token_id == config.pad_token_id:
            config.pad_token_id = None

        for model_class in self.all_generative_model_classes:
            model = model_class(config)
            model.set_train(False)

            output_ids_generate = model.generate(
                do_sample=False, max_new_tokens=self.max_new_tokens, remove_invalid_values=True
            )
            self.assertIsNotNone(output_ids_generate)

    @is_flaky()
    @pytest.mark.generate
    def test_constrained_beam_search_generate(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            model = model_class(config).set_train(False)

            # Sample constraints
            min_id = 3
            max_id = config.get_text_config(decoder=True).vocab_size

            force_tokens = mint.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs()
            output_generate = self._constrained_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

            # check`constrained_beam_search` for higher than 1 `num_return_sequences`
            # Sample constraints
            force_tokens = mint.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs(num_return_sequences=2)

            output_generate = self._constrained_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

            for generation_output in output_generate:
                self._check_sequence_inside_sequence(force_tokens, generation_output)

    @pytest.mark.generate
    def test_constrained_beam_search_generate_dict_output(self):
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).set_train(False)

            # Sample constraints
            min_id = 3
            max_id = model.config.get_text_config(decoder=True).vocab_size
            force_tokens = mint.randint(min_id, max_id, (1, 2)).tolist()[0]
            constraints = [
                PhrasalConstraint(force_tokens),
            ]

            beam_kwargs = self._get_constrained_beam_kwargs()
            output_generate = self._constrained_beam_search_generate(
                model=model,
                inputs_dict=inputs_dict,
                constraints=constraints,
                beam_kwargs=beam_kwargs,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=False,
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
                self.assertIsInstance(output_generate, GenerateBeamEncoderDecoderOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchEncoderDecoderOutput)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )
                self.assertIsInstance(output_generate, GenerateBeamDecoderOnlyOutput)
                # Retrocompatibility check
                self.assertIsInstance(output_generate, BeamSearchDecoderOnlyOutput)

            self._check_generate_outputs(
                output_generate,
                model.config,
                num_return_sequences=beam_kwargs["num_return_sequences"],
                num_beams=beam_kwargs["num_beams"],
            )

    @pytest.mark.generate
    def test_contrastive_generate(self):
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support contrastive search generation")

            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            config.is_decoder = True

            # test old generation output for backwards compatibility
            model = model_class(config).set_train(False)
            output_generate = self._contrastive_generate(
                model=model,
                inputs_dict=inputs_dict,
                use_cache=True,  # Enable cache
            )
            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(output_generate.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1])

    @pytest.mark.generate
    def test_contrastive_generate_dict_outputs_use_cache(self):
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support contrastive search generation")

            # won't fix: FSMT and Reformer have a different cache variable type (and format).
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")
            config.is_decoder = True
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            model = model_class(config).set_train(False)
            output_generate = self._contrastive_generate(
                model=model,
                inputs_dict=inputs_dict,
                output_scores=True,
                output_logits=True,
                output_hidden_states=True,
                output_attentions=self.has_attentions,
                return_dict_in_generate=True,
                use_cache=True,  # Enable cache
            )

            if model.config.is_encoder_decoder:
                self.assertTrue(output_generate.sequences.shape[-1] == self.max_new_tokens + 1)
            else:
                self.assertTrue(
                    output_generate.sequences.shape[-1] == self.max_new_tokens + inputs_dict["input_ids"].shape[-1]
                )

            self._check_generate_outputs(output_generate, model.config, use_cache=True)

    @pytest.mark.generate
    def test_contrastive_generate_low_memory(self):
        # Check that choosing 'low_memory' does not change the model output
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support contrastive search generation")

            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer", "speech2text"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(model_name in model_class.__name__.lower() for model_name in ["gptbigcode"]):
                self.skipTest(reason="TODO: fix me")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: contrastive search only works with cache on at the moment.
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True

            # test output equality of low versus high memory
            model = model_class(config).set_train(False)

            low_output = model.generate(
                top_k=4,
                penalty_alpha=0.6,
                low_memory=True,
                max_new_tokens=self.max_new_tokens,
                **inputs_dict,
                use_cache=True,
            )

            high_output = model.generate(
                top_k=4,
                penalty_alpha=0.6,
                low_memory=False,
                max_new_tokens=self.max_new_tokens,
                **inputs_dict,
                use_cache=True,
            )
            self.assertListEqual(low_output.tolist(), high_output.tolist())

    @parameterized.expand([("random",), ("same",)])
    @pytest.mark.generate
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        # This test ensures that the assisted generation does not introduce output changes over greedy search.
        # See https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535 for more info.
        # NOTE: It breaks the pattern in the tests above, for multiple reasons:
        # - assisted_decoding, contrarily to the other methods, can't be called on its own (e.g. needs to
        # prepare the assistant encoder outputs in the main generate body);
        # - assisted_decoding does not support `use_cache = False`
        # - assisted_decoding does not support `batch_size > 1`

        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support assisted generation")
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "moshi",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                    "mllama",  # special cache sizes
                    "blip2",  # overridden `generate()`
                    "instructblip",
                    "instructblipvideo",
                ]
            ):
                self.skipTest(reason="May fix in the future: need model-specific fixes")

            # enable cache
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True
            model = model_class(config).set_train(False)
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the assistant model always generates two tokens when it is called, to ensure the input preparation of
            #    the assistant model is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": False,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": True,
            }
            logits_processor_kwargs = self._get_logits_processor_kwargs(config=model.config)

            output_greedy = model.generate(**generation_kwargs, **inputs_dict, **logits_processor_kwargs)

            # test with the same assistant model or randomly init one
            # in the first case all candidate tokens are accepted, in the second none is accepted
            # case when some are accepted and some not is hard to reproduce, so let's hope this catches most errors :)
            if assistant_type == "random":
                assistant_model = model_class(config).set_train(False)
            else:
                assistant_model = model
            assistant_model.generation_config.num_assistant_tokens = 2  # see b)
            assistant_model.generation_config.num_assistant_tokens_schedule = "constant"  # see b)
            generation_kwargs.update({"assistant_model": assistant_model})
            output_assisted = model.generate(**generation_kwargs, **inputs_dict, **logits_processor_kwargs)

            # The two outputs must match and their shape must be as expected
            self._check_similar_generate_outputs(output_greedy, output_assisted)
            for output in (output_greedy, output_assisted):
                self._check_generate_outputs(output, model.config, use_cache=True)

    @pytest.mark.generate
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        # This test ensures that the prompt lookup generation does not introduce output changes over greedy search.
        # This test is mostly a copy of test_assisted_decoding_matches_greedy_search

        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support assisted generation")
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "moshi",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                    "fuyu",
                    "mllama",  # special cache sizes
                    "blip2",  # overridden `generate()`
                    "instructblip",
                    "instructblipvideo",
                    *VLM_CLASS_NAMES,  # shouldn't suggest image tokens
                ]
            ):
                self.skipTest(reason="May fix in the future: need model-specific fixes")

            # enable cache
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True
            model = model_class(config).set_train(False)
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the prompt lookup tries to give the model 2 tokens, to ensure the input preparation of
            #    prompt lookup is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": False,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": True,
            }

            output_greedy = model.generate(**generation_kwargs, **inputs_dict)

            generation_kwargs.update({"prompt_lookup_num_tokens": 2})  # see b)
            output_prompt_lookup = model.generate(**generation_kwargs, **inputs_dict)

            # The two outputs must match and their shape must be as expected
            self._check_similar_generate_outputs(output_greedy, output_prompt_lookup)
            for output in (output_greedy, output_prompt_lookup):
                self._check_generate_outputs(output, model.config, use_cache=True)

    @pytest.mark.generate
    def test_dola_decoding_sample(self):
        # TODO (joao): investigate skips, try to reduce incompatibilities
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support DoLa decoding")

            if any(model_name in model_class.__name__.lower() for model_name in ["reformer"]):
                self.skipTest("Skip Reformer as the lm_head input size is 2 * hidden size, adopted from Rev Nets.")

            if any(model_name in model_class.__name__.lower() for model_name in ["marian", "mbart", "pegasus"]):
                self.skipTest("DoLa is not supported for models that don't return layerwise hidden states")

            if any(model_name == model_class.__name__ for model_name in ["LlavaNextVideoForConditionalGeneration"]):
                self.skipTest(f"DoLa is failing for {model_class.__name__}")

            # enable cache if the model is not openai-gpt, xlnet, cpm, or xlm
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # Encoder-decoder models are not supported
            if config.is_encoder_decoder:
                self.skipTest("DoLa is not supported for encoder-decoder models")
            config.is_decoder = True
            model = model_class(config).set_train(False)

            if model.get_output_embeddings() is None:
                self.skipTest("DoLa is not supported for models that don't have output embeddings")

            logits_processor_kwargs = self._get_logits_processor_kwargs(do_sample=True, config=model.config)

            # Sets dola generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see b)
                "num_beams": 1,
                "do_sample": True,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": getattr(config, "use_cache", False),  # Some models don't support the cache
                "dola_layers": "low",
            }
            output_dola = model.generate(**generation_kwargs, **logits_processor_kwargs, **inputs_dict)
            self._check_generate_outputs(output_dola, model.config, use_cache=getattr(config, "use_cache", False))

    @pytest.mark.generate
    def test_assisted_decoding_sample(self):
        # In this test we don't check assisted vs non-assisted output -- seeded assisted decoding with sample will not
        # match sample for the same seed, as the forward pass does not return the exact same logits (due to matmul with
        # different shapes, see https://github.com/huggingface/transformers/issues/25420#issuecomment-1775317535).
        for model_class in self.all_generative_model_classes:
            if model_class._is_stateful:
                self.skipTest(reason="Stateful models don't support assisted generation")
            if any(model_name in model_class.__name__.lower() for model_name in ["fsmt", "reformer"]):
                self.skipTest(reason="Won't fix: old model with different cache format")
            if any(
                model_name in model_class.__name__.lower()
                for model_name in [
                    "bigbirdpegasus",
                    "led",
                    "mega",
                    "moshi",
                    "speech2text",
                    "git",
                    "prophetnet",
                    "seamlessm4t",
                    "clvp",
                    "mllama",  # special cache sizes
                    "blip2",  # overridden `generate()`
                    "instructblip",
                    "instructblipvideo",
                ]
            ):
                self.skipTest(reason="May fix in the future: need model-specific fixes")

            # enable cache
            config, inputs_dict = self.prepare_config_and_inputs_for_generate(batch_size=1)

            # NOTE: assisted generation only works with cache on at the moment.
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            config.is_decoder = True
            model = model_class(config).set_train(False)
            # Sets assisted generation arguments such that:
            # a) no EOS is generated, to ensure generation doesn't break early
            # b) the assistant model always generates two tokens when it is called, to ensure the input preparation of
            #    the assistant model is correct
            # c) there are at least two forward passes in the main model, to ensure the input preparation of
            #    the main model is correct
            assistant_model = model
            assistant_model.generation_config.num_assistant_tokens = 2  # see b)
            assistant_model.generation_config.num_assistant_tokens_schedule = "constant"  # see b)
            generation_kwargs = {
                "eos_token_id": -1,  # see a)
                "max_new_tokens": 4,  # see c)
                "num_beams": 1,
                "do_sample": True,
                "assistant_model": assistant_model,
                "output_scores": True,
                "output_logits": True,
                "output_hidden_states": True,
                "output_attentions": self.has_attentions,
                "return_dict_in_generate": True,
                "use_cache": True,
            }
            logits_processor_kwargs = self._get_logits_processor_kwargs(config=model.config)
            output_assisted = model.generate(**generation_kwargs, **inputs_dict, **logits_processor_kwargs)

            self._check_generate_outputs(output_assisted, config, use_cache=True)

    @pytest.mark.generate
    def test_prompt_lookup_decoding_stops_at_eos(self):
        # This test ensures that the prompt lookup generation stops at eos token and does not suggest more tokens
        # (see https://github.com/huggingface/transformers/pull/31301)

        # The main idea is to have an ngram (unigram in our case) that is repeated twice in the input ids.
        # First time at the very end, so input ends with the unigrams, and second any arbitrary location.
        # Also, we need an EOS token which will be injected just after the arbitrary located ngram.
        # We verify that PLD will not copy and propose candidated that contain an EOS token, even if there are overlapping ngrams
        # in input ids. Otherwise a proposed EOS along with the trailing (ngrams-1) tokens might be accepted by the target model.
        # That seems as if the model "generated" and EOS but didn't stop from user's perspective

        input_ids = mint.randint(1, 50, (1, 10))  # generate inputs in range from 1-50
        arbitrary_ngram = 51  # this is the arbitrary ngram, specifically chosen OOV to prevent flaky tests
        input_ids[:, 3] = arbitrary_ngram  # set pre-eos to arbitrary_ngram which is for sure not present in inputs
        input_ids[:, -1] = arbitrary_ngram  # put arbitrary_ngram in the end for the necessary match to happen

        eos_token_id = mint.tensor([0])
        input_ids[:, 4] = eos_token_id  # inject eos-token-id in input ids so that it is located after arbitrary_ngram

        # init cand geenerator with max_matching_ngram_size=1 to match per-token
        candidate_generator = PromptLookupCandidateGenerator(
            eos_token_id=eos_token_id, num_output_tokens=4, max_matching_ngram_size=1
        )
        output_prompt_lookup = candidate_generator.get_candidates(input_ids)[0]

        # PLD shouldn't propose any new tokens based on eos-match
        self.assertTrue(output_prompt_lookup.shape[-1] == 10)

    @pytest.mark.generate
    def test_generate_with_head_masking(self):
        """Test designed for encoder-decoder models to ensure the attention head masking is used."""
        attention_names = ["encoder_attentions", "decoder_attentions", "cross_attentions"]
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            text_config = config.get_text_config()
            if self.has_attentions:
                config._attn_implementation = "eager"  # can't output attentions otherwise

            # We want to test only encoder-decoder models
            if not text_config.is_encoder_decoder:
                continue
            model = model_class(config)

            head_masking = {
                "head_mask": mint.zeros((text_config.encoder_layers, text_config.encoder_attention_heads)),
                "decoder_head_mask": mint.zeros((text_config.decoder_layers, text_config.decoder_attention_heads)),
                "cross_attn_head_mask": mint.zeros((text_config.decoder_layers, text_config.decoder_attention_heads)),
            }

            signature = inspect.signature(model.forward)
            # We want to test only models where encoder/decoder head masking is implemented
            if not set(head_masking.keys()) < {*signature.parameters.keys()}:
                continue

            for attn_name, (name, mask) in zip(attention_names, head_masking.items()):
                out = model.generate(
                    num_beams=1,
                    output_attentions=self.has_attentions,
                    return_dict_in_generate=True,
                    remove_invalid_values=True,
                    **{name: mask},
                    **inputs_dict,
                )
                # We check the state of decoder_attentions and cross_attentions just from the last step
                attn_weights = out[attn_name] if attn_name == attention_names[0] else out[attn_name][-1]
                self.assertEqual(sum([w.sum().item() for w in attn_weights]), 0.0)

    @pytest.mark.generate
    def test_past_key_values_format(self):
        # Test that the KV cache is formatted correctly. Exceptions need to explicitly overwrite this test. Having a
        # standard KV cache format is important for a consistent API (and for advanced generation methods).
        for model_class in self.all_generative_model_classes:
            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            # If it doesn't support cache, pass the test
            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            model = model_class(config).set_train(False)
            if "use_cache" not in inputs:
                inputs["use_cache"] = True
            outputs = model(**inputs)

            # If "past_key_values" is not returned, pass the test (e.g. RWKV uses a different cache name and format)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            text_config = config.get_text_config()
            num_hidden_layers = (
                getattr(text_config, "decoder_layers", None)
                or getattr(text_config, "num_decoder_layers", None)
                or text_config.num_hidden_layers
            )
            num_attention_heads = getattr(text_config, "decoder_attention_heads", text_config.num_attention_heads)
            embed_dim = getattr(text_config, "d_model", text_config.hidden_size)
            per_head_embed_dim = embed_dim // num_attention_heads

            # some models have different num-head for query vs key/value so we need to assign correct value
            # BUT only after `per_head_embed_dim` is set
            num_attention_heads = (
                text_config.num_key_value_heads
                if getattr(text_config, "num_key_value_heads", None) is not None
                else num_attention_heads
            )

            past_kv = outputs["past_key_values"]
            self.assertEqual(len(past_kv), num_hidden_layers)

            # Encoder-Decoder checks
            if config.is_encoder_decoder:
                # encoder-decoder models usually don't have text config
                # below is needed only for Pix2Struct which we cannot modify now due to BC
                config = config.get_text_config()
                encoder_num_attention_heads = (
                    config.encoder_attention_heads
                    if hasattr(config, "encoder_attention_heads")
                    else config.num_attention_heads
                )
                encoder_per_head_embed_dim = embed_dim // encoder_num_attention_heads
                batch_size, seq_length = inputs["decoder_input_ids"].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[i]), 4)  # K V for the decoder + K V for the encoder = 4
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    # The sequence length for the encoder K V depends on the model. Since it is not manipulated in
                    # autoregressive generation, I'm keeping the test general and not checking the 3rd dim
                    self.assertEqual(
                        (past_kv[i][2].shape[0], past_kv[i][2].shape[1], past_kv[i][2].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )
                    self.assertEqual(
                        (past_kv[i][3].shape[0], past_kv[i][3].shape[1], past_kv[i][3].shape[3]),
                        (batch_size, encoder_num_attention_heads, encoder_per_head_embed_dim),
                    )

            # Decoder-only checks
            else:
                # TODO: this line is only needed because of imagegpt, where "pixel_values" = "input_ids". Fix the
                # tests in imagegpt such that `prepare_config_and_inputs_for_common` returns the later (and the other
                # tests use it)
                key = "input_ids" if "input_ids" in inputs else "pixel_values"
                batch_size, seq_length = inputs[key].shape
                for i in range(num_hidden_layers):
                    self.assertEqual(len(past_kv[0]), 2)  # K V for the decoder = 2
                    self.assertEqual(
                        past_kv[i][0].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )
                    self.assertEqual(
                        past_kv[i][1].shape, (batch_size, num_attention_heads, seq_length, per_head_embed_dim)
                    )

    @pytest.mark.generate
    @parameterized.expand([("greedy", 1), ("beam search", 2)])
    def test_generate_from_inputs_embeds(self, _, num_beams):
        """Tests that we can generate from `inputs_embeds` instead of `input_ids` in LLMs, VLMs, etc"""
        # When supported, tests that the decoder model can generate from `inputs_embeds` instead of `input_ids`
        # if fails, you should probably update the `prepare_inputs_for_generation` function
        for model_class in self.all_generative_model_classes:
            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            # This test is for decoder-only models (encoder-decoder models have native input embeddings support in the
            # decoder)
            if config.is_encoder_decoder:
                continue
            config.is_decoder = True

            # Skip models without explicit support
            model = model_class(config).set_train(False)
            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters.keys():
                continue

            # There are a few exception patterns in this test:
            # 1 - Some models can't generate without `input_ids`, when `inputs_embeds` are passed
            requires_inputs_ids = any(model_name in model_class.__name__.lower() for model_name in ["idefics"])
            # 2 - Complex `inputs_embeds` computation, i.e. the correct computation of inputs embeds is more complex
            # than calling the embedding layer with `input_ids`. Subcases of this exception:
            #   2.A - Ignore `scale_embedding`, if the model supports it (it is controlled by a model-dependent flag)
            if hasattr(config, "scale_embedding"):
                config.scale_embedding = False
            #   2.B - Some VLMs assume `inputs_embeds` and `pixel_values` are mutually exclusive AND fall in the
            #   exception above (complex `inputs_embeds` computation). Popping `pixel_values` allow us to run the
            #   checks without adding test complexity. Ditto for `pixel_values_videos` and `pixel_values_images`
            pixel_values_is_mutually_exclusive = any(
                model_name in model_class.__name__.lower() for model_name in VLM_CLASS_NAMES
            )
            if pixel_values_is_mutually_exclusive:
                inputs_dict.pop("pixel_values", None)
                inputs_dict.pop("pixel_values_videos", None)
                inputs_dict.pop("pixel_values_images", None)
            #   2.C - No easy fix, let's skip the check that compares the outputs from `input_ids` and `inputs_embeds`
            has_complex_embeds_computation = any(model_name in model_class.__name__.lower() for model_name in ["moshi"])
            # 3 - `inputs_dict` doesn't contain `attention_mask`. When `attention_mask` is not passed to generate,
            # we infer it from `input_ids`. The last test case will fail if there is a pad token in the original input.
            missing_attention_mask = "attention_mask" not in inputs_dict

            # Traditional way of generating text
            input_ids = inputs_dict.pop("input_ids")
            generation_kwargs = {
                "return_dict_in_generate": True,
                "output_scores": True,
                "num_beams": num_beams,
                "do_sample": False,
                "max_new_tokens": 5,
                "min_new_tokens": 5,  # generate exactly 5 tokens
            }
            outputs_from_ids = model.generate(input_ids, **generation_kwargs, **inputs_dict)
            self.assertEqual(outputs_from_ids.sequences.shape, (input_ids.shape[0], input_ids.shape[1] + 5))

            # Same thing, but from input embeddings (`input_ids` is passed so the prompt is present in the output).
            # The output of the two calls should be the same.
            inputs_embeds = model.get_input_embeddings()(input_ids)
            outputs_from_embeds = model.generate(
                input_ids, inputs_embeds=inputs_embeds, **generation_kwargs, **inputs_dict
            )
            if not has_complex_embeds_computation:
                self._check_similar_generate_outputs(outputs_from_ids, outputs_from_embeds)

            # If we pass different inputs_embeds, we should get different outputs (the output text may be the
            # same, but the logits will almost surely be different)
            random_embeds = mint.rand_like(inputs_embeds)
            outputs_from_rand_embeds = model.generate(
                input_ids, inputs_embeds=random_embeds, **generation_kwargs, **inputs_dict
            )
            for i in range(len(outputs_from_rand_embeds.scores)):
                self.assertFalse(mint.allclose(outputs_from_embeds.scores[i], outputs_from_rand_embeds.scores[i]))

            # input_ids is not a required input on most models -- if we don't pass it, the newly generated tokens will
            # be the same
            if not (requires_inputs_ids or missing_attention_mask):
                outputs_from_embeds_wo_ids = model.generate(
                    inputs_embeds=inputs_embeds, **generation_kwargs, **inputs_dict
                )
                outputs_from_embeds.sequences = outputs_from_embeds.sequences[:, inputs_embeds.shape[1] :]
                self._check_similar_generate_outputs(outputs_from_embeds_wo_ids, outputs_from_embeds)

    @pytest.mark.generate
    def test_generate_from_inputs_embeds_with_static_cache(self):
        """
        Test that StaticCache can generate from inputs_embeds and calculates max_cache_length
        correctly in `generate()`. We force the model to not stop generation until max-length is reached
        to verify that the cache length is indeed set correctly and we don't run out of index when slicing the cache.
        """
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(reason="This model does not support the static cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            if config.is_encoder_decoder:
                self.skipTest(reason="This model is encoder-decoder and has Encoder-Decoder Cache")

            model = model_class(config).set_train(False)
            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters.keys():
                self.skipTest(reason="This model does not support `inputs_embeds` in generation")

            #   Some VLMs assume `inputs_embeds` and `pixel_values` are mutually exclusive AND fall in the
            #   exception above (complex `inputs_embeds` computation). Popping `pixel_values` allow us to run the
            #   checks without adding test complexity. Ditto for `pixel_values_videos` and `pixel_values_images`
            pixel_values_is_mutually_exclusive = any(
                model_name in model_class.__name__.lower() for model_name in VLM_CLASS_NAMES
            )
            if pixel_values_is_mutually_exclusive:
                inputs_dict.pop("pixel_values", None)
                inputs_dict.pop("pixel_values_videos", None)
                inputs_dict.pop("pixel_values_images", None)

            input_ids = inputs_dict.pop("input_ids")

            model.config.use_cache = True
            model.config.is_decoder = True
            batch_size = input_ids.shape[0]
            max_new_tokens = 10

            # here we force to not stop at eos and go until max-length
            model.generation_config.eos_token_id = model.config.get_text_config().eos_token_id = -1
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "cache_implementation": "static",
                "return_dict_in_generate": True,  # Required to return `past_key_values`
            }

            text_config = model.config.get_text_config()
            head_dim = (
                text_config.head_dim
                if hasattr(text_config, "head_dim")
                else text_config.hidden_size // text_config.num_attention_heads
            )
            num_key_value_heads = (
                text_config.num_attention_heads
                if getattr(text_config, "num_key_value_heads", None) is None
                else text_config.num_key_value_heads
            )
            num_hidden_layers = text_config.num_hidden_layers

            inputs_embeds = model.get_input_embeddings()(input_ids)
            outputs = model.generate(inputs_embeds=inputs_embeds, **generation_kwargs, **inputs_dict)

            # we should get `max_length - 1` in shape, not `max_length - embeds_length`.
            # -1 because the last generated token isn't yet in the cache.
            max_length = max_new_tokens + inputs_embeds.shape[1] - 1
            cache_shape = [batch_size, num_key_value_heads, max_length, head_dim]
            self.assertIsInstance(outputs.past_key_values, StaticCache)
            self.assertEqual(len(outputs.past_key_values.key_cache), num_hidden_layers)
            self.assertListEqual(list(outputs.past_key_values.key_cache[0].shape), cache_shape)

    @pytest.mark.generate
    def test_generate_continue_from_past_key_values(self):
        # Tests that we can continue generating from past key values, returned from a previous `generate` call
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt", "mllama"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs = self.model_tester.prepare_config_and_inputs_for_common()

            if not hasattr(config.get_text_config(), "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            # Let's make it always:
            # 1. use cache (for obvious reasons)
            # 2. generate to max length (which can be achieved by setting the eos token to an invalid value), which
            #    would make the test flaky (e.g. EOS is generated on iteration 1 on both generations, but the
            #    continuation would force it to generate beyond an EOS token)
            # 3. ignore `token_type_ids` for simplicity
            # 4. ignore `forced_eos_token_id`, which requires further manipulation of the continuation inputs and is
            #    active by default on some models
            # 5. ignore `encoder_no_repeat_ngram_size`, which is set by default in some encoder-decoder models. When
            #    we use their decoder as a stand-alone model, `encoder_no_repeat_ngram_size` actually prevents
            #    repetition exclusively from the prompt. This test relies on comparing one call vs 2 calls
            #    with cache, what is considered a prompt is different in the two cases.

            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]

            model = model_class(config)
            model.set_train(False)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id = -1
            model.generation_config.forced_eos_token_id = None
            model.generation_config.encoder_no_repeat_ngram_size = 0
            model.generation_config.use_cache = True

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values
            outputs = model.generate(**inputs, do_sample=False, max_new_tokens=4, return_dict_in_generate=True)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens). Note that the
            # inputs may need to be tweaked across `generate` calls (like the attention mask).
            outputs_cached = model.generate(**inputs, do_sample=False, max_new_tokens=3, return_dict_in_generate=True)

            # Continue from the tokens generated above, preparing the inputs accordingly
            inputs["past_key_values"] = outputs_cached.past_key_values
            new_attention_len = outputs_cached.sequences.shape[-1]
            if config.is_encoder_decoder:
                inputs["decoder_input_ids"] = outputs_cached.sequences
                if "decoder_attention_mask" in inputs:
                    inputs["decoder_attention_mask"] = mint.nn.functional.pad(
                        inputs["decoder_attention_mask"],
                        (0, new_attention_len - inputs["decoder_attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            else:
                inputs["input_ids"] = outputs_cached.sequences
                if "attention_mask" in inputs:
                    inputs["attention_mask"] = mint.nn.functional.pad(
                        inputs["attention_mask"],
                        (0, new_attention_len - inputs["attention_mask"].shape[1]),
                        mode="constant",
                        value=1,
                    )
            outputs_cached = model.generate(**inputs, do_sample=False, max_new_tokens=1, return_dict_in_generate=True)

            # The two sets of generated text and past kv should be equal to each other
            self.assertListEqual(outputs.sequences.tolist(), outputs_cached.sequences.tolist())
            for layer_idx in range(len(outputs_cached.past_key_values)):
                for kv_idx in range(len(outputs_cached.past_key_values[layer_idx])):
                    self.assertTrue(
                        mint.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            outputs_cached.past_key_values[layer_idx][kv_idx],
                        )
                    )

    @pytest.mark.generate
    def test_generate_continue_from_inputs_embeds(self):
        """Tests that we can continue generation from `inputs_embeds` and past key values returned from a previous `generate` call."""
        for model_class in self.all_generative_model_classes:
            if any(model_name in model_class.__name__.lower() for model_name in ["imagegpt"]):
                self.skipTest(reason="Won't fix: old model with unique inputs/caches/other")
            if any(model_name in model_class.__name__.lower() for model_name in ["umt5"]):
                self.skipTest(reason="TODO: needs modeling or test input preparation fixes for compatibility")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()

            if "token_type_ids" in inputs_dict:
                del inputs_dict["token_type_ids"]

            if config.is_encoder_decoder:
                self.skipTest(reason="This model is encoder-decoder")
            if not hasattr(config, "use_cache"):
                self.skipTest(reason=f"{model_class.__name__} doesn't support caching")

            model = model_class(config).set_train(False)

            if "inputs_embeds" not in inspect.signature(model.prepare_inputs_for_generation).parameters.keys():
                self.skipTest(reason="This model does not support `inputs_embeds` in generation")

            # If "past_key_values" is not returned, skip the test (e.g. RWKV uses a different cache name and format)
            outputs = model(**inputs_dict)
            if "past_key_values" not in outputs:
                self.skipTest(reason="This model doesn't return `past_key_values`")

            pixel_values_is_mutually_exclusive = any(
                model_name in model_class.__name__.lower() for model_name in VLM_CLASS_NAMES
            )
            if pixel_values_is_mutually_exclusive:
                inputs_dict.pop("pixel_values", None)
                inputs_dict.pop("pixel_values_videos", None)
                inputs_dict.pop("pixel_values_images", None)

            input_ids = inputs_dict.pop("input_ids")

            model.generation_config.pad_token_id = model.generation_config.eos_token_id = -1
            model.generation_config.forced_eos_token_id = None
            model.config.is_decoder = True
            model.generation_config.use_cache = True

            generation_kwargs = {
                "return_dict_in_generate": True,
                "do_sample": False,
            }

            # Traditional way of generating text, with `return_dict_in_generate` to return the past key values.
            input_embeds = model.get_input_embeddings()(input_ids)
            outputs = model.generate(inputs_embeds=input_embeds, max_new_tokens=4, **generation_kwargs)

            # Let's generate again, but passing the past key values in between (3 + 1 = 4 tokens)
            initial_output = model.generate(inputs_embeds=input_embeds, max_new_tokens=3, **generation_kwargs)
            continued_embeds = mint.cat([input_embeds, model.get_input_embeddings()(initial_output.sequences)], dim=1)
            cached_output = model.generate(
                inputs_embeds=continued_embeds,
                max_new_tokens=1,
                past_key_values=initial_output.past_key_values,
                **generation_kwargs,
            )

            # Combine the (3 + 1) generated tokens and verify it matches with full generation.
            combined_output_sequences = mint.concat([initial_output.sequences, cached_output.sequences], axis=1)
            self.assertListEqual(outputs.sequences.tolist(), combined_output_sequences.tolist())
            # The two sets of past kv should be equal to each other
            for layer_idx in range(len(cached_output.past_key_values)):
                for kv_idx in range(len(cached_output.past_key_values[layer_idx])):
                    self.assertTrue(
                        mint.allclose(
                            outputs.past_key_values[layer_idx][kv_idx],
                            cached_output.past_key_values[layer_idx][kv_idx],
                        )
                    )

    @pytest.mark.generate
    def test_generate_with_static_cache(self):
        """
        Tests that generating with static cache give almost same results as with dynamic cache, and the output cache
        has the expected shapes
        """
        set_model_tester_for_less_flaky_test(self)
        for model_class in self.all_generative_model_classes:
            if not model_class._supports_static_cache:
                self.skipTest(reason="This model does not support the static cache format")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            set_config_for_less_flaky_test(config)
            main_input = inputs_dict[model_class.main_input_name]

            if config.is_encoder_decoder:
                self.skipTest(reason="This model is encoder-decoder and has Encoder-Decoder Cache")

            config.is_decoder = True
            batch_size = main_input.shape[0]
            seq_length = self.model_tester.seq_length
            max_new_tokens = 20

            for dtype in (ms.float32, ms.float16):
                model = model_class(config).to(dtype).set_train(False)
                inputs_dict = {
                    k: v.to(dtype) if isinstance(v, ms.tensor) and ops.is_floating_point(v) else v
                    for k, v in inputs_dict.items()
                }
                set_model_for_less_flaky_test(model)

                generation_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "return_dict_in_generate": True,  # Required to return `past_key_values`
                    "output_scores": True,
                    "use_cache": True,
                }

                static_cache_generation = model.generate(
                    **generation_kwargs, **inputs_dict, cache_implementation="static"
                )

                # Check 1: The cache shapes must match the expected shapes
                max_cache_len = seq_length + max_new_tokens - 1  # cache len = gen len - 1, the last token has no cache
                text_config = config.text_config if hasattr(config, "text_config") else config
                head_dim = (
                    text_config.head_dim
                    if hasattr(text_config, "head_dim")
                    else text_config.hidden_size // text_config.num_attention_heads
                )
                num_key_value_heads = (
                    text_config.num_attention_heads
                    if getattr(text_config, "num_key_value_heads", None) is None
                    else text_config.num_key_value_heads
                )
                num_hidden_layers = text_config.num_hidden_layers
                cache_shape = (batch_size, num_key_value_heads, max_cache_len, head_dim)
                self.assertTrue(isinstance(static_cache_generation.past_key_values, StaticCache))
                self.assertTrue(len(static_cache_generation.past_key_values.key_cache) == num_hidden_layers)
                self.assertTrue(static_cache_generation.past_key_values.key_cache[0].shape == cache_shape)

                # Check 2: The outputs must be similar to the case with dynamic cache
                dynamic_cache_generation = model.generate(**generation_kwargs, **inputs_dict)
                self._check_similar_generate_outputs(dynamic_cache_generation, static_cache_generation)

    @pytest.mark.generate
    def test_generate_methods_with_logits_to_keep(self):
        for model_class in self.all_generative_model_classes:
            if "logits_to_keep" not in set(inspect.signature(model_class.forward).parameters.keys()):
                self.skipTest(reason="This model does not support `logits_to_keep` argument.")

            config, inputs_dict = self.prepare_config_and_inputs_for_generate()
            config.use_cache = True
            config.is_decoder = True

            model = model_class(config).set_train(False)
            # All generation methods (except assisted decoding) rely on always extracting the last token logits of the
            # full logits matrix, so testing out only greedy search and assisted decoding is enough (if it works,
            # other methods will work as well)
            generation_kwargs = {
                "max_new_tokens": 10,
                "do_sample": False,
            }

            # Setting logits_to_keep at 0 keeps all logits (old behavior)
            with_all_logits = model.generate(**generation_kwargs, **inputs_dict, logits_to_keep=0)
            # By default, logits_to_keep is automatically set to 1 if not provided (new behavior)
            without_all_logits = model.generate(**inputs_dict, **generation_kwargs)
            self.assertEqual(with_all_logits.tolist(), without_all_logits.tolist())

    @pytest.mark.generate
    def test_inherits_generation_mixin(self):
        """
        Tests that the model class directly inherits `GenerationMixin`, as opposed to relying on `PreTrainedModel`
        to inherit it.
        """
        for model_class in self.all_generative_model_classes:
            self.assertTrue("GenerationMixin" in str(model_class.__bases__))

    def _test_attention_implementation(self, attn_implementation):
        """
        Compares the output of generate with the eager attention implementation against other implementations.
        NOTE: despite the test logic being the same, different implementations actually need different decorators, hence
        this separate function.
        """
        max_new_tokens = 30
        support_flag = {
            "sdpa": "_supports_sdpa",
            "flash_attention_2": "_supports_flash_attn_2",
        }

        for model_class in self.all_generative_model_classes:
            if not getattr(model_class, support_flag[attn_implementation]):
                self.skipTest(f"{model_class.__name__} does not support `attn_implementation={attn_implementation}`")

            config, original_inputs_dict = self.prepare_config_and_inputs_for_generate()
            inputs_dict = {}
            for input_name, input_data in original_inputs_dict.items():
                if isinstance(input_data, ms.tensor) and input_data.dtype in [ms.float32, ms.bfloat16]:
                    inputs_dict[input_name] = input_data.to(ms.float16)
                else:
                    inputs_dict[input_name] = input_data
            main_input = inputs_dict[model_class.main_input_name]

            # make sure that all models have enough positions for generation
            if hasattr(config, "max_position_embeddings"):
                config.max_position_embeddings = max_new_tokens + main_input.shape[1] + 1

            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                del model
                gc.collect()

                generate_kwargs = {
                    "max_new_tokens": max_new_tokens,
                    "do_sample": False,
                    "return_dict_in_generate": True,
                    "output_scores": True,
                    "use_cache": True,
                }

                model_eager = model_class.from_pretrained(
                    tmpdirname,
                    mindspore_dtype=ms.float16,
                    attn_implementation="eager",
                )
                res_eager = model_eager.generate(**inputs_dict, **generate_kwargs)
                del model_eager
                gc.collect()

                model_attn = model_class.from_pretrained(
                    tmpdirname,
                    mindspore_dtype=ms.float16,
                    attn_implementation=attn_implementation,
                )
                res_attn = model_attn.generate(**inputs_dict, **generate_kwargs)
                del model_attn
                gc.collect()

                self._check_similar_generate_outputs(res_eager, res_attn, atol=1e-3, rtol=1e-3)

    @pytest.mark.generate
    @require_mindspore
    @slow
    def test_eager_matches_sdpa_generate(self):
        """Tests that generate has equivalent outputs with SDPA and eager attention implementations."""
        self._test_attention_implementation("sdpa")

    def _check_generate_outputs(self, output, config, use_cache=False, num_return_sequences=1, num_beams=1):
        input_batch_size = int(output.sequences.shape[0] / num_return_sequences)
        internal_batch_size = input_batch_size * num_beams if num_beams > 1 else input_batch_size * num_return_sequences

        prompt_length = getattr(self.model_tester, "seq_length", None)
        prompt_length = getattr(self.model_tester, "encoder_seq_length", prompt_length)
        prompt_length = getattr(self.model_tester, "text_seq_length", prompt_length)

        config = config.text_config if hasattr(config, "text_config") else config

        generated_length = (
            output.sequences.shape[-1] - 1 if config.is_encoder_decoder else output.sequences.shape[-1] - prompt_length
        )
        decoder_past_key_values = getattr(output, "past_key_values", None)
        if config.is_encoder_decoder and isinstance(decoder_past_key_values, EncoderDecoderCache):
            decoder_past_key_values = decoder_past_key_values.self_attention_cache

        # in some models we subsample the sequence length in inner layers
        if hasattr(self.model_tester, "get_subsampled_output_lengths"):
            prompt_length = self.model_tester.get_subsampled_output_lengths(prompt_length)

        # scores
        self._check_scores(
            batch_size=internal_batch_size, scores=output.scores, generated_length=generated_length, config=config
        )

        # unprocessed logits
        self._check_logits(batch_size=internal_batch_size, logits=output.logits, config=config)

        # Attentions
        if self.has_attentions:
            if config.is_encoder_decoder:
                # encoder
                self._check_encoder_attention_for_generate(
                    attentions=output.encoder_attentions,
                    batch_size=input_batch_size,
                    config=config,
                    prompt_length=prompt_length,
                )
                # decoder
                self._check_attentions_for_generate(
                    batch_size=internal_batch_size,
                    attentions=output.decoder_attentions,
                    prompt_length=1,  # the BOS token
                    output_length=output.sequences.shape[-1],
                    config=config,
                    decoder_past_key_values=decoder_past_key_values,
                )
            else:
                self._check_attentions_for_generate(
                    batch_size=internal_batch_size,
                    attentions=output.attentions,
                    prompt_length=prompt_length,
                    output_length=output.sequences.shape[-1],
                    config=config,
                    decoder_past_key_values=decoder_past_key_values,
                )

        # Hidden States
        if config.is_encoder_decoder:
            # encoder
            self._check_encoder_hidden_states_for_generate(
                hidden_states=output.encoder_hidden_states,
                batch_size=input_batch_size,
                config=config,
                prompt_length=prompt_length,
            )
            # decoder
            self._check_hidden_states_for_generate(
                batch_size=internal_batch_size,
                hidden_states=output.decoder_hidden_states,
                prompt_length=1,  # the BOS token
                output_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )
        else:
            self._check_hidden_states_for_generate(
                batch_size=internal_batch_size,
                hidden_states=output.hidden_states,
                prompt_length=prompt_length,
                output_length=output.sequences.shape[-1],
                config=config,
                use_cache=use_cache,
            )

        # Past Key Value States -- a few notes here:
        # 1. Its inner sequence length is with respect to the inputs of the latest forward pass, hence the "-1"
        # 2. We ignore models that have unique cache structures (e.g. mamba) or are in need of refatoring to match the
        #    standard cache format (e.g.gptbigcode )
        models_without_standard_cache = (
            "bamba",
            "ctrl",
            "fsmt",
            "gptbigcode",
            "mega",
            "reformer",
            "jamba",
            "mamba",
            "xlnet",
            "zamba",
            "zamba2",
        )
        has_standard_cache = not any(
            model_name in config.__class__.__name__.lower() for model_name in models_without_standard_cache
        )
        if has_standard_cache:
            if use_cache:
                cache_length = output.sequences.shape[-1] - 1
                self._check_past_key_values_for_generate(
                    batch_size=internal_batch_size,
                    decoder_past_key_values=decoder_past_key_values,
                    cache_length=cache_length,
                    config=config,
                )
            elif use_cache is False:
                self.assertTrue(decoder_past_key_values is None)

    def _check_scores(self, batch_size, scores, generated_length, config):
        vocab_size = config.get_text_config(decoder=True).vocab_size
        expected_shape = (batch_size, vocab_size)
        self.assertIsInstance(scores, tuple)
        self.assertEqual(len(scores), generated_length)
        self.assertListEqual([iter_scores.shape for iter_scores in scores], [expected_shape] * len(scores))

    def _check_logits(self, batch_size, logits, config):
        vocab_size = config.get_text_config(decoder=True).vocab_size
        self.assertIsInstance(logits, tuple)
        self.assertListEqual([iter_logits.shape[0] for iter_logits in logits], [batch_size] * len(logits))
        # vocabulary difference equal to one (imagegptmodel?) or zero (all other models)
        vocab_diff = vocab_size - logits[0].shape[-1]
        self.assertTrue(vocab_diff in [0, 1])
        self.assertListEqual([vocab_size - score.shape[-1] for score in logits], [vocab_diff] * len(logits))

    def _check_attentions_for_generate(
        self, batch_size, attentions, prompt_length, output_length, config, decoder_past_key_values
    ):
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [isinstance(iter_attentions, tuple) for iter_attentions in attentions], [True] * len(attentions)
        )
        self.assertEqual(len(attentions), (output_length - prompt_length))

        use_cache = decoder_past_key_values is not None
        has_static_cache = isinstance(decoder_past_key_values, (StaticCache, HybridCache))

        # When `output_attentions=True`, each iteration of generate appends the attentions corresponding to the new
        # token(s)
        # NOTE: `HybridCache` may have different lengths on different layers, if this test starts failing add more
        # elaborate checks
        for generated_length, iter_attentions in enumerate(attentions):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
            query_length = (
                prompt_length + generated_length
                if not has_static_cache
                else decoder_past_key_values.get_max_cache_shape()
            )

            expected_shape = (
                batch_size,
                config.num_attention_heads,
                model_input_length,
                query_length,
            )
            # check attn size
            self.assertListEqual(
                [layer_attention.shape for layer_attention in iter_attentions], [expected_shape] * len(iter_attentions)
            )

    def _check_encoder_attention_for_generate(self, attentions, batch_size, config, prompt_length):
        encoder_expected_shape = (batch_size, config.num_attention_heads, prompt_length, prompt_length)
        self.assertIsInstance(attentions, tuple)
        self.assertListEqual(
            [layer_attentions.shape for layer_attentions in attentions],
            [encoder_expected_shape] * len(attentions),
        )

    def _check_hidden_states_for_generate(
        self, batch_size, hidden_states, prompt_length, output_length, config, use_cache=False
    ):
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [isinstance(iter_hidden_states, tuple) for iter_hidden_states in hidden_states],
            [True] * len(hidden_states),
        )
        self.assertEqual(len(hidden_states), (output_length - prompt_length))

        # When `output_hidden_states=True`, each iteration of generate appends the hidden states corresponding to the
        # new token(s)
        # NOTE: `HybridCache` may have different lengths on different layers, if this test starts failing add more
        # elaborate checks
        for generated_length, iter_hidden_states in enumerate(hidden_states):
            # regardless of using cache, the first forward pass will have the full prompt as input
            if use_cache and generated_length > 0:
                model_input_length = 1
            else:
                model_input_length = prompt_length + generated_length
            expected_shape = (batch_size, model_input_length, config.hidden_size)
            # check hidden size
            self.assertListEqual(
                [layer_hidden_states.shape for layer_hidden_states in iter_hidden_states],
                [expected_shape] * len(iter_hidden_states),
            )

    def _check_encoder_hidden_states_for_generate(self, hidden_states, batch_size, config, prompt_length):
        encoder_expected_shape = (batch_size, prompt_length, config.hidden_size)
        self.assertIsInstance(hidden_states, tuple)
        self.assertListEqual(
            [layer_hidden_states.shape for layer_hidden_states in hidden_states],
            [encoder_expected_shape] * len(hidden_states),
        )

    def _check_past_key_values_for_generate(self, batch_size, decoder_past_key_values, cache_length, config):
        self.assertIsInstance(decoder_past_key_values, (tuple, Cache))

        # (batch, head, seq_length, head_features)
        expected_shape = (
            batch_size,
            config.num_key_value_heads if hasattr(config, "num_key_value_heads") else config.num_attention_heads,
            cache_length,
            config.hidden_size // config.num_attention_heads,
        )

        if isinstance(decoder_past_key_values, Cache):
            self.assertListEqual(
                [key_tensor.shape for key_tensor in decoder_past_key_values.key_cache],
                [expected_shape] * len(decoder_past_key_values.key_cache),
            )
            self.assertListEqual(
                [value_tensor.shape for value_tensor in decoder_past_key_values.value_cache],
                [expected_shape] * len(decoder_past_key_values.value_cache),
            )

        # Legacy cache format checks. This branch should be removed when all models use `Cache` by default
        else:
            self.assertListEqual(
                [isinstance(iter_past_key_values, tuple) for iter_past_key_values in decoder_past_key_values],
                [True] * len(decoder_past_key_values),
            )
            # check shape key, value
            self.assertListEqual(
                [layer_past_key_values[0].shape for layer_past_key_values in decoder_past_key_values],
                [expected_shape] * len(decoder_past_key_values),
            )
            self.assertListEqual(
                [layer_past_key_values[1].shape for layer_past_key_values in decoder_past_key_values],
                [expected_shape] * len(decoder_past_key_values),
            )

    def _check_sequence_inside_sequence(self, tensor_1, tensor_2):
        # check if tensor_1 inside tensor_2 or tensor_2 inside tensor_1.
        # set to same device. we don't care what device.

        if not isinstance(tensor_1, list):
            tensor_1 = tensor_1.tolist()
        if not isinstance(tensor_2, list):
            tensor_2 = tensor_2.tolist()

        in_order = len(tensor_1) <= len(tensor_2)
        longer = tensor_2 if in_order else tensor_1
        shorter = tensor_1 if in_order else tensor_2

        flag = False
        chunk_size = len(shorter)
        for chunk_idx in range(len(longer) - chunk_size + 1):
            subseq = longer[chunk_idx : chunk_idx + chunk_size]
            if subseq == shorter:
                flag = True
                break

        self.assertTrue(flag)


@require_mindspore
class UtilsFunctionsTest(unittest.TestCase):
    def test_speculative_sampling(self):
        # assume vocab size 10, input length 5 + 3 generated candidates
        candidate_input_ids = ms.tensor([[8, 0, 3, 9, 8, 1, 4, 5]])  # input tokens
        candidate_logits = ms.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 4
                    [-10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0],  # generated 5
                ]
            ]
        )
        candidate_length = 3
        inf = float("inf")
        new_logits = ms.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # accepts 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # accepts 4
                    [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, 10.0, -inf],  # rejects 5, accepts 8
                    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # N/A
                ]
            ]
        )
        last_assistant_token_is_eos = False
        validated_tokens, n_matches = _speculative_sampling(
            candidate_input_ids,
            candidate_logits,
            candidate_length,
            new_logits,
            last_assistant_token_is_eos,
        )
        self.assertTrue(n_matches.item() == 2)
        self.assertTrue(validated_tokens.tolist()[0] == [1, 4, 8])

    def test_speculative_sampling_target_distribution(self):
        """
        Asserts that the target distribution is preserved.
        Should help with catching issues like #32867.
        """
        # assume vocab size 10, input length 5 + 3 generated candidates
        candidate_input_ids = ms.tensor([[8, 0, 3, 9, 8, 1, 4, 5]])  # input tokens
        candidate_logits = ms.tensor(
            [
                [
                    [-10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 1
                    [-10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0, -10.0],  # generated 4
                    [-10.0, -10.0, -10.0, -10.0, -10.0, 10.0, -10.0, -10.0, -10.0, -10.0],  # generated 5
                ]
            ]
        )
        candidate_length = 3
        inf = float("inf")
        new_logits = ms.tensor(
            [
                [
                    # accepts 1:
                    [-inf, 10.0, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                    # accepts 4:
                    [-inf, -inf, -inf, -inf, 10.0, -inf, -inf, -inf, -inf, -inf],
                    # most likely to be 1 or 8, less likely to be 3, then 7, and should never be any other value:
                    [-inf, 2.0, -inf, 1.0, -inf, -inf, -inf, -0.01, 2.0, -inf],
                    # N/A:
                    [-inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
                ]
            ]
        )
        last_assistant_token_is_eos = False
        last_validated_token = []
        for _ in range(10_000):
            validated_tokens, n_matches = _speculative_sampling(
                candidate_input_ids,
                candidate_logits,
                candidate_length,
                new_logits,
                last_assistant_token_is_eos,
            )
            self.assertTrue(n_matches.item() == 2)
            self.assertTrue(validated_tokens.tolist()[0][0] == 1)
            self.assertTrue(validated_tokens.tolist()[0][1] == 4)
            self.assertTrue(validated_tokens.tolist()[0][2] in [1, 3, 7, 8])
            last_validated_token.append(validated_tokens.tolist()[0][2])
        # check that the most likely tokens are selected more often than the less likely ones
        last_token_counts = collections.Counter(last_validated_token)
        self.assertTrue(last_token_counts[1] > last_token_counts[3] > last_token_counts[7] > 0)
        self.assertTrue(last_token_counts[8] > last_token_counts[3])


global_rng = random.Random()


# Copied from tests.test_modeling_common.ids_tensor
def ids_tensor(shape, vocab_size, rng=None, name=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return ms.tensor(values, dtype=ms.int64).view(shape).contiguous()


# Copied from tests.test_modeling_common.floats_tensor
def floats_tensor(shape, scale=1.0, rng=None, name=None):
    """Creates a random float32 tensor"""
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.random() * scale)

    return ms.tensor(values, dtype=ms.float32).view(shape).contiguous()


@pytest.mark.generate
@require_mindspore
class GenerationIntegrationTests(unittest.TestCase):
    """Check the mean bias inserted by the watermarking algorithm."""

    @slow
    def test_beam_search_example_integration(self):
        # exactly the example provided in the docstrings of beam search, which previously
        # failed after directly copying from it. Refer to PR #15555
        tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-base")
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-base")

        encoder_input_str = "translate English to German: How old are you?"
        encoder_input_ids = ms.tensor(tokenizer(encoder_input_str, return_tensors="np").input_ids)

        # lets run beam search using 3 beams
        num_beams = 3
        # define decoder start token ids
        input_ids = mint.ones((1, 1), dtype=ms.int64)
        input_ids = input_ids * model.config.decoder_start_token_id

        # add encoder_outputs to model keyword arguments
        model_kwargs = {"encoder_outputs": model.get_encoder()(encoder_input_ids, return_dict=True)}

        outputs = model.generate(
            input_ids, num_beams=num_beams, min_length=5, eos_token_id=model.config.eos_token_id, **model_kwargs
        )
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        self.assertListEqual(outputs, ["Wie alt bist du?"])

    def test_special_tokens_fall_back_to_model_default(self):
        model = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-MistralForCausalLM", revision="refs/pr/29"
        )
        test_bos_id = 50

        # Sanity-check: the model has a BOS token set, and the first generated token is a BOS token
        gen_output = model.generate()
        self.assertTrue(model.generation_config.bos_token_id is not None)
        self.assertTrue(model.generation_config.bos_token_id == gen_output[0, 0])

        # If we pass a generation config **with** a BOS token, `generate` will use it
        generation_config = GenerationConfig(bos_token_id=test_bos_id)
        gen_output = model.generate(generation_config=generation_config)
        self.assertFalse(model.generation_config.bos_token_id == gen_output[0, 0])
        self.assertTrue(generation_config.bos_token_id == gen_output[0, 0])
        self.assertTrue(test_bos_id == gen_output[0, 0])

        # If we pass a generation config **without** a BOS token, `generate` will fetch the BOS token from
        # `model.generation_config`
        generation_config = GenerationConfig(bos_token_id=None)
        gen_output = model.generate(generation_config=generation_config)
        self.assertTrue(model.generation_config.bos_token_id == gen_output[0, 0])
        self.assertFalse(test_bos_id == gen_output[0, 0])
        self.assertTrue(generation_config.bos_token_id is None)

        # Changing `model.generation_config` will affect fallback behavior
        model.generation_config.bos_token_id = test_bos_id
        gen_output = model.generate(generation_config=generation_config)
        self.assertTrue(model.generation_config.bos_token_id == gen_output[0, 0])
        self.assertTrue(test_bos_id == gen_output[0, 0])
        self.assertTrue(generation_config.bos_token_id is None)

    def test_validate_generation_inputs(self):
        """Tests validation of inputs to `generate`"""
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-t5", revision="refs/pr/1")
        model = AutoModelForSeq2SeqLM.from_pretrained("hf-internal-testing/tiny-random-t5", revision="refs/pr/1")

        encoder_input_str = "Hello world"
        input_ids = ms.tensor(tokenizer(encoder_input_str, return_tensors="np").input_ids)

        # typos are quickly detected (the correct argument is `do_sample`)
        with self.assertRaisesRegex(ValueError, "do_samples"):
            model.generate(input_ids, do_samples=True)

        # arbitrary arguments that will not be used anywhere are also not accepted
        with self.assertRaisesRegex(ValueError, "foo"):
            fake_model_kwargs = {"foo": "bar"}
            model.generate(input_ids, **fake_model_kwargs)

        # however, valid model_kwargs are accepted
        valid_model_kwargs = {"attention_mask": ms.tensor(np.zeros_like(input_ids))}
        model.generate(input_ids, **valid_model_kwargs)

    @slow
    def test_transition_scores_early_stopping(self):
        """
        Test that `compute_transition_scores` is working as expected with beam search and early stopping

        This is an aggressive test that makes sure that `beam_search's`
        transition scores are computed correctly for varying `num_return_sequences`, `num_beams` and `batch_size > 1`
        2 x input_ids for "question: How are you? \n context: I had a long day, "
        """
        input_ids = ms.tensor(2 * [[822, 10, 571, 33, 25, 58, 2625, 10, 27, 141, 3, 9, 307, 239, 6, 1]])
        model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small")

        outputs = model.generate(
            input_ids,
            max_length=10,
            return_dict_in_generate=True,
            output_scores=True,
            forced_eos_token_id=model.config.eos_token_id,
            num_beams=4,
            do_sample=False,
            num_return_sequences=3,
            length_penalty=0.0,
        )

        transition_scores = model.compute_transition_scores(
            sequences=outputs.sequences, scores=outputs.scores, beam_indices=outputs.beam_indices
        )
        transition_scores = transition_scores.asnumpy()
        outputs.sequences_scores = outputs.sequences_scores.asnumpy()

        self.assertTrue(np.allclose(np.sum(transition_scores, axis=-1), outputs.sequences_scores))

    def test_generate_inputs_and_encoder_kwargs(self):
        """
        Test that an exception is thrown if the main tensor (`input_ids` in LLMs) is passed as both a positional and
        keyword argument
        """
        article = "I need input_ids to generate"
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2", max_length=10)
        input_ids = ms.tensor(tokenizer(article, return_tensors="np").input_ids)
        with self.assertRaises(ValueError):
            model.generate(input_ids, input_ids=input_ids)

    def test_generate_too_many_encoder_kwargs(self):
        """Test that passing redundant inputs results in an exception (`input_ids` and `inputs_embeds` in LLMs)"""
        article = "I need input_ids to generate"
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-bart", revision="refs/pr/1")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "hf-internal-testing/tiny-random-bart", revision="refs/pr/1", max_length=10
        )
        input_ids = ms.tensor(tokenizer(article, return_tensors="np").input_ids)
        with self.assertRaises(ValueError):
            model.generate(input_ids=input_ids, inputs_embeds=input_ids)


@require_mindspore
class TokenHealingTestCase(unittest.TestCase):
    @parameterized.expand(
        [
            ("trailing_whitespace", "I read a book about ", "I read a book about"),
            ("nothing_to_heal", "I read a book about", "I read a book about"),
            ("single_token", "I", "I"),
            ("empty_prompt", "", ""),
        ]
    )
    def test_prompts(self, name, input, expected):
        """
        tokenizer.pad_token value can be empty but it is required in the latter codes
        so assigned it here with eos_token
        """
        model_name_or_path = "distilbert/distilgpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
        completion_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            revision="main",
            trust_remote_code=False,
            use_cache=True,
        )
        tokenizer.pad_token = tokenizer.eos_token

        input_ids = ms.tensor(tokenizer(input, return_tensors="np").input_ids)

        healed_ids = completion_model.heal_tokens(input_ids, tokenizer=tokenizer)
        predicted = tokenizer.decode(healed_ids[0], skip_special_tokens=True)

        self.assertEqual(predicted, expected)


class TestAssistedCandidateGeneratorDifferentTokenizers(unittest.TestCase):
    def test_no_intersection(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[4, 5, 6]])
        result = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(prompt, prompt_plus_new_tokens)
        self.assertEqual(result, (None, None, None))

    def test_complete_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_partial_overlap(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[2, 3, 4, 5]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[4, 5]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))

    def test_no_new_tokens(self):
        prompt = np.array([[1, 2, 3]])
        prompt_plus_new_tokens = np.array([[1, 2, 3]])
        discrep_length, new_tokens_only, discrep_only = AssistedCandidateGeneratorDifferentTokenizers._get_tokens_diag(
            prompt, prompt_plus_new_tokens
        )
        self.assertEqual(discrep_length, 0)
        np.testing.assert_array_equal(new_tokens_only, np.array([[]]))
        np.testing.assert_array_equal(discrep_only, np.array([[]]))
