# coding=utf-8
# Copyright 2020 The HuggingFace Team Inc.
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

import unittest

from mindone.transformers.mindspore_adapter import dtype_to_max, dtype_to_min
from mindone.transformers.testing_utils import require_mindspore
from mindone.transformers.utils import is_mindspore_available
from tests.transformers_tests.test_modeling_common import ids_tensor

if is_mindspore_available():
    import mindspore as ms
    from mindspore import mint

    from mindone.transformers.generation import (
        EncoderNoRepeatNGramLogitsProcessor,
        EncoderRepetitionPenaltyLogitsProcessor,
        ExponentialDecayLengthPenalty,
        ForcedBOSTokenLogitsProcessor,
        ForcedEOSTokenLogitsProcessor,
        HammingDiversityLogitsProcessor,
        InfNanRemoveLogitsProcessor,
        LogitNormalization,
        NoRepeatNGramLogitsProcessor,
        RepetitionPenaltyLogitsProcessor,
        SequenceBiasLogitsProcessor,
        TopKLogitsWarper,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
    )
    from mindone.transformers.generation.logits_process import BarkEosPrioritizerLogitsProcessor


@require_mindspore
class LogitsProcessorTest(unittest.TestCase):
    def _get_uniform_logits(self, batch_size: int, length: int):
        scores = mint.ones((batch_size, length), dtype=ms.float32) / length
        return scores

    def test_repetition_penalty_dist_process(self):
        input_ids = ms.tensor([[0, 1], [5, 0]], dtype=ms.int64)
        vocab_size = 10

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        # give values special values
        scores[0, 0] = -(1 / vocab_size)
        scores[1, 5] = 4 / vocab_size

        rep_penalty_proc = RepetitionPenaltyLogitsProcessor(penalty=2.0)

        processed_scores = rep_penalty_proc(input_ids, scores)

        # check that values were correctly changed
        self.assertAlmostEqual(processed_scores[0, 0].item(), -(1 / vocab_size) * 2)
        self.assertAlmostEqual(processed_scores[0, 1].item(), (1 / vocab_size) / 2)

        self.assertAlmostEqual(processed_scores[1, 0].item(), (1 / vocab_size) / 2)
        self.assertAlmostEqual(processed_scores[1, 5].item(), (4 / vocab_size) / 2)

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == processed_scores))

    def test_encoder_repetition_penalty_dist_process(self):
        input_ids = ms.tensor([[0, 1], [5, 0]], dtype=ms.int64)
        vocab_size = 10

        scores = self._get_uniform_logits(batch_size=2, length=vocab_size)

        # give values special values
        scores[0, 0] = -(1 / vocab_size)
        scores[1, 5] = 4 / vocab_size

        rep_penalty_proc = EncoderRepetitionPenaltyLogitsProcessor(penalty=2.0, encoder_input_ids=input_ids)

        processed_scores = rep_penalty_proc(input_ids, scores)

        # check that values were correctly changed
        self.assertAlmostEqual(processed_scores[0, 0].item(), -(1 / vocab_size) / 2)
        self.assertAlmostEqual(processed_scores[0, 1].item(), (1 / vocab_size) * 2)

        self.assertAlmostEqual(processed_scores[1, 0].item(), (1 / vocab_size) * 2)
        self.assertAlmostEqual(processed_scores[1, 5].item(), (4 / vocab_size) * 2)

        # check that values not in the encoder ids were NOT changed
        self.assertAlmostEqual(processed_scores[0, 2].item(), (1 / vocab_size))
        self.assertAlmostEqual(processed_scores[1, 2].item(), (1 / vocab_size))

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == processed_scores))

    def test_top_k_dist_warper(self):
        input_ids = None
        vocab_size = 10
        batch_size = 2

        # create ramp distribution
        ramp_logits = mint.arange(vocab_size, dtype=ms.float32).unsqueeze(0).repeat(batch_size, 1)
        ramp_logits[1:, : vocab_size // 2] = ramp_logits[1:, : vocab_size // 2] + vocab_size

        top_k_warp = TopKLogitsWarper(3)

        scores = top_k_warp(input_ids, ramp_logits)

        # check that correct tokens are filtered
        self.assertListEqual(mint.isinf(scores[0]).tolist(), 7 * [True] + 3 * [False])
        self.assertListEqual(mint.isinf(scores[1]).tolist(), 2 * [True] + 3 * [False] + 5 * [True])

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == ramp_logits))

        # check special cases
        length = 5

        logits = self._get_uniform_logits(batch_size=batch_size, length=length)
        top_k_warp_safety_check = TopKLogitsWarper(top_k=1, filter_value=0.0, min_tokens_to_keep=3)

        scores = top_k_warp_safety_check(input_ids, logits)
        # uniform dist is not changed
        self.assertListEqual((scores == 0.0).to(ms.int64).sum(dim=-1).tolist(), [0, 0])

        ramp_logits = mint.arange(length, dtype=ms.float32).unsqueeze(0).repeat(batch_size, 1)
        scores = top_k_warp_safety_check(input_ids, ramp_logits)

        # min_tokens overwrites k: 3 tokens are kept => 2 tokens are nullified
        self.assertListEqual((scores == 0.0).to(ms.int64).sum(dim=-1).tolist(), [2, 2])

    def test_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        batch_size = 2

        input_ids = ms.tensor([[1, 1, 2, 1], [0, 1, 0, 1]], dtype=ms.int64)
        scores = self._get_uniform_logits(batch_size, vocab_size)

        no_repeat_proc_2_gram = NoRepeatNGramLogitsProcessor(2)
        no_repeat_proc_3_gram = NoRepeatNGramLogitsProcessor(3)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, scores)
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, scores)

        # 2-gram would forbid 2nd and 3rd token (1,2) at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(mint.isinf(filtered_scores_2_gram).tolist(), [[False, True, True], [True, False, False]])

        # 3-gram would forbid no token at 1st batch and 1st token (0) at 2nd batch
        self.assertListEqual(mint.isinf(filtered_scores_3_gram).tolist(), [[False, False, False], [True, False, False]])

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == filtered_scores_2_gram))
        self.assertFalse(mint.all(scores == filtered_scores_3_gram))

    def test_encoder_no_repeat_ngram_dist_processor(self):
        vocab_size = 3
        num_beams = 2
        batch_size = 1

        encoder_input_ids = ms.tensor([1, 2, 1, 1], dtype=ms.int64)

        input_ids = ms.tensor([[1, 2, 1], [8, 0, 2]], dtype=ms.int64)
        scores = self._get_uniform_logits(batch_size * num_beams, vocab_size)

        no_repeat_proc_2_gram = EncoderNoRepeatNGramLogitsProcessor(2, encoder_input_ids=encoder_input_ids)
        no_repeat_proc_3_gram = EncoderNoRepeatNGramLogitsProcessor(3, encoder_input_ids=encoder_input_ids)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, scores)
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, scores)

        # 2-gram would forbid 1st and 2nd token at 1st beam and 1st token (0) at 2nd beam
        self.assertListEqual(mint.isinf(filtered_scores_2_gram).tolist(), [[False, True, True], [False, True, False]])

        # 3-gram would forbid 1st token at 1st beam and no token at 2nd beam
        self.assertListEqual(mint.isinf(filtered_scores_3_gram).tolist(), [[False, True, False], [False, False, False]])

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == filtered_scores_2_gram))
        self.assertFalse(mint.all(scores == filtered_scores_3_gram))

        # Batched input
        vocab_size = 3
        num_beams = 2
        batch_size = 2
        encoder_input_ids = ms.tensor([[1, 2, 1, 1], [0, 0, 2, 1]], dtype=ms.int64)

        input_ids = ms.tensor([[1, 2, 1], [1, 0, 2], [0, 0, 0], [0, 2, 2]], dtype=ms.int64)
        scores = self._get_uniform_logits(batch_size * num_beams, vocab_size)

        no_repeat_proc_2_gram = EncoderNoRepeatNGramLogitsProcessor(2, encoder_input_ids=encoder_input_ids)
        no_repeat_proc_3_gram = EncoderNoRepeatNGramLogitsProcessor(3, encoder_input_ids=encoder_input_ids)

        filtered_scores_2_gram = no_repeat_proc_2_gram(input_ids, scores.clone())
        filtered_scores_3_gram = no_repeat_proc_3_gram(input_ids, scores.clone())

        # 2gram
        # Batch 1
        #   - Beam 1: tokens (1, 2) forbidden
        #   - Beam 2: tokens (1) forbidden
        # Batch 2
        #   - Beam 1: tokens (0, 2) forbidden
        #   - Beam 2: tokens (1) forbidden
        self.assertListEqual(
            mint.isinf(filtered_scores_2_gram).tolist(),
            [[False, True, True], [False, True, False], [True, False, True], [False, True, False]],
        )

        # Batch 1
        #   - Beam 1: tokens (1) forbidden
        #   - Beam 2: tokens () forbidden
        # Batch 2
        #   - Beam 1: tokens (2) forbidden
        #   - Beam 2: tokens () forbidden
        self.assertListEqual(
            mint.isinf(filtered_scores_3_gram).tolist(),
            [[False, True, False], [False, False, False], [False, False, True], [False, False, False]],
        )

    def test_bias_dist_processor(self):
        vocab_size = 5
        batch_size = 2

        input_ids = ms.tensor([[0, 1, 3, 1], [0, 1, 0, 1]], dtype=ms.int64)
        positive_bias = {(1,): 100.0, (4,): 100.0}
        negative_bias = {(1, 0): -100.0, (0, 1, 2): -100.0, (1, 3, 1, 3): -100.0}
        # biases the same termination twice, to ensure we can handle overlapping terminations (it won't have an effect
        # on the test cases, though)
        negative_bias.update({(1, 3, 1, 3, 1, 3): -100.0})
        sequence_bias = {**positive_bias, **negative_bias}

        # scores = 0 to facilitate checks
        scores = mint.zeros((batch_size, vocab_size), dtype=ms.float32)

        bias_dist_proc = SequenceBiasLogitsProcessor(sequence_bias=sequence_bias)
        filtered_scores = bias_dist_proc(input_ids, scores)

        # batch 1: positive bias: tokens (1, 4); negative bias: tokens (0, 3); neutral: tokens (2)
        # batch 2: positive bias: tokens (1, 4); negative bias: tokens (0, 2); neutral: tokens (3)
        self.assertListEqual(
            filtered_scores.tolist(), [[-100.0, 100.0, 0.0, -100.0, 100.0], [-100.0, 100.0, -100.0, 0.0, 100.0]]
        )

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == filtered_scores))

    def test_hamming_diversity(self):
        vocab_size = 4
        num_beams = 2
        num_beam_groups = 2

        scores = self._get_uniform_logits(num_beams, vocab_size)
        # batch_idx = 0 -> index batch_idx * num_beam_groups -> idx = 0 * 2 = 0 -> penalises tokens 1
        # batch_idx = 1 -> index batch_idx * num_beam_groups -> idx = 1 * 2 = 2 -> penalises tokens 1
        current_tokens = ms.tensor([0, 3, 1, 2], dtype=ms.int64)

        diversity_logits_processor = HammingDiversityLogitsProcessor(
            diversity_penalty=1.0, num_beams=num_beams, num_beam_groups=num_beam_groups
        )

        processed_scores = diversity_logits_processor(None, scores, current_tokens, 1)

        self.assertTrue(mint.allclose(processed_scores[0], ms.tensor([-0.7500, 0.2500, 0.2500, 0.2500]), atol=1e-3))
        self.assertTrue(mint.allclose(processed_scores[1], ms.tensor([0.2500, -0.7500, 0.2500, 0.2500]), atol=1e-3))

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == processed_scores))

    def test_forced_bos_token_logits_processor(self):
        vocab_size = 20
        batch_size = 4
        bos_token_id = 0

        logits_processor = ForcedBOSTokenLogitsProcessor(bos_token_id=bos_token_id)

        # check that all scores are -inf except the bos_token_id score
        input_ids = ids_tensor((batch_size, 1), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        processed_scores = logits_processor(input_ids, scores)
        self.assertTrue(mint.isneginf(processed_scores[:, bos_token_id + 1 :]).all())
        # score for bos_token_id should be zero
        self.assertListEqual(processed_scores[:, bos_token_id].tolist(), 4 * [0])

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == processed_scores))

        # check that bos_token_id is not forced if current length is greater than 1
        input_ids = ids_tensor((batch_size, 4), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        processed_scores = logits_processor(input_ids, scores)
        self.assertFalse(mint.isinf(processed_scores).any())

    def test_forced_eos_token_logits_processor(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0
        max_length = 5

        logits_processor = ForcedEOSTokenLogitsProcessor(max_length=max_length, eos_token_id=eos_token_id)

        # check that all scores are -inf except the eos_token_id when max_length-1 is reached
        input_ids = ids_tensor((batch_size, 4), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        processed_scores = logits_processor(input_ids, scores)
        self.assertTrue(mint.isneginf(processed_scores[:, eos_token_id + 1 :]).all())
        # score for eos_token_id should be zero
        self.assertListEqual(processed_scores[:, eos_token_id].tolist(), 4 * [0])

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == processed_scores))

        # check that eos_token_id is not forced if max_length-1 is not reached
        input_ids = ids_tensor((batch_size, 3), vocab_size=20)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        processed_scores = logits_processor(input_ids, scores)
        self.assertFalse(mint.isinf(processed_scores).any())

    def test_remove_nan_inf_logits_processor(self):
        scores = ms.tensor([[0.0, 0.7, 0.8, float("nan")], [0.1, float("inf"), 0.3, float("-inf")]])
        input_ids = ids_tensor((2, 4), vocab_size=20)

        logits_processor = InfNanRemoveLogitsProcessor()

        processed_scores = logits_processor(input_ids, scores)

        self.assertTrue(
            mint.allclose(
                processed_scores,
                ms.tensor(
                    [
                        [0.0, 0.7, 0.8, 0.0],
                        [0.1, dtype_to_max(processed_scores.dtype), 0.3, dtype_to_min(processed_scores.dtype)],
                    ],
                ),
                atol=1e-6,
            )
        )

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == processed_scores))

    def test_exponential_decay_length_penalty(self):
        vocab_size = 20
        batch_size = 4
        eos_token_id = 0

        penalty_start = 5
        penalty_factor = 1.1

        input_ids = ids_tensor((batch_size, 2), vocab_size=vocab_size)
        input_ids_seq_length = input_ids.shape[-1]

        length_decay_processor = ExponentialDecayLengthPenalty(
            exponential_decay_length_penalty=(penalty_start, penalty_factor),
            eos_token_id=eos_token_id,
            input_ids_seq_length=input_ids_seq_length,
        )

        # check that penalty is not applied before start
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_before_start = length_decay_processor(input_ids, scores)
        self.assertListEqual(scores_before_start[:, eos_token_id].tolist(), scores[:, eos_token_id].tolist())

        # check that penalty is applied after start
        input_ids = ids_tensor((batch_size, 20), vocab_size=vocab_size)
        scores = self._get_uniform_logits(batch_size, vocab_size)
        scores_after_start = length_decay_processor(input_ids, scores)
        self.assertTrue(mint.gt(scores_after_start[:, eos_token_id], scores[:, eos_token_id]).all())

        # check the penalty increases negative scores
        input_ids = ids_tensor((batch_size, 20), vocab_size=vocab_size)
        scores = mint.neg(self._get_uniform_logits(batch_size, vocab_size))
        scores_after_start = length_decay_processor(input_ids, scores)
        self.assertTrue(mint.gt(scores_after_start[:, eos_token_id], scores[:, eos_token_id]).all())

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == scores_after_start))

    def test_normalization(self):
        input_ids = None

        scores = ms.tensor([[-23.18, -29.96, -43.54, 47.77], [-33.58, -26.87, -32.96, 22.51]], dtype=ms.float32)

        logit_normalization = LogitNormalization()
        normalized_scores = logit_normalization(input_ids, scores).exp()

        ones = mint.ones(scores.shape[0], dtype=ms.float32)
        self.assertTrue(normalized_scores.sum(dim=-1).allclose(ones))

        self.assertTrue(normalized_scores.allclose(mint.softmax(scores, dim=-1)))

        # processor should not change logits in-place
        self.assertFalse(mint.all(scores == normalized_scores))

    def test_classifier_free_guidance(self):
        class Namespace(dict):
            pass

        logits_uncond = ms.tensor([[[1.0, 0, 1.5]]])
        logits_cond = ms.tensor([[[1.0, 1.0, 1.0]]])

        def dummy_model(input_ids, attention_mask, use_cache=True, past_key_values=None):
            out = Namespace()
            out.logits = logits_uncond
            out.past_key_values = None
            return out

        def lsm(x):
            return mint.nn.functional.log_softmax(x, dim=-1)

        # explicit unconditional prompt + attention mask
        input_ids = ms.tensor([[0]])
        cfg = UnbatchedClassifierFreeGuidanceLogitsProcessor(
            1.5, dummy_model, input_ids, mint.ones_like(input_ids, dtype=ms.int64)
        )
        out = cfg(input_ids, logits_cond)[0, -1]

        res = (lsm(logits_uncond) + 1.5 * (lsm(logits_cond) - lsm(logits_uncond)))[0, -1]

        self.assertAlmostEqual(out[0].item(), res[0].item())
        self.assertAlmostEqual(out[1].item(), res[1].item())
        self.assertAlmostEqual(out[2].item(), res[2].item())

        # explicit unconditional prompt
        input_ids = ms.tensor([[0]])
        cfg = UnbatchedClassifierFreeGuidanceLogitsProcessor(1.5, dummy_model, input_ids)
        out = cfg(input_ids, logits_cond)[0, -1]

        res = (lsm(logits_uncond) + 1.5 * (lsm(logits_cond) - lsm(logits_uncond)))[0, -1]

        self.assertAlmostEqual(out[0].item(), res[0].item())
        self.assertAlmostEqual(out[1].item(), res[1].item())
        self.assertAlmostEqual(out[2].item(), res[2].item())

        # all implicit
        input_ids = ms.tensor([[0]])
        cfg = UnbatchedClassifierFreeGuidanceLogitsProcessor(1.5, dummy_model)
        out = cfg(input_ids, logits_cond)[0, -1]

        res = (lsm(logits_uncond) + 1.5 * (lsm(logits_cond) - lsm(logits_uncond)))[0, -1]

        self.assertAlmostEqual(out[0].item(), res[0].item())
        self.assertAlmostEqual(out[1].item(), res[1].item())
        self.assertAlmostEqual(out[2].item(), res[2].item())

    def test_early_stop_processor(self):
        input_ids = None
        eos_token_id = 2
        min_eos_p = 0.1  # some small float

        scores = self._get_uniform_logits(2, 4)
        scores[0][eos_token_id] = -6  # less than log(min_eos_p)

        esp = BarkEosPrioritizerLogitsProcessor(eos_token_id=eos_token_id, min_eos_p=min_eos_p)
        actual_scores = esp(input_ids, scores)
        expected_scores_list = [
            scores[0].tolist(),
            [float("-inf"), float("-inf"), scores[0][0], float("-inf")],
        ]
        self.assertListEqual(actual_scores.tolist(), expected_scores_list)

    def test_early_stop_processor_multi_eos(self):
        input_ids = None
        eos_token_id = [2, 3]
        min_eos_p = 0.1  # some small float

        scores = self._get_uniform_logits(2, 4)
        scores[0][eos_token_id] = -6  # less than log(min_eos_p)

        esp = BarkEosPrioritizerLogitsProcessor(eos_token_id=eos_token_id, min_eos_p=min_eos_p)
        actual_scores = esp(input_ids, scores)
        expected_scores_list = [
            scores[0].tolist(),
            [float("-inf"), float("-inf"), scores[0][0], scores[0][0]],
        ]
        self.assertListEqual(actual_scores.tolist(), expected_scores_list)
