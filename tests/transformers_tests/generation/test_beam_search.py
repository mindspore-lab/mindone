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

from mindone.transformers.testing_utils import require_mindspore
from mindone.transformers.utils import is_mindspore_available
from tests.transformers_tests.test_modeling_common import floats_tensor, ids_tensor

if is_mindspore_available():
    import mindspore as ms
    from mindspore import mint

    from mindone.transformers.generation import BeamHypotheses, BeamSearchScorer


class BeamSearchTester:
    def __init__(
        self,
        parent,
        batch_size=3,
        sequence_length=10,
        vocab_size=99,
        pad_token_id=0,
        max_length=20,
        num_beams=4,
        length_penalty=2.0,
        do_early_stopping=True,
        num_beam_hyps_to_keep=2,
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.num_beams = num_beams
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        # cannot be randomly generated
        self.eos_token_id = vocab_size + 1

    def prepare_beam_scorer(self, **kwargs):
        return BeamSearchScorer(
            batch_size=kwargs.get("batch_size", self.batch_size),
            num_beams=kwargs.get("num_beams", self.num_beams),
            length_penalty=kwargs.get("length_penalty", self.length_penalty),
            do_early_stopping=kwargs.get("do_early_stopping", self.do_early_stopping),
            num_beam_hyps_to_keep=kwargs.get("num_beam_hyps_to_keep", self.num_beam_hyps_to_keep),
        )

    def prepare_inputs(self):
        input_ids = ids_tensor((self.batch_size * self.num_beams, self.sequence_length), self.vocab_size)
        next_tokens = ids_tensor((self.batch_size, 2 * self.num_beams), self.vocab_size)
        next_indices = ids_tensor((self.batch_size, 2 * self.num_beams), self.num_beams)
        next_scores, _ = (-floats_tensor((self.batch_size, 2 * self.num_beams))).sort(descending=True)
        return (input_ids, next_tokens, next_indices, next_scores)

    def check_beam_hypotheses(self, input_ids, *args):
        # check that correct number of beam hypotheses is set in beam scorer
        beam_scorer = self.prepare_beam_scorer(do_early_stopping=True)
        beam_hyp = beam_scorer._beam_hyps[0]

        self.parent.assertEqual(len(beam_scorer._beam_hyps), self.batch_size)

        # check correct type
        self.parent.assertTrue(isinstance(beam_hyp, BeamHypotheses))

        # check that num_beams is correctly set
        self.parent.assertEqual(beam_hyp.num_beams, self.num_beams)

        # check for early stopping deactivated
        for beam_idx in range(self.num_beams):
            beam_hyp.add(input_ids[beam_idx], -10.0)

        # if early stopping True -> score does not matter
        self.parent.assertTrue(beam_hyp.is_done(-10.0, 5))

        # re-init
        beam_scorer = self.prepare_beam_scorer(do_early_stopping=False)
        beam_hyp = beam_scorer._beam_hyps[0]

        # add `num_beams + 1` beams to change `worst_score`
        for beam_idx in range(self.num_beams + 1):
            beam_hyp.add(input_ids[beam_idx], -10.0 + float(beam_idx))

        # -10.0 is removed => -9.0 is worst score
        self.parent.assertAlmostEqual(beam_hyp.worst_score, -9.0 / (self.sequence_length**beam_hyp.length_penalty))

        # -5.0 is better than worst score => should not be finished
        self.parent.assertFalse(beam_hyp.is_done(-5.0, self.sequence_length))

        # -20.0 is worse than worst score => should be finished
        self.parent.assertTrue(beam_hyp.is_done(-20.0, self.sequence_length))

    def check_beam_scorer_update(self, input_ids, next_tokens, next_indices, next_scores):
        # check too many eos tokens
        beam_scorer = self.prepare_beam_scorer()

        tokens = next_tokens.clone()
        tokens[0, :] = self.eos_token_id

        with self.parent.assertRaises(ValueError):
            beam_scorer.process(input_ids, next_scores, tokens, next_indices, eos_token_id=self.eos_token_id)

        # check all batches are done
        beam_scorer = self.prepare_beam_scorer()

        tokens = next_tokens.clone()
        tokens[:, : self.num_beams] = self.eos_token_id
        beam_indices = mint.zeros_like(input_ids) + mint.arange(input_ids.shape[-1])
        beam_indices = tuple(tuple(b) for b in beam_indices)
        beam_scorer.process(
            input_ids, next_scores, tokens, next_indices, eos_token_id=self.eos_token_id, beam_indices=beam_indices
        )
        # beam scorer should be done
        self.parent.assertTrue(beam_scorer.is_done)

        # check
        beam_scorer = self.prepare_beam_scorer()

        tokens = next_tokens.clone()
        tokens[:, 1] = self.eos_token_id
        beam_outputs = beam_scorer.process(
            input_ids, next_scores, tokens, next_indices, eos_token_id=self.eos_token_id, beam_indices=beam_indices
        )
        output_scores = beam_outputs["next_beam_scores"]
        output_tokens = beam_outputs["next_beam_tokens"]
        output_indices = beam_outputs["next_beam_indices"]

        def cut_expected_tensor(tensor):
            return mint.cat([tensor[:, :1], tensor[:, 2 : self.num_beams + 1]], dim=1).flatten()

        # check all outptus
        # cut out id of eos token and take best `num_beams` outputs
        expected_output_tokens = cut_expected_tensor(tokens)
        expected_output_scores = cut_expected_tensor(next_scores)

        # add num_beams * batch_idx
        offset = mint.div(mint.arange(self.num_beams * self.batch_size), self.num_beams, rounding_mode="floor")
        expected_output_indices = cut_expected_tensor(next_indices) + offset * self.num_beams

        self.parent.assertListEqual(expected_output_tokens.tolist(), output_tokens.tolist())
        self.parent.assertListEqual(expected_output_indices.tolist(), output_indices.tolist())
        self.parent.assertTrue(mint.allclose(expected_output_scores, output_scores, atol=1e-3))

        # make sure ids of eos token are correctly saved in beam_hyps of beam scorer
        expected_beam_indices = list(range(10))
        for batch_idx in range(self.batch_size):
            correct_idx = batch_idx * self.num_beams + next_indices[batch_idx, 1]
            self.parent.assertListEqual(
                input_ids[correct_idx].tolist(), beam_scorer._beam_hyps[batch_idx].beams[0][1].tolist()
            )
            self.parent.assertListEqual(
                expected_beam_indices + [correct_idx],
                ms.tensor(beam_scorer._beam_hyps[batch_idx].beams[0][2]).tolist(),
            )

    def check_beam_scores_finalize(self, input_ids, next_tokens, next_indices, next_scores):
        # max_length should be only one more than current input_ids to check that eos is correctly appended
        max_length = self.sequence_length + 1
        beam_scorer = self.prepare_beam_scorer(num_beam_hyps_to_keep=1, length_penalty=1.0, do_early_stopping=False)

        # update beams and append to input_ids
        tokens = next_tokens.clone()
        # first batch, first output has to finish with eos token id since scores are correctly sorted
        tokens[0, 0] = self.eos_token_id
        # make sure corresponding score is as good as possible to surely be picked first
        next_scores[0, 0] = 0.0
        beam_outputs = beam_scorer.process(input_ids, next_scores, tokens, next_indices, eos_token_id=self.eos_token_id)
        output_scores = beam_outputs["next_beam_scores"]
        output_tokens = beam_outputs["next_beam_tokens"]
        output_indices = beam_outputs["next_beam_indices"]

        input_ids = mint.cat([input_ids[output_indices, :], output_tokens.unsqueeze(-1)], dim=-1)

        # finalize
        beam_indices = mint.zeros_like(input_ids) + mint.arange(input_ids.shape[-1])
        beam_indices = tuple(tuple(b) for b in beam_indices)
        sequence_output = beam_scorer.finalize(
            input_ids,
            output_scores,
            output_tokens,
            output_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=max_length,
            beam_indices=beam_indices,
        )

        sequences = sequence_output["sequences"]
        sequence_scores = sequence_output["sequence_scores"]

        # since `num_beam_hyps_to_keep` = 1 => only return `batch_size` x `max_length`
        self.parent.assertListEqual(list(sequences.shape), [self.batch_size, max_length])
        self.parent.assertListEqual(list(sequence_scores.shape), [self.batch_size])

        # check sequence_scores
        self.parent.assertFalse((sequence_scores > 0).any().item())

        # first batch has to finish with eos_token
        self.parent.assertEqual(sequences[0, -1].item(), self.eos_token_id)

        # other batches cannot finish with eos token
        self.parent.assertNotEqual(sequences[1, -1].item(), self.eos_token_id)
        self.parent.assertNotEqual(sequences[2, -1].item(), self.eos_token_id)

        # now test that if `num_beam_hyps_to_keep` is 3 => all beams are returned
        beam_scorer.num_beam_hyps_to_keep = self.num_beams
        sequence_output = beam_scorer.finalize(
            input_ids,
            output_scores,
            output_tokens,
            output_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            max_length=max_length,
            beam_indices=beam_indices,
        )
        sequences = sequence_output["sequences"]
        sequence_scores = sequence_output["sequence_scores"]

        self.parent.assertListEqual(list(sequences.shape), [self.num_beams * self.batch_size, max_length])
        self.parent.assertListEqual(list(sequence_scores.shape), [self.num_beams * self.batch_size])


@require_mindspore
class BeamSearchTest(unittest.TestCase):
    def setUp(self):
        self.beam_search_tester = BeamSearchTester(self)

    def test_beam_hypotheses(self):
        inputs = self.beam_search_tester.prepare_inputs()
        self.beam_search_tester.check_beam_hypotheses(*inputs)

    def test_beam_scorer_update(self):
        inputs = self.beam_search_tester.prepare_inputs()
        self.beam_search_tester.check_beam_scorer_update(*inputs)
