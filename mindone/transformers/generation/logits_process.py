import inspect
from typing import Callable, List, Optional, Union

import numpy as np
from transformers.utils import add_start_docstrings
from transformers.utils.logging import get_logger

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import mint, ops

from mindone.transformers.mindspore_adapter.utils import dtype_to_min

from ..mindspore_utils import isin_mps_friendly

INF = 1e5


logger = get_logger(__name__)


LOGITS_PROCESSOR_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`ms.Tensor or numpy.ndarray` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
        scores (`ms.Tensor or numpy.ndarray` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be logits for each vocabulary when not using beam
            search or log softmax for each vocabulary token when using beam search

    Return:
        `ms.Tensor or numpy.ndarray` of shape `(batch_size, config.vocab_size)`: The processed prediction scores.

"""


class LogitsProcessor:
    """Abstract base class for all logit processors that can be applied during generation."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] to subsequently process a `scores` input tensor.
    This class inherits from list and adds a specific *__call__* method to apply each [`LogitsProcessor`] to the
    inputs.
    """

    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray], **kwargs
    ) -> Union[ms.Tensor, np.ndarray]:
        r"""
        Args:
            input_ids (`Union[ms.Tensor, np.ndarray]` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`Union[ms.Tensor, np.ndarray]` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `Union[ms.Tensor, np.ndarray]` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)

        return scores


class MinLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length by setting EOS probability to 0. Note that, for decoder-only models
    like most LLMs, the length includes the prompt.

    Args:
        min_length (`int`):
            The minimum length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int], ms.Tensor, np.ndarray]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, min_length: int, eos_token_id: Union[int, List[int], ms.Tensor, np.ndarray], **ignore):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        # to list
        if not isinstance(eos_token_id, ms.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            elif isinstance(eos_token_id, np.ndarray):
                eos_token_id = eos_token_id.tolist()
        else:
            eos_token_id = eos_token_id.asnumpy().tolist()

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(scores, ms.Tensor):
            vocab_tensor = ops.arange(0, scores.shape[-1])
            eos_token_mask = mnp.isin(vocab_tensor, self.eos_token_id)
            scores_processed = scores[:]
            if input_ids.shape[-1] < self.min_length:
                scores_processed = ops.where(eos_token_mask, -INF, scores)
        elif isinstance(scores, np.ndarray):
            vocab_tensor = np.arange(0, scores.shape[-1])
            eos_token_mask = np.isin(vocab_tensor, self.eos_token_id)
            scores_processed = scores[:]
            if input_ids.shape[-1] < self.min_length:
                scores_processed = ops.where(eos_token_mask, -INF, scores)
        else:
            raise NotImplementedError

        return scores_processed


class MinNewTokensLengthLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] enforcing a min-length of new tokens by setting EOS (End-Of-Sequence) token probability to 0.
    Contrarily to [`MinLengthLogitsProcessor`], this processor ignores the prompt.

    Args:
        prompt_length_to_skip (`int`):
            The input tokens length. Not a valid argument when used with `generate` as it will automatically assign the
            input length.
        min_new_tokens (`int`):
            The minimum *new* tokens length below which the score of `eos_token_id` is set to `-float("Inf")`.
        eos_token_id (`Union[int, List[int], ms.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(
        self, prompt_length_to_skip: int, min_new_tokens: int, eos_token_id: Union[int, List[int], ms.Tensor], **ignore
    ):
        for arg_name, arg_value in [
            ("prompt_length_to_skip", prompt_length_to_skip),
            ("min_new_tokens", min_new_tokens),
        ]:
            if not isinstance(arg_value, int) or arg_value < 0:
                raise ValueError(f"`{arg_name}` has to be a positive integer, but is {arg_value}")

        # to list
        if not isinstance(eos_token_id, ms.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            elif isinstance(eos_token_id, np.ndarray):
                eos_token_id = eos_token_id.tolist()
        else:
            eos_token_id = eos_token_id.asnumpy().tolist()

        self.prompt_length_to_skip = prompt_length_to_skip
        self.min_new_tokens = min_new_tokens
        self.eos_token_id = eos_token_id

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(scores, ms.Tensor):
            new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
            scores_processed = scores[:]
            vocab_tensor = ops.arange(0, scores.shape[-1])
            eos_token_mask = mnp.isin(vocab_tensor, self.eos_token_id)
            if new_tokens_length < self.min_new_tokens:
                scores_processed = ops.where(eos_token_mask, -INF, scores)
        elif isinstance(scores, np.ndarray):
            new_tokens_length = input_ids.shape[-1] - self.prompt_length_to_skip
            scores_processed = scores[:]
            vocab_tensor = np.arange(0, scores.shape[-1])
            eos_token_mask = np.isin(vocab_tensor, self.eos_token_id)
            if new_tokens_length < self.min_new_tokens:
                scores_processed = np.where(eos_token_mask, -INF, scores)
        else:
            raise NotImplementedError

        return scores_processed


class TemperatureLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for temperature (exponential scaling output probability distribution), which effectively means
    that it can control the randomness of the predicted tokens. Often used together with [`TopPLogitsWarper`] and
    [`TopKLogitsWarper`].

    <Tip>

    Make sure that `do_sample=True` is included in the `generate` arguments otherwise the temperature value won't have
    any effect.

    </Tip>

    Args:
        temperature (`float`):
            Strictly positive float value used to modulate the logits distribution. A value smaller than `1` decreases
            randomness (and vice versa), with `0` being equivalent to shifting all probability mass to the most likely
            token.
    """

    def __init__(self, temperature: float):
        if not isinstance(temperature, float) or not (temperature > 0):
            except_msg = (
                f"`temperature` (={temperature}) has to be a strictly positive float, otherwise your next token "
                "scores will be invalid."
            )
            if isinstance(temperature, float) and temperature == 0.0:
                except_msg += " If you're looking for greedy decoding strategies, set `do_sample=False`."
            raise ValueError(except_msg)

        self.temperature = temperature

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        scores_processed = scores / self.temperature
        return scores_processed


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that prevents the repetition of previous tokens through a penalty. This penalty is applied at
    most once per token. Note that, for decoder-only models like most LLMs, the considered tokens include the prompt.

    In the original [paper](https://arxiv.org/pdf/1909.05858.pdf), the authors suggest the use of a penalty of around
    1.2 to achieve a good balance between truthful generation and lack of repetition. To penalize and reduce
    repetition, use `penalty` values above 1.0, where a higher value penalizes more strongly. To reward and encourage
    repetition, use `penalty` values between 0.0 and 1.0, where a lower value rewards more strongly.

    Args:
        penalty (`float`):
            The parameter for repetition penalty. 1.0 means no penalty. Above 1.0 penalizes previously generated
            tokens. Between 0.0 and 1.0 rewards previously generated tokens.
    """

    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor) -> ms.Tensor:
        score = mint.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the token probabilities
        score = mint.where(score < 0, score * self.penalty, score / self.penalty)

        scores_processed = scores.scatter(1, input_ids, score)
        return scores_processed


class TopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs top-p, i.e. restricting to top tokens summing to prob_cut_off <= prob_cut_off. Often
    used together with [`TemperatureLogitsWarper`] and [`TopKLogitsWarper`].

    Args:
        top_p (`float`):
            If set to < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or
            higher are kept for generation.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.

    Examples:

    ```python
    >>> from transformers import AutoTokenizer, set_seed
    >>> from mindone.transformers.models.llama import LlamaForCausalLM

    >>> set_seed(1)
    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

    >>> inputs = tokenizer("A sequence: 1, 2", return_tensors="np")

    >>> # With sampling, the output is unexpected -- sometimes too unexpected.
    >>> outputs = model.generate(**inputs, do_sample=True)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3 | < 4 (left-hand pointer) ;
    <BLANKLINE>
    <BLANKLINE>

    >>> # With `top_p` sampling, the output gets restricted to high-probability tokens.
    >>> # Pro tip: In practice, LLMs use `top_p` in the 0.9-0.95 range.
    >>> outputs = model.generate(**inputs, do_sample=True, top_p=0.1)
    >>> print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
    A sequence: 1, 2, 3, 4, 5, 6, 7, 8, 9
    ```
    """

    def __init__(self, top_p: float, filter_value: float = None, min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(scores, ms.Tensor):
            filter_value = self.filter_value if self.filter_value is not None else dtype_to_min(scores.dtype)

            sorted_logits, sorted_indices = ops.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(axis=-1).cumsum(axis=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

            # scatter sorted tensors to original indexing
            # indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            sorted_indices_to_remove = sorted_indices_to_remove.astype(ms.int32)
            indices_to_remove = ops.tensor_scatter_elements(
                sorted_indices_to_remove, indices=sorted_indices, updates=sorted_indices_to_remove, axis=1
            )

            scores_processed = scores.masked_fill(indices_to_remove.astype(ms.bool_), filter_value)
        elif isinstance(scores, np.ndarray):
            raise NotImplementedError
        else:
            raise NotImplementedError

        return scores_processed


class TopKLogitsWarper(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that performs top-k, i.e. restricting to the k highest probability elements. Often used
    together with [`TemperatureLogitsWarper`] and [`TopPLogitsWarper`].

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to -inf):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_k: int, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(scores, ms.Tensor):
            top_k = min(self.top_k, scores.shape[-1])  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = scores < mint.topk(scores, top_k)[0][..., -1, None]
            scores_processed = scores.masked_fill(indices_to_remove, self.filter_value)
        elif isinstance(scores, np.ndarray):
            raise NotImplementedError
        else:
            raise NotImplementedError
        return scores_processed


class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    r"""
    [`LogitsProcessor`] that enforces constrained generation and is useful for prefix-conditioned constrained
    generation. See [Autoregressive Entity Retrieval](https://arxiv.org/abs/2010.00904) for more information.

    Args:
        prefix_allowed_tokens_fn (`Callable[[int, ms.Tensor], List[int]]`):
            This function constraints the beam search to allowed tokens only at each step. This function takes 2
            arguments `inputs_ids` and the batch ID `batch_id`. It has to return a list with the allowed tokens for the
            next generation step conditioned on the previously generated tokens `inputs_ids` and the batch ID
            `batch_id`.
    """

    def __init__(self, prefix_allowed_tokens_fn: Callable[[int, ms.Tensor], List[int]], num_beams: int):
        self._prefix_allowed_tokens_fn = prefix_allowed_tokens_fn
        self._num_beams = num_beams

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(input_ids, ms.Tensor):
            assert isinstance(scores, ms.Tensor)
            mask = ops.full_like(scores, -INF)
            for batch_id, beam_sent in enumerate(input_ids.view(-1, self._num_beams, input_ids.shape[-1])):
                for beam_id, sent in enumerate(beam_sent):
                    prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                    if len(prefix_allowed_tokens) == 0:
                        raise ValueError(
                            f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                            f"This means that the constraint is unsatisfiable. Please check your implementation"
                            f"of `prefix_allowed_tokens_fn` "
                        )
                    mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0
        elif isinstance(input_ids, np.ndarray):
            assert isinstance(scores, np.ndarray)
            mask = np.full_like(scores, -INF)
            for batch_id, beam_sent in enumerate(input_ids.reshape((-1, self._num_beams, input_ids.shape[-1]))):
                for beam_id, sent in enumerate(beam_sent):
                    prefix_allowed_tokens = self._prefix_allowed_tokens_fn(batch_id, sent)
                    if len(prefix_allowed_tokens) == 0:
                        raise ValueError(
                            f"`prefix_allowed_tokens_fn` returned an empty list for batch ID {batch_id}."
                            f"This means that the constraint is unsatisfiable. Please check your implementation"
                            f"of `prefix_allowed_tokens_fn` "
                        )
                    mask[batch_id * self._num_beams + beam_id, prefix_allowed_tokens] = 0

        else:
            raise NotImplementedError

        scores_processed = scores + mask
        return scores_processed


class LogitNormalization(LogitsProcessor):
    r"""
    [`LogitsProcessor`] for normalizing the scores using log-softmax. It's important to normalize
    the scores during beam search, after applying the logits processors or warpers, since the search algorithm used in
    this library doesn't do it (it only does it before, but they may need re-normalization) but it still supposes that
    the scores are normalized when comparing the hypotheses.

    """

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray]
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(scores, ms.Tensor):
            scores_processed = ops.log_softmax(scores.to(ms.float32), axis=-1).to(scores.dtype)
        elif isinstance(scores, np.ndarray):
            exp_scores = np.exp(scores)
            scores_processed = np.log(exp_scores / exp_scores.sum(-1))
        else:
            raise NotImplementedError

        return scores_processed


class UnbatchedClassifierFreeGuidanceLogitsProcessor(LogitsProcessor):
    r"""
    Logits processor for Classifier-Free Guidance (CFG). The processors computes a weighted average across scores
    from prompt conditional and prompt unconditional (or negative) logits, parameterized by the `guidance_scale`.
    The unconditional scores are computed internally by prompting `model` with the `unconditional_ids` branch.

    See [the paper](https://arxiv.org/abs/2306.17806) for more information.

    Args:
        guidance_scale (`float`):
            The guidance scale for classifier free guidance (CFG). CFG is enabled by setting `guidance_scale != 1`.
            Higher guidance scale encourages the model to generate samples that are more closely linked to the input
            prompt, usually at the expense of poorer quality. A value smaller than 1 has the opposite effect, while
            making the negative prompt provided with negative_prompt_ids (if any) act as a positive prompt.
        model (`MSPreTrainedModel`):
            The model computing the unconditional scores. Supposedly the same as the one computing the conditional
            scores. Both models must use the same tokenizer.
        unconditional_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary for the unconditional branch. If unset, will default to
            the last token of the prompt.
        unconditional_attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Attention mask for unconditional_ids.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether to cache key/values during the negative prompt forward pass.

    Examples:

    ```python

    >>> from mindone.transformers import AutoTokenizer, AutoModelForCausalLM
    >>> from mindspore import Tensor

    >>> model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    >>> inputs = tokenizer(["Today, a dragon flew over Paris, France,"], return_tensors="np")
    >>> out = model.generate(Tensor(inputs["input_ids"]), guidance_scale=1.5)
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 50 people and injuring more than 100'

    >>> # with a negative prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(Tensor(inputs["input_ids"]), guidance_scale=2, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    'Today, a dragon flew over Paris, France, killing at least 130 people. French media reported that'

    >>> # with a positive prompt
    >>> neg_inputs = tokenizer(["A very happy event happened,"], return_tensors="pt")
    >>> out = model.generate(Tensor(inputs["input_ids"]), guidance_scale=0, negative_prompt_ids=neg_inputs["input_ids"])
    >>> tokenizer.batch_decode(out, skip_special_tokens=True)[0]
    "Today, a dragon flew over Paris, France, and I'm very happy to be here. I"
    """

    def __init__(
        self,
        guidance_scale: float,
        model,
        unconditional_ids: Optional[ms.Tensor] = None,
        unconditional_attention_mask: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = True,
    ):
        self.guidance_scale = guidance_scale
        self.model = model
        self.unconditional_context = {
            "input_ids": unconditional_ids,
            "attention_mask": unconditional_attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
            "first_pass": True,
        }

    def get_unconditional_logits(self, input_ids):
        if self.unconditional_context["first_pass"]:
            if self.unconditional_context["input_ids"] is None:
                self.unconditional_context["input_ids"] = input_ids[:, -1:]
            if self.unconditional_context["attention_mask"] is None:
                self.unconditional_context["attention_mask"] = ops.ones_like(
                    self.unconditional_context["input_ids"], dtype=ms.int32
                )
            input_ids = self.unconditional_context["input_ids"]
            attention_mask = self.unconditional_context["attention_mask"]
            self.unconditional_context["first_pass"] = False
        else:
            attention_mask = mint.cat(
                [
                    self.unconditional_context["attention_mask"],
                    ops.ones_like(input_ids[:, -1:], dtype=ms.int32),
                ],
                dim=1,
            )
            if not self.unconditional_context["use_cache"]:
                input_ids = mint.cat([self.unconditional_context["input_ids"], input_ids[:, -1:]], dim=1)
            else:
                input_ids = input_ids[:, -1:]
            self.unconditional_context["input_ids"] = input_ids
            self.unconditional_context["attention_mask"] = attention_mask

        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=self.unconditional_context["use_cache"],
            past_key_values=self.unconditional_context["past_key_values"],
        )
        self.unconditional_context["past_key_values"] = out.get("past_key_values", None)

        return out.logits

    def __call__(self, input_ids, scores):
        scores = mint.nn.functional.log_softmax(scores, dim=-1)
        if self.guidance_scale == 1:
            return scores

        logits = self.get_unconditional_logits(input_ids)

        unconditional_logits = mint.nn.functional.log_softmax(logits[:, -1], dim=-1)
        scores_processed = self.guidance_scale * (scores - unconditional_logits) + unconditional_logits
        return scores_processed


class SuppressTokensAtBeginLogitsProcessor(LogitsProcessor):
    r"""
    [`SuppressTokensAtBeginLogitsProcessor`] supresses a list of tokens as soon as the `generate` function starts
    generating using `begin_index` tokens. This should ensure that the tokens defined by `begin_suppress_tokens` are
    not generated at the beginning. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).



    >>> from mindone.transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has `begin_suppress_tokens` set by default (= `[220, 50256]`). 50256 is the EOS token, so this means
    >>> # it can't generate and EOS token in the first iteration, but it can in the others.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[0][0, 50256])
    tensor(-inf)
    >>> print(outputs.scores[-1][0, 50256])  # in other places we can see some probability mass for EOS
    tensor(29.9010)

    >>> # If we disable `begin_suppress_tokens`, we can generate EOS in the first iteration.
    >>> outputs = model.generate(
    ...     **inputs, return_dict_in_generate=True, output_scores=True, begin_suppress_tokens=None
    ... )
    >>> print(outputs.scores[0][0, 50256])
    tensor(11.2027)
    ```
    """

    def __init__(self, begin_suppress_tokens, begin_index):
        self.begin_suppress_tokens = ms.tensor(list(begin_suppress_tokens))
        self.begin_index = begin_index

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor) -> ms.Tensor:
        vocab_tensor = mint.arange(scores.shape[-1])
        suppress_token_mask = isin_mps_friendly(vocab_tensor, self.begin_suppress_tokens)
        scores_processed = scores
        if input_ids.shape[-1] == self.begin_index:
            scores_processed = mint.where(suppress_token_mask, -float("inf"), scores)

        return scores_processed


class SuppressTokensLogitsProcessor(LogitsProcessor):
    r"""
    This processor can be used to suppress a list of tokens. The processor will set their log probs to `-inf` so
    that they are not generated. Originally created for
    [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

    Examples:

    ```python
    >>> from mindone.transformers import AutoProcessor, WhisperForConditionalGeneration
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")

    >>> # Whisper has a long list of suppressed tokens. For instance, in this case, the token 1 is suppressed by default.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    >>> print(outputs.scores[1][0, 1])  # 1 (and not 0) is the first freely generated token
    tensor(-inf)

    >>> # If we disable `suppress_tokens`, we can generate it.
    >>> outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True, suppress_tokens=None)
    >>> print(outputs.scores[1][0, 1])
    tensor(6.0678)
    ```
    """

    def __init__(self, suppress_tokens):
        self.suppress_tokens = ms.tensor(list(suppress_tokens))

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor) -> ms.Tensor:
        vocab_tensor = mint.arange(scores.shape[-1])
        suppress_token_mask = isin_mps_friendly(vocab_tensor, self.suppress_tokens)
        scores = mint.where(suppress_token_mask, -float("inf"), scores)
        return scores


class WhisperNoSpeechDetection(LogitsProcessor):
    r"""This processor can be used to detect silence when using Whisper. It should take as input unprocessed logits
    to follow the original implementation"""

    def __init__(self, no_speech_token: int, begin_index: int, scores_is_logprobs: bool = False):
        self.no_speech_token = no_speech_token
        # offset between <start-of-transcription> token, <SOT>, in paper and first generated token
        # is equal to the position of the first generated token index
        self.start_of_trans_offset = begin_index

        # `self.begin_index` is a running value that is changed on the fly
        self.begin_index = begin_index
        self._no_speech_prob = [0.0]
        self.is_scores_logprobs = scores_is_logprobs

        # overwritten dynamically
        self.model = None
        self.inputs = None

    def set_model(self, model):
        self.model = model

    def set_inputs(self, inputs):
        self.inputs = {**self.model.prepare_inputs_for_generation(**inputs), **inputs}
        self.inputs["input_features"] = self.inputs.pop("inputs")

    @property
    def no_speech_prob(self):
        return self._no_speech_prob

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor) -> ms.Tensor:
        is_scores_logprobs = self.is_scores_logprobs

        if input_ids.shape[1] == self.begin_index:
            if self.start_of_trans_offset > 1:
                logits = self.model(**self.inputs).logits

                no_speech_index = self.begin_index - self.start_of_trans_offset
                no_speech_scores = logits[:, no_speech_index]
                is_scores_logprobs = False
            else:
                no_speech_scores = scores

            if is_scores_logprobs:
                probs = no_speech_scores.exp()
            else:
                probs = no_speech_scores.float().softmax(dim=-1)

            self._no_speech_prob = probs[:, self.no_speech_token]

        return scores


class WhisperTimeStampLogitsProcessor(LogitsProcessor):
    r"""

    [`LogitsProcessor`] that modifies the logits for the generation of timestamps in the transcription. When the input
    tokens are at a specific threshold, the processor sets the scores to negative infinity. The processor makes sure
    that timestamp tokens appear in pairs, by masking out the logits that would break this pairing pattern. This is
    done to maintain the consistency and structure of generated timestamps. It also ensures that when the predicted
    probability of sampling any of the timestamp token is greater than any individual non-timestamp token, those
    non-timestamp logits are set to negative infinity. This is done to ensure the generation of timestamps over other
    potential tokens.


    See [the paper](https://arxiv.org/abs/2212.04356) for more information.

    Args:
        generate_config (`GenerateConfig`):
            The generate config used to generate the output. The following parameters are required:
                eos_token_id (`int`, *optional*, defaults to 50257):
                    The id of the *end-of-sequence* token.
                no_timestamps_token_id (`int`, *optional*, defaults to 50363):
                    The id of the `"<|notimestamps|>"` token.
                max_initial_timestamp_index (`int`, *optional*, defaults to 1):
                    Used to set the maximum value of the initial timestamp. This is used to prevent the model from
                    predicting timestamps that are too far in the future.
        begin_index (`Optional`, *optional*): Token index of the first token that is generated by the model.
        _detect_timestamp_from_logprob (`bool`, *optional*): Whether timestamps can be predicted from logprobs over all timestamps.

    Examples:
    ``` python
    >>> from mindone.transformers import AutoProcessor, WhisperForConditionalGeneration, GenerationConfig
    >>> from datasets import load_dataset

    >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
    >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    >>> inputs = processor(ds[3]["audio"]["array"], return_tensors="pt")
    >>> input_features = inputs.input_features

    >>> #Displaying timestamps
    >>> generated_ids = model.generate(inputs=input_features, return_timestamps=True)
    >>> transcription = processor.batch_decode(generated_ids, decode_with_timestamps=True)[0]
    >>> print("Transcription:", transcription)
    Transcription: <|startoftranscript|><|0.00|> He has grave doubts whether Sir Frederick Layton's work is really Greek after all,
    and can<|6.44|><|6.44|> discover in it but little of rocky Ithaca.<|9.44|><|endoftext|>


    >>> #No timestamps & change EOS:
    >>> #This allows the user to select a specific token to terminate the sequence on, in this case it's the word "can"(460)
    >>> model.generation_config.eos_token_id = 460
    >>> generated_ids = model.generate(inputs=input_features,return_timestamps=False)
    >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    >>> print("Transcription:", transcription)
    Transcription:  He has grave doubts whether Sir Frederick Layton's work is really Greek after all and can

    ```
    """

    def __init__(
        self,
        generate_config,
        begin_index: Optional[int] = None,
        _detect_timestamp_from_logprob: Optional[bool] = None,
    ):  # support for the kwargs
        self.no_timestamps_token_id = generate_config.no_timestamps_token_id
        self.timestamp_begin = generate_config.no_timestamps_token_id + 1
        self.eos_token_id = generate_config.eos_token_id or generate_config.bos_token_id

        # this variable is mostly just used for testing
        self._detect_timestamp_from_logprob = (
            _detect_timestamp_from_logprob
            if _detect_timestamp_from_logprob is not None
            else getattr(generate_config, "_detect_timestamp_from_logprob", True)
        )

        num_forced_ids = (
            len(generate_config.forced_decoder_ids) if generate_config.forced_decoder_ids is not None else 0
        )
        self.begin_index = begin_index or (num_forced_ids + 1)

        self.max_initial_timestamp_index = getattr(generate_config, "max_initial_timestamp_index", None)
        # TODO(Patrick): Make sure that official models have max_initial_timestamp_index set to 50
        # self.max_initial_timestamp_index = 50

    def set_begin_index(self, begin_index):
        self.begin_index = begin_index

    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor) -> ms.Tensor:
        # suppress <|notimestamps|> which is handled by without_timestamps
        scores_processed = scores.clone()
        scores_processed[:, self.no_timestamps_token_id] = -float("inf")

        # timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
        for k in range(input_ids.shape[0]):
            sampled_tokens = input_ids[k, self.begin_index :]
            seq = list(sampled_tokens.tolist())

            last_was_timestamp = len(seq) >= 1 and seq[-1] >= self.timestamp_begin
            penultimate_was_timestamp = len(seq) < 2 or seq[-2] >= self.timestamp_begin

            if last_was_timestamp:
                if penultimate_was_timestamp:  # has to be non-timestamp
                    scores_processed[k, self.timestamp_begin :] = -float("inf")
                else:  # cannot be normal text tokens
                    scores_processed[k, : self.eos_token_id] = -float("inf")

            timestamps = sampled_tokens[sampled_tokens.ge(self.timestamp_begin)]
            if timestamps.numel() > 0:
                # `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
                # The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
                if last_was_timestamp and not penultimate_was_timestamp:
                    timestamp_last = timestamps[-1]
                else:
                    # Avoid to emit <|0.00|> again
                    timestamp_last = timestamps[-1] + 1

                scores_processed[k, self.timestamp_begin : timestamp_last] = -float("inf")

        # apply the `max_initial_timestamp` option
        if input_ids.shape[1] == self.begin_index:
            scores_processed[:, : self.timestamp_begin] = -float("inf")

            if self.max_initial_timestamp_index is not None:
                last_allowed = self.timestamp_begin + self.max_initial_timestamp_index
                scores_processed[:, last_allowed + 1 :] = -float("inf")

        # if sum of probability over timestamps is above any other token, sample timestamp
        logprobs = mint.nn.functional.log_softmax(scores_processed.float(), dim=-1)
        for k in range(input_ids.shape[0]):
            timestamp_logprob = logprobs[k, self.timestamp_begin :].logsumexp(dim=-1)
            max_text_token_logprob = logprobs[k, : self.timestamp_begin].max()
            if timestamp_logprob > max_text_token_logprob and self._detect_timestamp_from_logprob:
                scores_processed[k, : self.timestamp_begin] = -float("inf")

        return scores_processed
