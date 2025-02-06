import inspect
from typing import Callable, List, Union

import numpy as np
from transformers.utils import add_start_docstrings
from transformers.utils.logging import get_logger

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops

from mindone.transformers.mindspore_adapter.utils import dtype_to_min

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
            indices_to_remove = scores < ops.topk(scores, top_k)[0][..., -1, None]
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
