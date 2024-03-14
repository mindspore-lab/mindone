import inspect
from abc import ABC
from typing import List, Optional, Union

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class LayerNorm(nn.LayerNorm):
    """Subclass LayerNorm to handle fp16."""

    def construct(self, x: Tensor):
        orig_type = x.dtype
        ret = super().construct(ops.cast(x, ms.float32))
        return ops.cast(ret, orig_type)


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * ops.sigmoid(1.702 * x)


class LogitsProcessor:
    def __call__(self, input_ids, scores):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inherting this class an be called."
        )


class StoppingCriteriaList(list):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return any(criteria(input_ids, scores) for criteria in self)

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
            elif isinstance(stopping_criterium, MaxNewTokensCriteria):
                return stopping_criterium.max_length
        return None


class StoppingCriteria(ABC):
    def __call__(self, input_ids, scores, **kwargs) -> bool:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int):
        self.max_length = max_length

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


class MaxNewTokensCriteria(StoppingCriteria):
    def __init__(self, start_length: int, max_new_tokens: int):
        self.start_length = start_length
        self.max_new_tokens = max_new_tokens
        self.max_length = start_length + max_new_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return input_ids.shape[-1] >= self.max_length


class MinLengthLogitsProcessor(LogitsProcessor):
    def __init__(self, min_length: int, eos_token_id: Union[int, List[int]]):
        if not isinstance(min_length, int) or min_length < 0:
            raise ValueError(f"`min_length` has to be a non-negative integer, but is {min_length}")

        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        if not all([isinstance(i, int) for i in eos_token_id]) or any([i < 0 for i in eos_token_id]):
            print(f"warning: `eos_token_id` has to be a list of positive integers, but is {eos_token_id}")

        self.min_length = min_length
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids, scores):
        cur_len = input_ids.shape[-1]
        if cur_len < self.min_length:
            for i in self.eos_token_id:
                scores[:, i] = -float("inf")
        return scores


class RepetitionPenaltyLogitsProcessor(LogitsProcessor):
    def __init__(self, penalty: float):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty

    def __call__(self, input_ids, scores):
        score = ops.gather_elements(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = ops.where(score < 0, score * self.penalty, score / self.penalty)
        outputs = ops.tensor_scatter_elements(scores, input_ids, score, 1)
        return outputs


class LogitsProcessorList(list):
    def __call__(self, input_ids, scores, **kwargs):
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


class LogitsWarper:
    def __call__(self, input_ids, scores):
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class TopPLogitsWarper(LogitsWarper):
    def __init__(
        self,
        top_p: float,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.top_p = top_p
        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids, scores):
        sorted_logits, sorted_indices = ops.sort(scores, descending=False)
        cumulative_probs = ops.cumsum(ops.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopKLogitsWarper(LogitsWarper):
    r"""
    [`LogitsWarper`] that performs top-k, i.e. restricting to the k highest probability elements.

    Args:
        top_k (`int`):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(
        self,
        top_k: int,
        filter_value: float = -float("Inf"),
        min_tokens_to_keep: int = 1,
    ):
        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        self.top_k = max(top_k, min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids, scores):
        top_k = min(self.top_k, scores.shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices = ops.topk(scores, top_k)[1]
        mask = ops.zeros_like(scores)
        mask = mask.scatter(1, indices, ops.ones(indices.shape))
        indices_to_remove = mask == 0
        # indices_to_remove = scores < (ops.topk(scores, top_k)[0][..., -1, None] - 1e-5)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
