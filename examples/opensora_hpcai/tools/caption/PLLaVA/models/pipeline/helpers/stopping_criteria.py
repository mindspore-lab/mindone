import abc
import logging
from typing import List, Optional, Union

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor

logger = logging.getLogger(__name__)


__all__ = ["MaxLengthCriteria", "EosTokenCriteria", "StoppingCriteriaList"]


class StoppingCriteria(abc.ABC):
    @abc.abstractmethod
    def __call__(self, input_ids: Tensor) -> Tensor:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int) -> None:
        self.max_length = max_length

    def __call__(self, input_ids: Tensor) -> Tensor:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        return ops.full((input_ids.shape[0],), is_done, dtype=ms.bool_)


class EosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: Union[int, List[int], Tensor]) -> None:
        if not isinstance(eos_token_id, Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = Tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: Tensor) -> Tensor:
        is_done = ms.numpy.isin(input_ids[:, -1], self.eos_token_id)
        return is_done


class StoppingCriteriaList(list):
    def __call__(self, input_ids: Tensor) -> Tensor:
        is_done = ops.full((input_ids.shape[0],), False, dtype=ms.bool_)
        for criteria in self:
            is_done = ops.logical_or(is_done, criteria(input_ids))
        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None
