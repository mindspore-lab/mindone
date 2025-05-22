import time
from abc import ABC
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from transformers.utils import add_start_docstrings, logging

import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops

logger = logging.get_logger(__name__)
# We maintain a module-level cache of the embedding vectors for the stop string criterion
# because they are slow to compute
STOP_STRING_EMBEDDING_CACHE = OrderedDict()


STOPPING_CRITERIA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`Union[ms.Tensor, numpy.ndarray]` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        scores (`Union[ms.Tensor, numpy.ndarray]` of shape `(batch_size, config.vocab_size)`):
            Prediction scores of a language modeling head. These can be scores for each vocabulary token before SoftMax
            or scores for each vocabulary token after SoftMax. If this stopping criteria depends on the `scores` input,
            make sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`.
        kwargs (`Dict[str, Any]`, *optional*):
            Additional stopping criteria specific kwargs.

    Return:
        `Union[ms.Tensor, numpy.ndarray]`. (`Union[ms.Tensor, numpy.ndarray]` of shape `(batch_size, 1)`), where `True` indicates we stop generation
            for a particular row, `True` indicates we should continue.

"""


class StoppingCriteria(ABC):
    """Abstract base class for all stopping criteria that can be applied during generation.

    If your stopping criteria depends on the `scores` input, make sure you pass `return_dict_in_generate=True,
    output_scores=True` to `generate`.
    """

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: ms.Tensor, scores: ms.Tensor, **kwargs) -> ms.Tensor:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generated number of tokens exceeds `max_length`. Keep
    in mind for decoder-only type of transformers, this will include the initial prompted tokens.

    Args:
        max_length (`int`):
            The maximum length that the output sequence can have in number of tokens.
        max_position_embeddings (`int`, *optional*):
            The maximum model length, as defined by the model's `config.max_position_embeddings` attribute.
    """

    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray], **kwargs
    ) -> Union[ms.Tensor, np.ndarray]:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        if self.max_position_embeddings is not None and not is_done and cur_len >= self.max_position_embeddings:
            logger.warning_once(
                "This is a friendly reminder - the current text generation call will exceed the model's predefined "
                f"maximum length ({self.max_position_embeddings}). Depending on the model, you may observe "
                "exceptions, performance degradation, or nothing at all."
            )

        if isinstance(input_ids, ms.Tensor):
            return ops.full((input_ids.shape[0],), is_done, dtype=ms.bool_)
        elif isinstance(input_ids, np.ndarray):
            return np.full((input_ids.shape[0],), is_done, dtype=np.bool_)
        else:
            raise NotImplementedError


class MaxTimeCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the full generation exceeds some amount of time. By default, the
    time will start being counted when you initialize this function. You can override this by passing an
    `initial_time`.

    Args:
        max_time (`float`):
            The maximum allowed time in seconds for the generation.
        initial_time (`float`, *optional*, defaults to `time.time()`):
            The start of the generation allowed time.
    """

    def __init__(self, max_time: float, initial_timestamp: Optional[float] = None):
        self.max_time = max_time
        self.initial_timestamp = time.time() if initial_timestamp is None else initial_timestamp

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray], **kwargs
    ) -> Union[ms.Tensor, np.ndarray]:
        is_done = time.time() - self.initial_timestamp > self.max_time

        if isinstance(input_ids, ms.Tensor):
            return ops.full((input_ids.shape[0],), is_done, dtype=ms.bool_)
        elif isinstance(input_ids, np.ndarray):
            return np.full((input_ids.shape[0],), is_done, dtype=np.bool_)
        else:
            raise NotImplementedError


class EosTokenCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever the "end-of-sequence" token is generated.
    By default, it uses the `model.generation_config.eos_token_id`.

    Args:
        eos_token_id (`Union[int, List[int], ms.Tensor]`):
            The id(s) of the *end-of-sequence* token.
    """

    def __init__(self, eos_token_id: Union[int, List[int], ms.Tensor]):
        # to list
        if not isinstance(eos_token_id, ms.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            elif isinstance(eos_token_id, np.ndarray):
                eos_token_id = eos_token_id.tolist()
        else:
            eos_token_id = eos_token_id.asnumpy().tolist()

        self.eos_token_id = eos_token_id

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray], **kwargs
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(input_ids, ms.Tensor):
            is_done = mnp.isin(input_ids[:, -1], self.eos_token_id)
        elif isinstance(input_ids, np.ndarray):
            is_done = np.isin(input_ids[:, -1], self.eos_token_id)
        else:
            raise NotImplementedError

        return is_done


class StoppingCriteriaList(list):
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: Union[ms.Tensor, np.ndarray], scores: Union[ms.Tensor, np.ndarray], **kwargs
    ) -> Union[ms.Tensor, np.ndarray]:
        if isinstance(input_ids, ms.Tensor):
            is_done = ops.full((input_ids.shape[0],), False, dtype=ms.bool_)
            for criteria in self:
                is_done = ops.logical_or(is_done, criteria(input_ids, scores, **kwargs))
        elif isinstance(input_ids, np.ndarray):
            is_done = np.full((input_ids.shape[0],), False, dtype=np.bool_)
            for criteria in self:
                is_done = np.logical_or(is_done, criteria(input_ids, scores, **kwargs))
        else:
            raise NotImplementedError

        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None
