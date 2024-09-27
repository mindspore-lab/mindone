import logging
from typing import Dict, Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from .helpers import EosTokenCriteria, MaxLengthCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)


class TextGenerator:
    def __init__(
        self,
        model: nn.Cell,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: Optional[int] = None,
        max_new_tokens: Optional[int] = 100,
        min_new_tokens: Optional[int] = None,
        use_kv_cache: bool = False,
    ) -> None:
        self.model = model.set_train(False)
        for param in self.model.trainable_params():
            param.requires_grad = False

        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id
        self._max_new_tokens = max_new_tokens
        self._min_new_tokens = min_new_tokens
        self._use_kv_cache = use_kv_cache

        self._max_length: Optional[int] = None
        self._min_length: Optional[int] = None

        if not hasattr(self.model, "prepare_inputs_for_generation"):
            raise NotImplementedError(
                "A model class needs to define a `prepare_inputs_for_generation` method in order to use `.generate()`."
            )

        if self._use_kv_cache:
            self._past_key_cache_list: Optional[Tensor] = None
            self._past_value_cache_list: Optional[Tensor] = None

    def _prepare_model_inputs(
        self, bos_token_id: Optional[Tensor] = None, model_kwargs: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        input_name = "input_ids"  # support inputs id only
        model_kwargs = {k: v for k, v in model_kwargs.items() if v is not None or k != input_name}

        inputs = model_kwargs.pop(input_name, None)

        # if `inputs` is still None, try to create `input_ids` from BOS token
        inputs = self._maybe_initialize_input_ids_for_generation(inputs, bos_token_id, model_kwargs)
        return inputs, model_kwargs

    def _maybe_initialize_input_ids_for_generation(
        self,
        inputs: Optional[Tensor] = None,
        bos_token_id: Optional[Tensor] = None,
        model_kwargs: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs

        # If there is some tensor in `model_kwargs`, we can infer the batch size from it. This is helpful with
        # soft-prompting or in multimodal implementations built on top of decoder-only language models.
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, Tensor):
                batch_size = value.shape[0]
                break

        if bos_token_id is None:
            raise ValueError("`bos_token_id` has to be defined when no `input_ids` are provided.")

        return ops.ones((batch_size, 1), dtype=ms.int32) * bos_token_id

    def _prepare_attention_mask_for_generation(
        self, inputs: Tensor, pad_token_id: Optional[Tensor], eos_token_id: Optional[Tensor]
    ) -> Tensor:
        # No information for attention mask inference -> return default attention mask
        default_attention_mask = ops.ones(inputs.shape[:2], dtype=ms.int32)
        if pad_token_id is None:
            return default_attention_mask

        is_input_ids = len(inputs.shape) == 2 and inputs.dtype in [ms.int32, ms.int64]
        if not is_input_ids:
            return default_attention_mask

        is_pad_token_in_inputs = (pad_token_id is not None) and (
            ms.numpy.isin(element=inputs, test_elements=pad_token_id).any()
        )
        is_pad_token_not_equal_to_eos_token_id = (eos_token_id is None) or ~(
            ms.numpy.isin(element=eos_token_id, test_elements=pad_token_id).any()
        )
        can_infer_attention_mask = is_pad_token_in_inputs * is_pad_token_not_equal_to_eos_token_id
        attention_mask_from_padding = inputs.ne(pad_token_id).to(ms.int32)

        attention_mask = (
            attention_mask_from_padding * can_infer_attention_mask + default_attention_mask * ~can_infer_attention_mask
        )
        return attention_mask

    def _update_model_kwargs_for_generation(
        self,
        model_kwargs: Dict[str, Tensor],
        key_cache_list: Optional[Tensor] = None,
        value_cache_list: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = ops.concat(
                [attention_mask, ops.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype)], axis=-1
            )

        # update kv cache
        if key_cache_list is not None and value_cache_list is not None:
            if self._past_key_cache_list is not None and self._past_value_cache_list is not None:
                self._past_key_cache_list = ops.concat([self._past_key_cache_list, key_cache_list], axis=-2)
                self._past_value_cache_list = ops.concat([self._past_value_cache_list, value_cache_list], axis=-2)
            else:
                self._past_key_cache_list = key_cache_list
                self._past_value_cache_list = value_cache_list

            model_kwargs["past_key_cache_list"] = self._past_key_cache_list
            model_kwargs["past_value_cache_list"] = self._past_value_cache_list

        return model_kwargs

    def _get_stopping_criteria(self) -> StoppingCriteriaList:
        criteria = StoppingCriteriaList()
        if self._max_length is not None:
            criteria.append(MaxLengthCriteria(self._max_length))

        if self._eos_token_id is not None:
            criteria.append(EosTokenCriteria(eos_token_id=self._eos_token_id))
        return criteria

    def _prepare_generated_length(self, input_ids_length: int) -> None:
        """Prepared max and min length in generaion configs to avoid clashes between similar attributes"""
        if self._max_new_tokens is not None:
            self._max_length = self._max_new_tokens + input_ids_length
        if self._min_new_tokens is not None:
            self._min_length = self._min_new_tokens + input_ids_length

    def _prepare_special_tokens(self, kwargs_has_attention_mask: Optional[bool] = None):
        # Convert special tokens to tensors (if they exist either in kwargs or in self.config)
        def _tensor_or_none(token):
            if token is None or isinstance(token, Tensor):
                return token
            return Tensor(token, dtype=ms.int32)

        bos_token_id = _tensor_or_none(self._bos_token_id)
        eos_token_id = _tensor_or_none(self._eos_token_id)
        pad_token_id = _tensor_or_none(self._pad_token_id)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_id is not None and eos_token_id.ndim == 0:
            eos_token_id = eos_token_id.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_id is None and eos_token_id is not None:
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning(
                    "The attention mask and the pad token id were not set. As a consequence, you may observe "
                    "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
                )
            pad_token_id = eos_token_id[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_id} for open-end generation.")

        # we can't infer attn mask if pad token is set to be eos token in model's generation config
        if eos_token_id is not None and ms.numpy.isin(element=eos_token_id, test_elements=pad_token_id).any():
            if kwargs_has_attention_mask is not None and not kwargs_has_attention_mask:
                logger.warning(
                    "The attention mask is not set and cannot be inferred from input because pad token is same as eos token."
                    "As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` "
                    "to obtain reliable results."
                )

        # Sanity checks/warnings
        if eos_token_id is not None and (eos_token_id < 0).any():
            logger.warning(
                f"`eos_token_id` should consist of positive integers, but is {eos_token_id}. Your generation will not "
                "stop until the maximum length is reached. Depending on other flags, it may even crash."
            )

        # Update generation config with the updated special tokens tensors
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._pad_token_id = pad_token_id

    def generate(self, **model_kwargs) -> Tensor:
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # Define model inputs
        input_ids, model_kwargs = self._prepare_model_inputs(self._bos_token_id, model_kwargs)
        batch_size = input_ids.shape[0]
        self._prepare_special_tokens(kwargs_has_attention_mask)

        # decoder-only models must use left-padding for batched generation.
        # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
        if (
            self._pad_token_id is not None
            and batch_size > 1
            and len(input_ids.shape) == 2
            and ops.sum(input_ids[:, -1] == self._pad_token_id) > 0
        ):
            logger.warning(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )

        if not kwargs_has_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                input_ids, self._pad_token_id, self._eos_token_id
            )

        # prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[-1]
        self._prepare_generated_length(input_ids_length)

        # reset cache if neccesary
        if self._use_kv_cache:
            self._past_key_cache_list, self._past_value_cache_list = None, None

        # prepare stopping criteria
        prepared_stopping_criteria = self._get_stopping_criteria()

        # run sample
        result = self._sample(input_ids, stopping_criteria=prepared_stopping_criteria, **model_kwargs)

        return result

    def _sample(self, input_ids: Tensor, stopping_criteria: StoppingCriteriaList, **model_kwargs: Tensor) -> Tensor:
        # init values
        pad_token_id = self._pad_token_id
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)

        # keep track of which sequences are already finished
        batch_size = input_ids.shape[0]
        this_peer_finished = False
        unfinished_sequences = ops.ones(batch_size, dtype=ms.int32)

        while not this_peer_finished:
            # prepare model inputs
            model_inputs = self.model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # inject kv cache state
            model_inputs["return_key_value_cache"] = self._use_kv_cache

            # forward pass to get next token
            loss, logits, key_cache_list, value_cache_list = self.model(**model_inputs)
            next_token_scores = logits[:, -1, :]

            # token selection
            next_tokens = ops.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = ops.concat([input_ids, next_tokens[:, None]], axis=-1)
            model_kwargs = self._update_model_kwargs_for_generation(model_kwargs, key_cache_list, value_cache_list)

            unfinished_sequences = ops.logical_and(unfinished_sequences, ~stopping_criteria(input_ids))
            this_peer_finished = unfinished_sequences.max() == 0

        return input_ids
