# Copyright 2023 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import ops

_MIN_FP16 = ms.tensor(np.finfo(np.float16).min, dtype=ms.float16)
_MIN_FP32 = ms.tensor(np.finfo(np.float32).min, dtype=ms.float32)
_MIN_FP64 = ms.tensor(np.finfo(np.float64).min, dtype=ms.float64)
_MIN_BF16 = ms.tensor(float.fromhex("-0x1.fe00000000000p+127"), dtype=ms.bfloat16)


def dtype_to_min(dtype):
    if dtype == ms.float16:
        return _MIN_FP16
    if dtype == ms.float32:
        return _MIN_FP32
    if dtype == ms.float64:
        return _MIN_FP64
    if dtype == ms.bfloat16:
        return _MIN_BF16
    else:
        raise ValueError(f"Only support get minimum value of (float16, ), but got {dtype}")


@dataclass
class AttentionMaskConverter:
    """
    A utility attention mask class that allows one to:
        - Create a causal 4d mask
        - Create a causal 4d mask with slided window
        - Convert a 2d attention mask (batch_size, query_length) to a 4d attention mask (batch_size, 1, query_length,
          key_value_length) that can be multiplied with attention scores

    Examples:

    ```python
    >>> import mindspore as ms
    >>> from transformers.modeling_attn_mask_utils import AttentionMaskConverter

    >>> converter = AttentionMaskConverter(True)
    >>> converter.to_4d(ms.tensor([[0, 0, 0, 1, 1]]), 5, key_value_length=5, dtype=ms.float32)
    tensor([[[[-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00, -3.4028e+38],
            [-3.4028e+38, -3.4028e+38, -3.4028e+38,  0.0000e+00,  0.0000e+00]]]])
    ```

    Parameters:
        is_causal (`bool`):
            Whether the attention mask should be a uni-directional (causal) or bi-directional mask.

        sliding_window (`int`, *optional*):
            Optionally, the sliding window masks can be created if `sliding_window` is defined to a positive integer.
    """

    is_causal: bool
    sliding_window: int

    def __init__(self, is_causal: bool, sliding_window: Optional[int] = None):
        self.is_causal = is_causal
        self.sliding_window = sliding_window

        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError(
                f"Make sure that when passing `sliding_window` that its value is a strictly positive integer, not `{self.sliding_window}`"
            )

    def to_causal_4d(
        self,
        batch_size: int,
        query_length: int,
        key_value_length: int,
        dtype: ms.Type,
    ) -> Optional[ms.Tensor]:
        """
        Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
        bias to upper right hand triangular matrix (causal mask).
        """
        if not self.is_causal:
            raise ValueError(f"Please use `to_causal_4d` only if {self.__class__} has `is_causal` set to True.")
        return to_causal_4d(batch_size, query_length, key_value_length, dtype, self.sliding_window)

    def to_4d(
        self,
        attention_mask_2d: ms.Tensor,
        query_length: int,
        dtype: ms.Type,
        key_value_length: Optional[int] = None,
    ) -> ms.Tensor:
        """
        Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
        key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
        causal, a causal mask will be added.
        """
        return to_4d(attention_mask_2d, query_length, dtype, key_value_length, self.is_causal, self.sliding_window)

    @staticmethod
    def _make_causal_mask(
        input_ids_shape: Union[Tuple, List],
        dtype: ms.Type,
        past_key_values_length: int = 0,
        sliding_window: Optional[int] = None,
    ):
        """
        Make causal mask used for bi-directional self-attention.
        """
        return _make_causal_mask(input_ids_shape, dtype, past_key_values_length, sliding_window)

    @staticmethod
    def _expand_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
        """
        Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
        """
        return _expand_mask(mask, dtype, tgt_len)


def to_causal_4d(
    batch_size: int,
    query_length: int,
    key_value_length: int,
    dtype: ms.Type,
    sliding_window: Optional[int] = None,
) -> Optional[ms.Tensor]:
    """
    Creates a causal 4D mask of (bsz, head_dim=1, query_length, key_value_length) shape and adds large negative
    bias to upper right hand triangular matrix (causal mask).
    """
    # If shape is not cached, create a new causal mask and cache it
    input_shape = (batch_size, query_length)
    past_key_values_length = key_value_length - query_length

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    causal_4d_mask = None
    if input_shape[-1] > 1 or sliding_window is not None:
        causal_4d_mask = _make_causal_mask(
            input_shape,
            dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )

    return causal_4d_mask


def to_4d(
    attention_mask_2d: ms.Tensor,
    query_length: int,
    dtype: ms.Type,
    key_value_length: Optional[int] = None,
    is_causal: bool = True,
    sliding_window: Optional[int] = None,
) -> ms.Tensor:
    """
    Converts 2D attention mask to 4D attention mask by expanding mask to (bsz, head_dim=1, query_length,
    key_value_length) shape and by adding a large negative bias to not-attended positions. If attention_mask is
    causal, a causal mask will be added.
    """
    input_shape = (attention_mask_2d.shape[0], query_length)

    # create causal mask
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    causal_4d_mask = None
    if (input_shape[-1] > 1 or sliding_window is not None) and is_causal:
        if key_value_length is None:
            raise ValueError(
                "This attention mask converter is causal. Make sure to pass `key_value_length` to correctly create a causal mask."
            )

        past_key_values_length = key_value_length - query_length
        causal_4d_mask = _make_causal_mask(
            input_shape,
            dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window,
        )
    elif sliding_window is not None:
        raise NotImplementedError("Sliding window is currently only implemented for causal masking")

    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    expanded_attn_mask = _expand_mask(attention_mask_2d, dtype, tgt_len=input_shape[-1])

    if causal_4d_mask is not None:
        expanded_attn_mask = causal_4d_mask.masked_fill(expanded_attn_mask.bool(), dtype_to_min(dtype))

    # expanded_attn_mask + causal_4d_mask can cause some overflow
    expanded_4d_mask = expanded_attn_mask

    return expanded_4d_mask


def _make_causal_mask(
    input_ids_shape: Union[Tuple, List],
    dtype: ms.Type,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full((tgt_len, tgt_len), dtype_to_min(dtype), dtype=dtype)
    mask_cond = ops.arange(mask.shape[-1])
    mask = mask.masked_fill(mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), ms.tensor(0).to(dtype))

    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = ops.cat([ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

    # add lower triangular sliding window mask if necessary
    if sliding_window is not None:
        diagonal = past_key_values_length - sliding_window + 1

        context_mask = 1 - ops.triu(ops.ones_like(mask, dtype=ms.int32), diagonal=diagonal)
        mask = mask.masked_fill(context_mask.bool(), dtype_to_min(dtype))

    return mask[None, None, :, :].tile((bsz, 1, 1, 1))


def _expand_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].tile((1, 1, tgt_len, 1)).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), dtype_to_min(dtype))


def _prepare_4d_causal_attention_mask(
    attention_mask: Optional[ms.Tensor],
    input_shape: Union[Tuple, List],
    inputs_embeds: ms.Tensor,
    past_key_values_length: int,
    sliding_window: Optional[int] = None,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        attention_mask (`ms.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        input_shape (`tuple(int)` or `list(int)` or `torch.Size`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        inputs_embeds (`ms.Tensor`):
            The embedded inputs as a torch Tensor.
        past_key_values_length (`int`):
            The length of the key value cache.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    key_value_length = input_shape[-1] + past_key_values_length

    # 4d mask is passed through the layers
    if attention_mask is not None and len(attention_mask.shape) == 2:
        attention_mask = to_4d(
            attention_mask,
            input_shape[-1],
            key_value_length=key_value_length,
            dtype=inputs_embeds.dtype,
            is_causal=True,
            sliding_window=sliding_window,
        )
    elif attention_mask is not None and len(attention_mask.shape) == 4:
        expected_shape = (input_shape[0], 1, input_shape[1], key_value_length)
        if tuple(attention_mask.shape) != expected_shape:
            raise ValueError(
                f"Incorrect 4D attention_mask shape: {tuple(attention_mask.shape)}; expected: {expected_shape}."
            )
        else:
            # if the 4D mask has correct shape - invert it and fill with negative infinity
            inverted_mask = 1.0 - attention_mask
            attention_mask = inverted_mask.masked_fill(inverted_mask.bool(), dtype_to_min(inputs_embeds.dtype))
    else:
        attention_mask = to_causal_4d(
            input_shape[0], input_shape[-1], key_value_length, dtype=inputs_embeds.dtype, sliding_window=sliding_window
        )

    return attention_mask


def _prepare_4d_attention_mask(mask: ms.Tensor, dtype: ms.Type, tgt_len: Optional[int] = None):
    """
    Creates a non-causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`

    Args:
        mask (`ms.Tensor` or `None`):
            A 2D attention mask of shape `(batch_size, key_value_length)`
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        tgt_len (`int`):
            The target length or query length the created mask shall have.
    """
    return _expand_mask(mask=mask, dtype=dtype, tgt_len=tgt_len)


def _create_4d_causal_attention_mask(
    input_shape: Union[Tuple, List],
    dtype: ms.Type,
    past_key_values_length: int = 0,
    sliding_window: Optional[int] = None,
) -> Optional[ms.Tensor]:
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)`

    Args:
        input_shape (`tuple(int)` or `list(int)`):
            The input shape should be a tuple that defines `(batch_size, query_length)`.
        dtype (`torch.dtype`):
            The torch dtype the created mask shall have.
        sliding_window (`int`, *optional*):
            If the model uses windowed attention, a sliding window should be passed.
    """
    key_value_length = past_key_values_length + input_shape[-1]
    attention_mask = to_causal_4d(
        input_shape[0], input_shape[-1], key_value_length, dtype=dtype, sliding_window=sliding_window
    )

    return attention_mask
