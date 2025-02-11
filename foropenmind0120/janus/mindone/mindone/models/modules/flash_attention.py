"""FlashAttention Wrapper"""
import logging
from typing import List, Optional

import mindspore as ms
from mindspore import nn, ops

from mindone.utils.version_control import check_valid_flash_attention, choose_flash_attention_dtype

FLASH_IS_AVAILABLE = check_valid_flash_attention()
USE_NEW_FA = False
if FLASH_IS_AVAILABLE:
    try:
        from mindspore.nn.layer.flash_attention import FlashAttention
    except Exception:
        from mindspore.ops.operations.nn_ops import FlashAttentionScore as FlashAttention

        USE_NEW_FA = True
        print("Get New FA API! ")

logger = logging.getLogger(__name__)
if FLASH_IS_AVAILABLE:
    logger.info("Flash attention is available.")
else:
    logger.info("Flash attention is unavailable.")

__all__ = ["FLASH_IS_AVAILABLE", "MSFlashAttention"]


class MSFlashAttention(nn.Cell):
    """
    This class represents a FlashAttention module compatible for different MS versions.
    Args:
        head_dim (int): The dimensionality of each attention head.
        head_num (int): The number of attention heads.
        fix_head_dims (list or None): A list of integers representing head dimensions to be padded to 2**n * 64, where n is the integer value.
        attention_dropout (float): The dropout rate applied to attention matrix.
        input_layout (str): The input data layout. Defaults to "BNSD".
        high_precision (bool): Determines whether to use high precision mode for attention calculations. Defaults to True.
        dtype (ms.dtype): The data type for query, key, and value tensors. Defaults to ms.float16.
    Attributes:
        use_new_flash_attention (bool): Indicates whether the new FlashAttention module supported in ms 2.3.0.
        flash_attention (FlashAttention): An instance of the FlashAttention module used for attention calculations.
        fa_mask_dtype (dtype): The data type used for the attention mask (ms.uint8 or ms.float16 depending on the version).
        fix_head_dims (list): A list of integers representing head dimensions to be padded to 2**n * 64.
        dtype (ms.dtype): The data type for query, key, and value tensors.
    """

    def __init__(
        self,
        head_dim: int,
        head_num: int,
        fix_head_dims: Optional[List[int]] = None,
        attention_dropout: float = 0.0,
        input_layout: str = "BNSD",
        high_precision: bool = True,
        dtype: ms.dtype = ms.float16,
    ):
        super().__init__()
        assert FLASH_IS_AVAILABLE, "FlashAttention is not Available!"
        self.use_new_flash_attention = USE_NEW_FA
        self.input_layout = input_layout
        if input_layout not in ["BSH", "BNSD"]:
            raise ValueError(f"input_layout must be in ['BSH', 'BNSD'], but get {input_layout}.")
        self.head_dim = head_dim
        if self.use_new_flash_attention:
            self.flash_attention = FlashAttention(
                scale_value=head_dim**-0.5,
                head_num=head_num,
                input_layout=input_layout,
                keep_prob=1 - attention_dropout,
            )
        else:
            self.flash_attention = FlashAttention(
                head_dim=head_dim,
                head_num=head_num,
                high_precision=high_precision,
                dropout_rate=attention_dropout,
            )  # TODO: how high_precision affect the training or inference quality
        self.fa_mask_dtype = choose_flash_attention_dtype()  # ms.uint8 or ms.float16 depending on version
        self.dtype = dtype
        cand_d_list = [64, 80, 96, 120, 128, 256]
        self.d_pad = 0
        for d in cand_d_list:
            if head_dim == d:
                self.d_pad = 0
                break
            elif head_dim < d:
                self.d_pad = d - head_dim
                break
        if head_dim > 256:
            raise ValueError("head_dim must <= 256!")
        self.need_pad = self.d_pad != 0

    def _rearange_input(self, x):
        x = x.to(self.dtype)
        if self.need_pad:
            if self.input_layout == "BNSD":
                B, N, S, D = x.shape
                pad = ops.zeros((B, N, S, self.d_pad), x.dtype)
            else:
                B, S = x.shape[:2]
                x = x.reshape(B, S, -1, self.head_dim)
                pad = ops.zeros((B, S, x.shape[2], self.d_pad), x.dtype)
            x = ops.concat((x, pad), axis=-1)
        if self.input_layout == "BSH":
            B, S = x.shape[:2]
            x = x.reshape(B, S, -1)
        return x

    def _rearange_output(self, x, dtype):
        if self.input_layout == "BSH":
            B, S = x.shape[:2]
            x = x.reshape(B, S, -1, self.head_dim + self.d_pad)
        if self.need_pad:
            x = x[:, :, :, : self.head_dim]
        return x.to(dtype)

    def construct(self, q, k, v, mask=None):
        if not self.use_new_flash_attention:
            B, N, S1, D = q.shape
            S2 = k.shape[2]
            if mask is None:
                mask = ops.zeros((B, S1, S2), self.fa_mask_dtype)
            out = self.flash_attention(
                q.to(self.dtype),
                k.to(self.dtype),
                v.to(self.dtype),
                mask.to(self.fa_mask_dtype),
            )
            return out

        q_dtype = q.dtype
        q = self._rearange_input(q)
        k = self._rearange_input(k)
        v = self._rearange_input(v)
        if mask is not None:
            mask = mask.to(ms.uint8)
        out = self.flash_attention(q, k, v, None, None, None, mask)[3]
        out = self._rearange_output(out, q_dtype)
        return out
