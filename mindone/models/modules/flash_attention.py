"""FlashAttention Wrapper"""
import logging
import math
from typing import List, Optional

import mindspore as ms
import mindspore.numpy as msnp
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
            raise ValueError(f"head_dim must <= 256!")
        self.need_pad = self.d_pad != 0

    def construct(self, q, k, v, mask=None):
        B, N, S1, D = q.shape
        _, _, S2, _ = k.shape

        if self.need_pad:
            q = ops.pad(q, (0, self.d_pad))
            k = ops.pad(k, (0, self.d_pad))
            v = ops.pad(v, (0, self.d_pad))
        if self.use_new_flash_attention:
            out = self.flash_attention(
                q.to(self.dtype),
                k.to(self.dtype),
                v.to(self.dtype),
                None,
                None,
                None,
                mask,
                None,
            )[3]
        else:
            if mask is None:
                mask = ops.zeros((B, S1, S2), self.fa_mask_dtype)
            out = self.flash_attention(
                q.to(self.dtype),
                k.to(self.dtype),
                v.to(self.dtype),
                mask.to(self.fa_mask_dtype),
            )
        if self.need_pad:
            out = out[:, :, :, :D]
        return out

