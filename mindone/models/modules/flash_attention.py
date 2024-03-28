"""FlashAttention Wrapper"""
import logging
import math
from typing import List, Optional

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn, ops

from mindone.utils.version_control import MS_VERSION, check_valid_flash_attention, choose_flash_attention_dtype

# try import fa
try:
    if MS_VERSION >= "2.3.0":
        from mindspore.ops.operations.nn_ops import FlashAttentionScore as FlashAttention
    else:
        from mindspore.nn.layer.flash_attention import FlashAttention
    import_fa_success = True
except Exception:
    import_fa_success = False

FLASH_IS_AVAILABLE = check_valid_flash_attention(import_fa_success)

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
        self.use_new_flash_attention = MS_VERSION >= "2.3.0"
        if self.use_new_flash_attention:
            self.flash_attention = FlashAttention(
                scale_value=1.0 / math.sqrt(head_dim),
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
        self.fix_head_dims = fix_head_dims  # A list of integers to be mapped to 2**n * 64
        if self.fix_head_dims is not None and not isinstance(self.fix_head_dims, (list, tuple)):
            self.fix_head_dims = [self.fix_head_dims]

    def construct(self, q, k, v, mask=None):
        q_b, h, q_n, d = q.shape  # (b, h, n, d)
        head_dim = d
        #   a trick to pad head dimensions to 2**n * 64
        if self.fix_head_dims is not None and head_dim in self.fix_head_dims:
            # pad to 2**n * 64 to avoid accuracy errors
            padding_size = 64 * 2 ** math.ceil(math.log(head_dim / 64, 2)) - head_dim
            q = msnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
            k = msnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
            v = msnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
        if self.use_new_flash_attention:
            if mask is not None:
                mask = mask.to(self.fa_mask_dtype)
                if mask.dim() == 3:
                    mask = mask[:, None, :, :]
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
                mask = ops.zeros((q_b, q_n, q_n), self.fa_mask_dtype)
            out = self.flash_attention(
                q.to(self.dtype),
                k.to(self.dtype),
                v.to(self.dtype),
                mask.to(self.fa_mask_dtype),
            )
        if self.fix_head_dims is not None and head_dim in self.fix_head_dims:
            out = ops.slice(out, [0, 0, 0, 0], [q_b, h, q_n, head_dim])
        return out
