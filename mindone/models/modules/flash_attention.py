"""FlashAttention Wrapper"""
import logging
import math

import mindspore as ms
import mindspore.numpy as msnp
from mindspore import nn, ops

from mindone.utils.version_control import MSVersion, check_valid_flash_attention, choose_flash_attention_dtype

# try import fa
try:
    if MSVersion >= "2.3.0":
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
    def __init__(
        self, head_dim, head_num, fix_head_dims=None, attention_dropout=0.0, input_layout="BNSD", high_precision=True
    ):
        assert FLASH_IS_AVAILABLE, "FlashAttention is not Available!"
        if MSVersion >= "2.3.0":
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
        self.fix_head_dims = fix_head_dims  # A list of integers to be mapped to 2**n * 64
        if self.fix_head_dims is not None and not isinstance(self.fix_head_dims, (list, tuple)):
            self.fix_head_dims = [self.fix_head_dims]

    def construct(self, q, k, v, mask=None, **kwargs):
        q_b, q_n, h, d = q.shape
        head_dim = d
        if mask is None:
            mask = ops.zeros((q_b, q_n, q_n), self.fa_mask_dtype)
        #   a trick to pad head dimensions to 2**n * 64
        if self.fix_head_dims is not None and head_dim in self.fix_head_dims:
            # pad to 2**n * 64 to avoid accuracy errors
            padding_size = 64 * 2 ** math.ceil(math.log(head_dim / 64, 2)) - head_dim
            q = msnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
            k = msnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
            v = msnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, padding_size)), constant_value=0)
        if MSVersion >= "2.3.0":
            out = self.flash_attention(
                q.to(ms.float16),
                k.to(ms.float16),
                v.to(ms.float16),
                None,
                None,
                None,
                mask[:, None, :, :].to(self.fa_mask_dtype),
                None,
                **kwargs
            )[3]
        else:
            out = self.flash_attention(
                q.to(ms.float16), k.to(ms.float16), v.to(ms.float16), mask.to(self.fa_mask_dtype), **kwargs
            )
        if self.fix_head_dims is not None and head_dim in self.fix_head_dims:
            out = ops.slice(out, [0, 0, 0, 0], [q_b, h, q_n, head_dim])
        return out
