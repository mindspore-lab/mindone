from typing import Optional

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.ops.operations.nn_ops import FlashAttentionScore

__all__ = ["FlashAttentionSP"]


class FlashAttentionSP(nn.Cell):
    """Flash Attention for Sequence Parallel"""

    def __init__(
        self,
        head_num: int,
        keep_prob: float = 1.0,
        scale_value: float = 1.0,
        pre_tokens: int = 2147483647,
        next_tokens: int = 2147483647,
        input_layout: str = "BSH",
        sparse_mode: int = 0,
        use_attention_mask: bool = True,
        use_alibi_mask: bool = False,
        use_mqa: bool = False,
        dp: int = 1,
        mp: int = 1,
        sp: int = 1,
    ):
        super().__init__()
        self.head_num = head_num
        self.enable_dropout = keep_prob < 1.0
        self.input_layout = input_layout
        self.sparse_mode = sparse_mode
        self.use_alibi_mask = use_alibi_mask
        self.use_attention_mask = use_attention_mask
        self.use_mqa = use_mqa
        self.dp = dp
        self.mp = mp
        self.sp = sp

        fa_strategies = self._generate_flash_attention_strategy(dp, mp, sp)
        self.flash_attention = FlashAttentionScore(
            head_num=head_num,
            keep_prob=keep_prob,
            scale_value=scale_value,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode,
        )
        self.flash_attention.shard(fa_strategies)

        if self.use_alibi_mask:
            self.alibi_rescale_factor = Tensor([1.0 / scale_value], dtype=ms.float16)
            self.alibi_rescale_mul = ops.Mul().shard(((dp, mp, sp, 1), (1,)))

        if self.enable_dropout:
            self.keep_prob_tensor = Tensor(keep_prob, dtype=ms.float16)
            self.drop_gen_mask = ops.DropoutGenMask()

    def construct(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        alibi_mask: Optional[Tensor] = None,
        prefix: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ):
        if self.input_layout == "BSH":
            bsz, q_seq_len, _ = query.shape
            _, kv_seq_len, _ = key.shape
        else:
            bsz, _, q_seq_len, _ = query.shape
            _, _, kv_seq_len, _ = key.shape

        if self.enable_dropout:
            drop_mask_bits = ops.Reshape(
                self.drop_gen_mask((bsz, self.head_num, q_seq_len, kv_seq_len), self.keep_prob_tensor),
                (bsz, self.head_num, q_seq_len, kv_seq_len // 8),
            )
        else:
            drop_mask_bits = None

        if self.use_alibi_mask:
            alibi_mask = self.alibi_rescale_mul(alibi_mask, self.alibi_rescale_factor.to(alibi_mask.dtype))

        _, _, _, output = self.flash_attention(
            query, key, value, alibi_mask, drop_mask_bits, padding_mask, attn_mask, prefix
        )

        return output

    def _generate_flash_attention_strategy(self, dp, mp, sp):
        kv_head_split_num = 1 if self.use_mqa else mp
        if self.input_layout == "BSH":
            fa_strategies = ((dp, sp, mp), (dp, 1, kv_head_split_num), (dp, 1, kv_head_split_num))
        else:
            fa_strategies = ((dp, mp, sp, 1), (dp, kv_head_split_num, 1, 1), (dp, kv_head_split_num, 1, 1))

        if self.use_alibi_mask:
            fa_strategies += ((dp, mp, sp, 1),)

        if self.enable_dropout:
            fa_strategies += ((dp, mp, sp, 1),)

        if self.use_attention_mask:
            if self.sparse_mode in [0, 1]:
                fa_strategies += ((dp, 1, sp, 1),)
            elif self.sparse_mode == 2:
                fa_strategies += ((1, 1),)
            else:
                raise RuntimeError(f"sparse_mode: `{self.sparse_mode}` is not supported currently")

        return fa_strategies
