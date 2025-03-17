from stepvideo.mindspore_adapter.all_to_all import SeqAllToAll4D

import mindspore as ms
from mindspore import Tensor, mint, nn
from mindspore.communication.management import get_group_size

from mindone.transformers.mindspore_adapter.attention import FlashAttention2


class LongContextAttention(nn.Cell):
    """Initialization.

    Arguments:
        ulysses_pg (str): ulysses process group
        ring_pg (str): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        attn_type: str = "fa",
        ring_pg: str = None,
        ulysses_pg: str = None,
        head_dim: int = None,
        head_num: int = None,
    ) -> None:
        super(LongContextAttention, self).__init__()
        self.ring_pg = ring_pg
        self.ulysses_pg = ulysses_pg

        sp_size = 1
        if ulysses_pg is not None:
            sp_size = get_group_size(ulysses_pg)

            assert head_num % sp_size == 0, f"unavailable {sp_size=} and {head_num=}"

        if ring_pg is not None:
            raise NotImplementedError
        if attn_type not in [
            "fa",
        ]:
            raise NotImplementedError

        if attn_type == "fa":
            self.fa_attn_fn = FlashAttention2(
                head_dim=head_dim,
                head_num=head_num // sp_size,
                attention_dropout=0.0,
                input_layout="BNSD",
                dtype=ms.bfloat16,
            )
        else:
            raise NotImplementedError

        # assert (
        #     self.ulysses_pg is not None or self.ring_pg is not None
        # ), f"use set_seq_parallel_pg() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"

        # if self.ulysses_pg is None and self.ring_pg is None:
        #     print("warning: ulysses pg and ring sp both is None.")

        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        self.seq_alltoall_4d = SeqAllToAll4D(group=ulysses_pg)

    def construct(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        """construct

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        query_layer = self.seq_alltoall_4d(query, self.scatter_idx, self.gather_idx)
        key_layer = self.seq_alltoall_4d(key, self.scatter_idx, self.gather_idx)
        value_layer = self.seq_alltoall_4d(value, self.scatter_idx, self.gather_idx)

        # BSND -> BNSD, for fa
        query_layer = mint.swapaxes(query_layer, 1, 2)
        key_layer = mint.swapaxes(key_layer, 1, 2)
        value_layer = mint.swapaxes(value_layer, 1, 2)

        out = self.fa_attn_fn(
            query_layer,
            key_layer,
            value_layer,
        )

        # BNSD -> BSND, for fa
        out = mint.swapaxes(out, 1, 2)

        context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = self.seq_alltoall_4d(context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [s/p::h]
        return output
