import mindspore.mint as mint
from mindspore import Tensor, nn, ops
from mindspore.communication import get_group_size


class SeqAllToAll4D(nn.Cell):
    def __init__(self, group: str = None):
        super().__init__()

        self.group = group

        if group is not None:
            self.sp_size = get_group_size(group)

            if self.sp_size > 1:
                self.alltoall = ops.AlltoAll(split_count=self.sp_size, split_dim=0, concat_dim=0, group=group)

        else:
            self.sp_size = 1

    def construct(self, input: Tensor, scatter_idx: int, gather_idx: int) -> Tensor:
        if scatter_idx == 2 and gather_idx == 1:
            # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
            bs, shard_seqlen, hc, hs = input.shape
            seqlen = shard_seqlen * self.sp_size
            shard_hc = hc // self.sp_size

            # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
            # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
            input_t = input.reshape(bs, shard_seqlen, self.sp_size, shard_hc, hs).swapaxes(0, 2).contiguous()

            output = mint.zeros_like(input_t)
            # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
            # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head

            if self.sp_size > 1:
                # (P, seq_len/P, bs, hc/P, hs)
                # dist.all_to_all_single(output, input_t, group=group)
                output = self.alltoall(input_t)

            else:
                output = input_t

            # if scattering the seq-dim, transpose the heads back to the original dimension
            # output = output.swapaxes(0, 2).reshape(bs, seqlen, shard_hc, hs)

            output = output.reshape(seqlen, bs, shard_hc, hs)
            # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
            output = mint.swapaxes(output, 0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)

        elif scatter_idx == 1 and gather_idx == 2:
            # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
            bs, seqlen, shard_hc, hs = input.shape
            hc = shard_hc * self.sp_size
            shard_seqlen = seqlen // self.sp_size

            # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
            # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs)
            #       -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
            input_t = (
                mint.swapaxes(
                    mint.swapaxes(input.reshape(bs, self.sp_size, shard_seqlen, shard_hc, hs), 0, 3), 0, 1
                )  # (b, P, sp/P, hc/P, hs) -> (hc/P, P, sp/P, b, hs) -> (P, hc/P, sp/P, b, hs)
                .contiguous()
                .reshape(self.sp_size, shard_hc, shard_seqlen, bs, hs)
            )

            output = mint.zeros_like(input_t)
            # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
            # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head

            if self.sp_size > 1:
                # (P, hc/P, seqlen/P, bs, hs)
                # dist.all_to_all_single(output, input_t, group=group)
                output = self.alltoall(input_t)

            else:
                output = input_t

            # if scattering the seq-dim, transpose the heads back to the original dimension
            output = output.reshape(hc, shard_seqlen, bs, hs)

            # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
            output = mint.swapaxes(output, 0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        else:
            output = None

        return output

    # FIXME: alltoall bprop
    def bprop(
        self, x: Tensor, scatter_idx: int, gather_idx: int, out: Tensor, dout: Tensor
    ) -> tuple[Tensor, None, None]:
        return (
            self.construct(dout, dout, gather_idx, scatter_idx),
            None,
            None,
        )
