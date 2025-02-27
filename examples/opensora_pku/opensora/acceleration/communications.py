import logging

from opensora.acceleration.parallel_states import hccl_info

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

logger = logging.getLogger(__name__)


class _SingleAll2ALL(nn.Cell):
    def __init__(self, scatter_dim: int, gather_dim: int):
        super(_SingleAll2ALL, self).__init__()
        self.sp_size = hccl_info.world_size
        self.spg = hccl_info.group
        self.scatter_dim = scatter_dim
        self.gather_dim = gather_dim
        self.alltoall = ops.AlltoAll(split_count=self.sp_size, split_dim=0, concat_dim=0, group=self.spg)
        # self.alltoall = AlltoAll(split_count=self.sp_size, group=self.spg)

    def construct(self, input_: Tensor):
        origin_dtype = input_.dtype
        if input_.dtype == ms.bfloat16:
            input_ = input_.to(ms.float32)

        scatter_dim, gather_dim, sp_size = self.scatter_dim, self.gather_dim, self.sp_size
        inp_shape = list(input_.shape)
        inp_shape[scatter_dim] = inp_shape[scatter_dim] // sp_size
        if scatter_dim < 1:
            input_t = input_.reshape([sp_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :])
        else:
            # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
            input_t = (
                input_.reshape([-1, sp_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :])
                .swapaxes(0, 1)
                .contiguous()
            )

        output = self.alltoall(input_t)

        if scatter_dim < 1:
            output = output.swapaxes(0, 1).contiguous()

        output = output.reshape(
            inp_shape[:gather_dim]
            + [
                inp_shape[gather_dim] * sp_size,
            ]
            + inp_shape[gather_dim + 1 :]
        )

        return output.to(origin_dtype)


class AllGather(nn.Cell):
    def __init__(self):
        super(AllGather, self).__init__()
        self.op = ops.AllGather(hccl_info.group)

    def construct(self, x):
        return self.op(x)


class AllToAll_SBH(_SingleAll2ALL):
    def __init__(self, scatter_dim: int = 1, gather_dim: int = 0):
        super(AllToAll_SBH, self).__init__(scatter_dim=scatter_dim, gather_dim=gather_dim)


def prepare_parallel_data(
    hidden_states, noise, encoder_hidden_states, attention_mask, encoder_attention_mask, use_image_num
):
    # split input data for seq parallelism
    sp_size = hccl_info.world_size
    index = hccl_info.rank % sp_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, f"frame {frame} should be a multiple of sp_size {sp_size}"
    # b 1 (n x) h -> b n x h
    b, one_, nx, h = encoder_hidden_states.shape
    encoder_hidden_states = encoder_hidden_states.view((b, sp_size, nx // sp_size, h))
    attention_mask = attention_mask.tile((1, sp_size, 1, 1))
    encoder_attention_mask = encoder_attention_mask.tile((1, sp_size, 1))

    assert one_ == 1
    assert attention_mask is not None
    assert noise.shape == hidden_states.shape

    assert hidden_states.shape[2] % sp_size == 0
    assert encoder_hidden_states.shape[1] % sp_size == 0
    assert attention_mask.shape[1] % sp_size == 0
    assert encoder_attention_mask.shape[1] % sp_size == 0

    hidden_states = mint.chunk(hidden_states, sp_size, 2)[index]
    noise = mint.chunk(noise, sp_size, 2)[index]
    encoder_hidden_states = mint.chunk(encoder_hidden_states, sp_size, 1)[index]
    encoder_attention_mask = mint.chunk(encoder_attention_mask, sp_size, 1)[index]
    attention_mask = mint.chunk(attention_mask, sp_size, 1)[index]
    return hidden_states, noise, encoder_hidden_states, attention_mask, encoder_attention_mask, use_image_num
