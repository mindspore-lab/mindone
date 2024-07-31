import logging

from opensora.acceleration.parallel_states import hccl_info

import mindspore as ms
from mindspore import Tensor, nn, ops

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

        return output


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
    # split of input data for seq parallelism
    sp_size = hccl_info.world_size
    index = hccl_info.rank % sp_size
    temp_attention_mask = None
    loss_mask = None

    if use_image_num == 0:
        assert hidden_states.shape[2] % sp_size == 0
        assert encoder_hidden_states.shape[1] % sp_size == 0
        assert encoder_attention_mask.shape[1] % sp_size == 0

        hidden_states = ops.chunk(hidden_states, sp_size, 2)[index]
        noise = ops.chunk(noise, sp_size, 2)[index]
        encoder_hidden_states = ops.chunk(encoder_hidden_states, sp_size, 1)[index]
        encoder_attention_mask = ops.chunk(encoder_attention_mask, sp_size, 1)[index]

        if attention_mask is not None:
            assert attention_mask.shape[1] % sp_size == 0
            attention_mask = ops.chunk(attention_mask, sp_size, 1)[index]

    else:
        video_states, image_states = hidden_states[:, :, :-use_image_num], hidden_states[:, :, -use_image_num:]
        video_noise, image_noise = noise[:, :, :-use_image_num], noise[:, :, -use_image_num:]
        video_encoder_states, image_encoder_states = (
            encoder_hidden_states[:, :-use_image_num],
            encoder_hidden_states[:, -use_image_num:],
        )
        video_encoder_attention_mask, image_encoder_attention_mask = (
            encoder_attention_mask[:, :-use_image_num],
            encoder_attention_mask[:, -use_image_num:],
        )

        if attention_mask is not None:
            video_attention_mask, image_attention_mask = (
                attention_mask[:, :-use_image_num],
                attention_mask[:, -use_image_num:],
            )
        else:
            video_attention_mask, image_attention_mask = None, None

        # 1. for video states
        padding_needed_v = (sp_size - video_states.shape[2] % sp_size) % sp_size
        if padding_needed_v > 0:
            logger.debug("Doing video padding")
            # B, C, T, H, W -> B, C, T', H, W
            video_states = ops.pad(video_states, (0, 0, 0, 0, 0, padding_needed_v), mode="constant", value=0)
            video_noise = ops.pad(video_noise, (0, 0, 0, 0, 0, padding_needed_v), mode="constant", value=0)
            if attention_mask is not None:
                # B, T, H, W -> B, T', H, W
                video_attention_mask = ops.pad(
                    video_attention_mask, (0, 0, 0, 0, 0, padding_needed_v), mode="constant", value=0
                )

            b, _, f, h, w = video_states.shape
            temp_attention_mask = ops.ones((b * h * w // 4, 1, f), ms.int32)
            temp_attention_mask[:, :, -padding_needed_v:] = 0

            video_loss_mask = ops.ones((1, 1, f, 1, 1), ms.int32)
            video_loss_mask[:, :, -padding_needed_v:] = 0
        else:
            video_loss_mask = ops.ones((1, 1, video_states.shape[2], 1, 1), ms.int32)

        video_states, video_encoder_states, video_encoder_attention_mask = (
            video_states,
            video_encoder_states.tile((1, sp_size, 1, 1)),
            video_encoder_attention_mask.tile((1, sp_size, 1)),
        )

        assert video_states.shape[2] % sp_size == 0
        assert video_encoder_states.shape[1] % sp_size == 0
        assert video_encoder_attention_mask.shape[1] % sp_size == 0

        video_states = ops.chunk(video_states, sp_size, 2)[index]
        video_noise = ops.chunk(video_noise, sp_size, 2)[index]
        video_encoder_states = ops.chunk(video_encoder_states, sp_size, 1)[index]
        video_encoder_attention_mask = ops.chunk(video_encoder_attention_mask, sp_size, 1)[index]
        video_loss_mask = ops.chunk(video_loss_mask, sp_size, 2)[index]

        if attention_mask is not None:
            assert video_attention_mask.shape[1] % sp_size == 0
            video_attention_mask = ops.chunk(video_attention_mask, sp_size, 1)[index]

        # 2. for image states
        padding_needed_i = (sp_size - image_states.shape[2] % sp_size) % sp_size
        if padding_needed_i > 0:
            image_states = ops.pad(image_states, (0, 0, 0, 0, 0, padding_needed_i), mode="constant", value=0)
            image_noise = ops.pad(image_noise, (0, 0, 0, 0, 0, padding_needed_i), mode="constant", value=0)
            image_encoder_states = ops.pad(
                image_encoder_states, (0, 0, 0, 0, 0, padding_needed_i), mode="constant", value=0
            )
            image_encoder_attention_mask = ops.pad(
                image_encoder_attention_mask, (0, 0, 0, padding_needed_i), mode="constant", value=0
            )
            if attention_mask is not None:
                image_attention_mask = ops.pad(
                    image_attention_mask, (0, 0, 0, 0, 0, padding_needed_i), mode="constant", value=0
                )

            image_loss_mask = ops.ones((1, 1, image_states.shape[2], 1, 1), ms.int32)
            image_loss_mask[:, :, -padding_needed_i:] = 0
        else:
            image_loss_mask = ops.ones((1, 1, image_states.shape[2], 1, 1), ms.int32)

        assert image_states.shape[2] % sp_size == 0
        assert image_encoder_states.shape[1] % sp_size == 0
        assert image_encoder_attention_mask.shape[1] % sp_size == 0

        image_states = ops.chunk(image_states, sp_size, 2)[index]
        image_noise = ops.chunk(image_noise, sp_size, 2)[index]
        image_encoder_states = ops.chunk(image_encoder_states, sp_size, 1)[index]
        image_encoder_attention_mask = ops.chunk(image_encoder_attention_mask, sp_size, 1)[index]
        image_loss_mask = ops.chunk(image_loss_mask, sp_size, 2)[index]

        if attention_mask is not None:
            assert image_attention_mask.shape[1] % sp_size == 0
            image_attention_mask = ops.chunk(image_attention_mask, sp_size, 1)[index]

        # 3. concat
        hidden_states = ops.concat([video_states, image_states], axis=2)
        noise = ops.concat([video_noise, image_noise], axis=2)
        encoder_hidden_states = ops.concat([video_encoder_states, image_encoder_states], axis=1)
        encoder_attention_mask = ops.concat([video_encoder_attention_mask, image_encoder_attention_mask], axis=1)
        loss_mask = (
            ops.concat([video_loss_mask, image_loss_mask], axis=2)
            if (padding_needed_i > 0 or padding_needed_v > 0)
            else None
        )
        use_image_num = (use_image_num + sp_size - 1) // sp_size

        if attention_mask is not None:
            attention_mask = ops.concat([video_attention_mask, image_attention_mask], axis=1)

    return (
        hidden_states,
        noise,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
        use_image_num,
        temp_attention_mask,
        loss_mask,
    )
