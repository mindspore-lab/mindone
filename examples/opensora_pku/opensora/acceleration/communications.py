from opensora.acceleration.parallel_states import hccl_info

from mindspore import Tensor, nn, ops


class AlltoAll(nn.Cell):
    def __init__(self, split_count=None, group=None):
        super(AlltoAll, self).__init__()
        self.all_gather = ops.AllGather(group=group)
        self.split_count = split_count
        self.index = hccl_info.rank % hccl_info.world_size

    def construct(self, x):
        x_shape = x.shape
        x = self.all_gather(x[None, ...])  # (8, ...)
        x = ops.chunk(x, self.split_count, axis=1)[self.index]
        x = x.view(x_shape)
        return x


class _SingleAll2ALL(nn.Cell):
    def __init__(self, scatter_dim: int, gather_dim: int):
        super(_SingleAll2ALL, self).__init__()
        self.sp_size = hccl_info.world_size
        self.spg = hccl_info.group
        self.scatter_dim = scatter_dim
        self.gather_dim = gather_dim
        # self.alltoall = ops.AlltoAll(split_count=self.sp_size, split_dim=0, concat_dim=0, group=self.spg)
        self.alltoall = AlltoAll(split_count=self.sp_size, group=self.spg)

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


class bak_SingleAll2ALL(nn.Cell):
    def __init__(self, scatter_dim: int, gather_dim: int):
        super(_SingleAll2ALL, self).__init__()
        self.sp_size = hccl_info.world_size
        self.spg = hccl_info.group
        self.scatter_dim = scatter_dim
        self.gather_dim = gather_dim
        self.alltoall = ops.AlltoAll(split_count=self.sp_size, split_dim=0, concat_dim=0, group=self.spg)

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


def prepare_parallel_data(hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, use_image_num):
    sp_size = hccl_info.world_size
    index = hccl_info.rank % sp_size

    if use_image_num == 0:
        assert hidden_states.shape[2] % sp_size == 0
        assert encoder_hidden_states.shape[1] % sp_size == 0
        assert encoder_attention_mask.shape[1] % sp_size == 0

        hidden_states = ops.chunk(hidden_states, sp_size, 2)[index]
        encoder_hidden_states = ops.chunk(encoder_hidden_states, sp_size, 1)[index]
        encoder_attention_mask = ops.chunk(encoder_attention_mask, sp_size, 1)[index]

        if attention_mask is not None:
            assert attention_mask.shape[1] % sp_size == 0
            attention_mask = ops.chunk(attention_mask, sp_size, 1)[index]

    else:
        video_states, image_states = hidden_states[:, :, :-use_image_num], hidden_states[:, :, -use_image_num:]
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
        padding_needed = (sp_size - video_states.shape[2] % sp_size) % sp_size
        if padding_needed > 0:
            print("Doing video padding")
            # B, C, T, H, W -> B, C, T', H, W
            video_states = ops.pad(video_states, (0, 0, 0, 0, 0, padding_needed), mode="constant", value=0)
            if attention_mask is not None:
                # B, T, H, W -> B, T', H, W
                video_attention_mask = ops.pad(
                    video_attention_mask, (0, 0, 0, 0, 0, padding_needed), mode="constant", value=0
                )

        video_states, video_encoder_states, video_encoder_attention_mask = (
            video_states,
            video_encoder_states.tile((1, sp_size, 1, 1)),
            video_encoder_attention_mask.tile((1, sp_size, 1)),
        )

        assert video_states.shape[2] % sp_size == 0
        assert video_encoder_states.shape[1] % sp_size == 0
        assert video_encoder_attention_mask.shape[1] % sp_size == 0

        video_states = ops.chunk(video_states, sp_size, 2)[index]
        video_encoder_states = ops.chunk(video_encoder_states, sp_size, 1)[index]
        video_encoder_attention_mask = ops.chunk(video_encoder_attention_mask, sp_size, 1)[index]

        if attention_mask is not None:
            assert video_attention_mask.shape[1] % sp_size == 0
            video_attention_mask = ops.chunk(video_attention_mask, sp_size, 1)[index]

        # 2. for image states
        padding_needed = (sp_size - image_states.shape[2] % sp_size) % sp_size
        if padding_needed > 0:
            image_states = ops.pad(image_states, (0, 0, 0, 0, 0, padding_needed), mode="constant", value=0)
            image_encoder_states = ops.pad(
                image_encoder_states, (0, 0, 0, 0, 0, padding_needed), mode="constant", value=0
            )
            image_encoder_attention_mask = ops.pad(
                image_encoder_attention_mask, (0, 0, 0, padding_needed), mode="constant", value=0
            )
            if attention_mask is not None:
                image_attention_mask = ops.pad(
                    image_attention_mask, (0, 0, 0, 0, 0, padding_needed), mode="constant", value=0
                )

        assert image_states.shape[2] % sp_size == 0
        assert image_encoder_states.shape[1] % sp_size == 0
        assert image_encoder_attention_mask.shape[1] % sp_size == 0

        image_states = ops.chunk(image_states, sp_size, 2)[index]
        image_encoder_states = ops.chunk(image_encoder_states, sp_size, 1)[index]
        image_encoder_attention_mask = ops.chunk(image_encoder_attention_mask, sp_size, 1)[index]

        if attention_mask is not None:
            assert image_attention_mask.shape[1] % sp_size == 0
            image_attention_mask = ops.chunk(image_attention_mask, sp_size, 1)[index]

        # 3. concat
        hidden_states = ops.concat([video_states, image_states], axis=2)
        encoder_hidden_states = ops.concat([video_encoder_states, image_encoder_states], axis=1)
        encoder_attention_mask = ops.concat([video_encoder_attention_mask, image_encoder_attention_mask], axis=1)
        use_image_num = (use_image_num + sp_size - 1) // sp_size

        if attention_mask is not None:
            attention_mask = ops.concat([video_attention_mask, image_attention_mask], axis=1)

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, use_image_num
