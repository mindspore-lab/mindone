import mindspore as ms
from mindspore import Tensor, mint, ops
from mindspore.communication.management import GlobalComm, get_group_size, get_rank, init

# FIXME: valid global variables in mindspore static graph
sp_group = None
sp_size = None
sp_rank = None
_is_distribute = False


# custom pipeline parallel
pp_group = None
pp_size = None
pp_rank = None
pp_split_index = None


def is_distribute():
    return _is_distribute


# new, w/ pp
def initialize_parall_group(args: any = None, ring_degree=1, ulysses_degree=1):
    global _is_distribute
    global pp_split_index

    pp_degree = args.pp_degree
    pp_split_index = args.pp_split_index

    world_size = 1
    rank_id = 0
    if ring_degree > 1 or ulysses_degree > 1 or pp_degree > 1:
        init()
        world_size = get_group_size()
        rank_id = get_rank()
        print(f"init_environment, rank_id: {rank_id}, world_size: {world_size}")

        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=world_size,
        )

        _is_distribute = True

    global sp_group
    global sp_size
    global sp_rank

    global pp_group
    global pp_size
    global pp_rank

    if ring_degree > 1:
        raise NotImplementedError
    elif pp_degree > 1:
        from mindspore.communication import create_group

        # overview
        # rank_id   : 0     , 1     , 2     , 3
        # stage num : stage1, stage1, stage2, stage2
        # sp group  : 0     , 0     , 1     , 1         # comm between same stage
        # sp rank   : 0     , 1     , 0     , 1
        # pp group  : 0     , 1     , 0     , 1         # comm between stage 0 and 1
        # pp rank   : 0     , 0     , 1     , 1
        # create sp group
        sp_group_id = rank_id // ulysses_degree  # 0, 1, 2, 3 -> //2 -> 0, 0, 1, 1
        s_sp_id, e_sp_id = (
            sp_group_id * ulysses_degree,
            (sp_group_id + 1) * ulysses_degree,
        )  # 0, 0, 1, 1 -> *2  -> [0:2], [0:2], [2:4], [2:4]
        sp_comm_group = f"sub_sp_group_{sp_group_id}"
        create_group(sp_comm_group, [_i for _i in range(s_sp_id, e_sp_id)])

        # create pp group
        assert pp_degree == 2
        assert ulysses_degree * pp_degree == world_size
        pp_group_id = rank_id % ulysses_degree  # 0, 1, 2, 3 -> %2  -> 0, 1, 0, 1
        pp_ranks = [_i for _i in range(world_size) if _i % ulysses_degree == pp_group_id]
        pp_comm_group = f"sub_pp_group_{pp_group_id}"
        create_group(pp_comm_group, pp_ranks)

        # set global var
        sp_size = ulysses_degree
        sp_rank = rank_id % ulysses_degree
        sp_group = sp_comm_group

        pp_size = pp_degree
        pp_rank = rank_id // ulysses_degree
        pp_group = pp_comm_group

        print(f"enable custom pipeline parallel, {pp_degree=}, {ulysses_degree=}")

    elif ulysses_degree > 1:
        if ulysses_degree == world_size:
            sp_group = GlobalComm.WORLD_COMM_GROUP
            sp_size = world_size
            sp_rank = rank_id
        else:
            from mindspore.communication import create_group

            g_id = rank_id // ulysses_degree
            s_id, e_id = g_id * ulysses_degree, (g_id + 1) * ulysses_degree
            comm_group = f"sub_sp_group_{g_id}"
            create_group(comm_group, [_i for _i in range(s_id, e_id)])

            sp_size = ulysses_degree
            sp_rank = rank_id % ulysses_degree
            sp_group = comm_group
    else:
        sp_size = 1
        sp_rank = 0
        sp_group = None

    # dist.init_process_group("nccl")
    # xfuser.core.distributed.init_distributed_environment(
    #     rank=dist.get_rank(),
    #     world_size=dist.get_world_size()
    # )
    #
    # xfuser.core.distributed.initialize_model_parallel(
    #     sequence_parallel_degree=dist.get_world_size(),
    #     ring_degree=ring_degree,
    #     ulysses_degree=ulysses_degree,
    # )


# old, w/o pp
def bak_initialize_parall_group(ring_degree=1, ulysses_degree=1):
    global _is_distribute

    world_size = 1
    rank_id = 0
    if ring_degree > 1 or ulysses_degree > 1:
        init()
        world_size = get_group_size()
        rank_id = get_rank()
        print(f"init_environment, rank_id: {rank_id}, world_size: {world_size}")

        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=world_size,
        )

        _is_distribute = True

    global sp_group
    global sp_size
    global sp_rank

    if ring_degree > 1:
        raise NotImplementedError
    elif ulysses_degree > 1:
        if ulysses_degree == world_size:
            sp_group = GlobalComm.WORLD_COMM_GROUP
            sp_size = world_size
            sp_rank = rank_id
        else:
            from mindspore.communication import create_group

            g_id = rank_id // ulysses_degree
            s_id, e_id = g_id * ulysses_degree, (g_id + 1) * ulysses_degree
            comm_group = f"sub_sp_group_{g_id}"
            create_group(comm_group, [_i for _i in range(s_id, e_id)])

            sp_size = ulysses_degree
            sp_rank = rank_id % ulysses_degree
            sp_group = comm_group
    else:
        sp_size = 1
        sp_rank = 0
        sp_group = None

    # dist.init_process_group("nccl")
    # xfuser.core.distributed.init_distributed_environment(
    #     rank=dist.get_rank(),
    #     world_size=dist.get_world_size()
    # )
    #
    # xfuser.core.distributed.initialize_model_parallel(
    #     sequence_parallel_degree=dist.get_world_size(),
    #     ring_degree=ring_degree,
    #     ulysses_degree=ulysses_degree,
    # )


def get_sequence_parallel_world_size():
    # return xfuser.core.distributed.parallel_state.get_sequence_parallel_world_size()
    return sp_size


def get_sequence_parallel_rank():
    # return xfuser.core.distributed.parallel_state.get_sequence_parallel_rank()
    return sp_rank


def get_sp_group():
    # return xfuser.core.distributed.parallel_state.get_sp_group()
    return sp_group


def get_pipeline_parallel_world_size():
    # return xfuser.core.distributed.parallel_state.get_sequence_parallel_world_size()
    return pp_size


def get_pipeline_parallel_rank():
    # return xfuser.core.distributed.parallel_state.get_sequence_parallel_rank()
    return pp_rank


def get_pp_group():
    # return xfuser.core.distributed.parallel_state.get_sp_group()
    return pp_group


def get_pp_split_index():
    # return xfuser.core.distributed.parallel_state.get_sp_group()
    return pp_split_index


def parallel_forward(fn_):
    def wrapTheFunction(_, hidden_states, *args, **kwargs):
        if kwargs["parallel"]:
            hidden_states = mint.chunk(hidden_states, get_sequence_parallel_world_size(), dim=-2)[
                get_sequence_parallel_rank()
            ]
            kwargs["attn_mask"] = mint.chunk(kwargs["attn_mask"], get_sequence_parallel_world_size(), dim=-2)[
                get_sequence_parallel_rank()
            ]
        output = fn_(_, hidden_states, *args, **kwargs)

        if kwargs["parallel"]:
            # output = get_sp_group().all_gather(output.contiguous(), dim=-2)
            output = sp_all_gather(output, dim=-2)

        return output

    return wrapTheFunction


def sp_all_gather(input_: Tensor, dim: int = 0):
    # w/o sp
    if get_sp_group() is None:
        return input_

    # w/ sp
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return input_

    if dim < 0:
        dim += input_.ndim

    input_size = list(input_.shape)
    input_size[0] *= world_size

    # All-gather.
    output_tensor = ops.AllGather(group=get_sp_group())(input_)  # e.g. (2, 8) -> (sp*2, 8)

    if dim != 0:
        input_size[0] //= world_size
        output_tensor = output_tensor.reshape(
            [
                world_size,
            ]
            + input_size
        )
        output_tensor = output_tensor.movedim(0, dim)

    input_size = list(input_.shape)
    input_size[dim] = input_size[dim] * world_size
    # Reshape
    output_tensor = output_tensor.reshape(input_size)
    return output_tensor
