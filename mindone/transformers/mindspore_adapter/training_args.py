import os
from dataclasses import dataclass, field
from typing import Optional

import mindspore as ms
from mindspore.communication.management import get_group_size, get_rank, init


@dataclass
class MindSporeArguments:
    # for mindspore

    mode: int = field(default=ms.GRAPH_MODE, metadata={"help": "Graph/Pynative"})

    jit_level: Optional[str] = field(default="O0", metadata={"help": ("jit level")})

    device_target: str = field(default="Ascend", metadata={"help": "Ascend/GPU/CPU"})

    is_distribute: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to use PyTorch Fully Sharded Data Parallel (FSDP) training (in distributed training"
                " only). The base option should be `full_shard`, `shard_grad_op` or `no_shard` and you can add"
                " CPU-offload to `full_shard` or `shard_grad_op` like this: full_shard offload` or `shard_grad_op"
                " offload`. You can add auto-wrap to `full_shard` or `shard_grad_op` with the same syntax: full_shard"
                " auto_wrap` or `shard_grad_op auto_wrap`."
            ),
        },
    )
    rank: int = field(default=0, metadata={"help": "rank id"})
    rank_size: int = field(default=1, metadata={"help": "device num"})

    enable_flash_attention: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "if enable_flash_attention is True, model attention implementation will be set to `flash_attention_2`"
            )
        },
    )

    adamw_enable_fuse: Optional[bool] = field(
        default=True,
        metadata={"help": ("enable fuse op")},
    )
    adamw_zero_shard_size: Optional[int] = field(
        default=None,
        metadata={"help": ("setting zero parallelism shard size")},
    )
    max_device_memory: Optional[str] = field(
        default=None,
        metadata={"help": ("max device memory")},
    )

    precision_mode: Optional[str] = field(
        default="must_keep_origin_dtype", metadata={"help": ("global precision_mode")}
    )


def init_environment(training_args: MindSporeArguments):
    # FIXME, stream synchronize bug when jit_level is `O0` on MindSpore 2.3.0
    if training_args.mode == 0:
        if os.environ.get("MS_DEV_RUNTIME_CONF") is None:
            os.environ["MS_DEV_RUNTIME_CONF"] = "synchronize:True"
            print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")
        else:
            if "synchronize:True" not in os.environ.get("MS_DEV_RUNTIME_CONF"):
                _old = os.environ.get("MS_DEV_RUNTIME_CONF")
                _old.replace("synchronize:False,", "")
                _old.replace(",synchronize:False", "")
                _old.replace("synchronize:False", "")
                _new = "synchronize:True," + _old if len(_old) > 0 else "synchronize:True"
                os.environ["MS_DEV_RUNTIME_CONF"] = _new
                print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")

    # set mindspore context
    ms.set_context(
        mode=training_args.mode,
        device_target=training_args.device_target,
        jit_config={"jit_level": training_args.jit_level},
        deterministic="ON",
        pynative_synchronize=True,
        memory_optimize_level="O1",
        # jit_syntax_level=ms.STRICT
    )

    if training_args.mode == ms.PYNATIVE_MODE:
        print("WARNING: run pynative mode, set `pynative_synchronize` True")

    if training_args.max_device_memory is not None:
        ms.set_context(max_device_memory=training_args.max_device_memory)

    if training_args.precision_mode is not None:
        ms.set_context(
            ascend_config={"precision_mode": training_args.precision_mode},
        )

    if training_args.is_distribute:
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

        training_args.rank = rank_id
        training_args.rank_size = world_size
    else:
        training_args.rank = 0
        training_args.rank_size = 1
