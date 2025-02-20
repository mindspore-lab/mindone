import logging

import mindspore as ms
from mindspore.communication.management import get_group_size, get_rank, init

from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    mempool_block_size: str = "9GB",
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    jit_level: str = "O0",
    strategy_ckpt_save_file: str = "",
    optimizer_weight_shard_size: int = 8,
    debug: bool = False,
    dtype: ms.dtype = ms.float32,
):
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)
    ms.set_context(mempool_block_size=mempool_block_size)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},
        )

        if parallel_mode == "optim":
            logger.info("use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                parallel_optimizer_config={"optimizer_weight_shard_size": optimizer_weight_shard_size},
                enable_parallel_optimizer=True,
                strategy_ckpt_config={
                    "save_file": strategy_ckpt_save_file,
                    "only_trainable_params": False,
                },
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()

        elif parallel_mode == "data":
            init()
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            pynative_synchronize=debug,
        )

    if jit_level in ["O0", "O1", "O2"]:
        ms.set_context(jit_config={"jit_level": jit_level}, jit_syntax_level=ms.LAX)
    else:
        logger.warning(f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method")

    if dtype == ms.bfloat16:
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})
        logger.info("Using precision_mode: allow_mix_precision_bf16")
    elif dtype == ms.float16:
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_fp16"})
        logger.info("Using precision_mode: allow_mix_precision_fp16")
    else:
        ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})

    return rank_id, device_num
