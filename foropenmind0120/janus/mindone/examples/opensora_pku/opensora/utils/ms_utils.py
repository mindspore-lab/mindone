import logging
import os
from typing import Tuple

from opensora.acceleration.parallel_states import initialize_sequence_parallel_state

import mindspore as ms
from mindspore import _no_grad
from mindspore.communication.management import get_group_size, get_rank, init

from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    device_id: int = None,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    mempool_block_size: str = "9GB",
    global_bf16: bool = False,
    strategy_ckpt_save_file: str = "",
    optimizer_weight_shard_size: int = 8,
    sp_size: int = 1,
    jit_level: str = None,
    enable_parallel_fusion: bool = False,
    precision_mode: str = None,
    jit_syntax_level: str = "strict",
    comm_fusion=False,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode (int, default 0): MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed (int, default 42): The seed value for reproducibility. Default is 42.
        distributed (bool, False): Whether to enable distributed training. Default is False.
        device_id (int, default None): If distributed is False (single-device), and if device_id is provided, \
            use the device_id to set the device index; if not provided, wil get os.environ["DEVICE_ID"]
        max_device_memory (str, default None): the maximum available device memory, e.g., "30GB" for Ascend 910A or "59GB" for Ascend 910B.
        device_target (str, default "Ascend"): the target device type, supporting "Ascend" or "GPU"
        parallel_mode (str, default "data"): if `distributed` is True, `parallel_mode` will be one of ["data", "optim"]
        mempool_block_size (str, default "9GB"): Set the size of the memory pool block in PyNative mode for devices. \
            The format is “xxGB”. Default: “1GB”. Minimum size is “1G”.
        global_bf16 (bool, default False): Whether to use global_bf16 in GE mode (jit_level="O2").
        strategy_ckpt_save_file (str, default None): The path to strategy_ckpt when parallel_mode == "optim". \
            This strategy_ckpt is useful for merging multiple checkpoint shards.
        optimizer_weight_shard_size (int, default 8): Set the size of the communication domain split by the optimizer \
            weight when parallel_mode == "optim". The numerical range can be (0, device_num].
        sp_size (int, default 1): Set the sequence parallel size. Default is 1. The device_num should be >= sp_size \
            and device_num should be divisble by sp_size.
        jit_level (str, default None): If set, will set the compilation optimization level. Supports ["O0", "O1", "O2"]. \
            "O1" means KernelByKernel (KBK) mode, "O2" means DVM mode, and "O3" means GE mode.
        enable_parallel_fusion (bool, default None): If True, will enable optimizer parallel fusion for AdamW.
        precision_mode (str, default None): If provided, will set precision_mode to overwrite the default option "allow_fp32_to_fp16".
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)
    ms.set_context(mempool_block_size=mempool_block_size)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)
    if enable_parallel_fusion:
        ms.set_context(graph_kernel_flags="--enable_parallel_fusion --enable_expand_ops=AdamApplyOneWithDecayAssign")

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
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
            logger.info("use data parallel")
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )
        elif parallel_mode == "zero":
            init()
            logger.info("use parallelism like deepspeed")
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
            )
            if comm_fusion:
                comm_fusion_dict = {
                    "allreduce": {"mode": "auto", "config": None},
                    "reducescatter": {"mode": "auto", "config": None},
                    "allgather": {"mode": "auto", "config": None},
                }
                ms.set_auto_parallel_context(comm_fusion=comm_fusion_dict)

        else:
            raise ValueError(f"{parallel_mode} not supported!")

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        device_id = device_id if device_id is not None else int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
        )

    if jit_level is not None:
        if mode == 1:
            logger.info(f"Only graph mode supports jit_level! Will ignore jit_level {jit_level} in Pynative mode.")
        else:
            try:
                if jit_level in ["O0", "O1", "O2"]:
                    logger.info(f"Using jit_level: {jit_level}")
                    ms.context.set_context(jit_config={"jit_level": jit_level})  # O0: KBK, O1:DVM, O2: GE
                else:
                    logger.warning(
                        f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method"
                    )
            except Exception:
                logger.warning(
                    "The current jit_level is not suitable because current MindSpore version does not match,"
                    "please upgrade the MindSpore version."
                )
                raise Exception

    if mode == 0:
        # graph mode apply jit_syntax_level
        jit_syntax_level = ms.STRICT if jit_syntax_level == "strict" else ms.LAX
        ms.set_context(jit_syntax_level=jit_syntax_level)
    if precision_mode is not None and len(precision_mode) > 0:
        ms.set_context(ascend_config={"precision_mode": precision_mode})
    if global_bf16:
        logger.info("Using global bf16")
        assert jit_level is not None and jit_level == "O2", "global_bf16 is supported in GE mode only!"
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})

    assert device_num >= sp_size and device_num % sp_size == 0, (
        f"unable to use sequence parallelism, " f"device num: {device_num}, sp size: {sp_size}"
    )
    initialize_sequence_parallel_state(sp_size)
    return rank_id, device_num


@ms.jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)
