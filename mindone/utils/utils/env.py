import logging
import os
from typing import Literal, Optional, Union

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from .version_control import MS_VERSION

_logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    device_target: Literal["Ascend", "GPU"] = "Ascend",
    debug: bool = False,
    seed: int = 42,
    cache_graph: bool = False,
    cache_path: str = "./cache",
    distributed: bool = False,
    precision_mode: Optional[
        Literal[
            "force_fp16",
            "allow_fp32_to_fp16",
            "allow_mix_precision",
            "must_keep_origin_dtype",
            "force_fp32",
            "allow_fp32_to_bf16",
            "allow_mix_precision_fp16",
            "allow_mix_precision_bf16",
        ]
    ] = None,
    jit_level: Optional[Literal["O0", "O1", "O2"]] = None,
    max_device_memory: str = None,
) -> tuple[Union[int, None], int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Options: 0 (ms.GRAPH_MODE), 1 (ms.PYNATIVE_MODE). Default is 0 (ms.GRAPH_MODE).
        device_target: The target execution device. Options: "Ascend", "GPU". Default is "Ascend".
        debug: Whether to enable debug mode (forces PyNative mode). Default is False.
        seed: The seed value for reproducibility. Default is 42.
        cache_graph: (Experimental) Save or load the saved computation graph to significantly reduce the graph
                     compilation time during the first epoch. Use this feature with great caution, as any changes to the
                     Python scripts may cause inconsistencies in the results.
        cache_path: The path to save or load the saved computation graph.
        distributed: Whether to enable distributed execution. Default is False.
        precision_mode: Configure mixed precision mode setting. Default is specific to hardware. For more details, please refer
                `here <https://www.mindspore.cn/docs/en/r2.5.0/api_python/device_context/mindspore.device_context.ascend.op_precision.precision_mode.html>`__.
        jit_level: The compilation optimization level. Options: "O0", "O1", "O2".
                   Default is None and the level selected based on the device.
        max_device_memory (str, default: None): The maximum amount of memory that can be allocated on the Ascend device.

    Returns:
        A tuple containing the device ID, rank ID, and number of devices.
    """
    ms.set_seed(seed)
    device_id = os.getenv("DEVICE_ID", None)
    if device_id is not None:
        device_id = int(device_id)

    context_kwargs = dict(mode=mode)
    if MS_VERSION >= "2.5.0":
        if cache_graph:
            os.environ["MS_COMPILER_CACHE_ENABLE"] = "1"
            os.environ["MS_COMPILER_CACHE_PATH"] = cache_path

        ms.set_device(device_target, device_id=device_id)
        if max_device_memory:
            ms.set_memory(max_size=max_device_memory)
        if precision_mode:
            ms.device_context.ascend.op_precision.precision_mode(precision_mode)
        if debug:
            if mode == ms.GRAPH_MODE:
                _logger.warning("Debug mode is on, switching execution mode to PyNative.")
            context_kwargs.update(mode=ms.PYNATIVE_MODE)
            ms.runtime.launch_blocking()
    else:
        context_kwargs.update(device_target=device_target)
        if device_id is not None:
            context_kwargs.update(device_id=device_id)
        if max_device_memory:
            context_kwargs.update(max_device_memory=max_device_memory)
        if precision_mode:
            context_kwargs.update(ascend_config={"precision_mode": precision_mode})
        if debug:
            if mode == ms.GRAPH_MODE:
                _logger.warning("Debug mode is on, switching execution mode to PyNative.")
            context_kwargs.update(mode=ms.PYNATIVE_MODE, pynative_synchronize=debug)
        if cache_graph:
            context_kwargs.update(enable_compile_cache=cache_graph, compile_cache_path=cache_path)

    if jit_level:
        if MS_VERSION >= "2.3":
            context_kwargs.update(jit_config={"jit_level": jit_level})
        else:
            _logger.warning("Compilation optimization (JIT Level) is supported only in MindSpore 2.3 or later.")

    ms.set_context(**context_kwargs)

    rank_id, device_num = 0, 1
    if distributed:
        init()
        rank_id, device_num = get_rank(), get_group_size()
        _logger.debug(f"{device_id=}, {rank_id=}, {device_num=}")
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num
        )
        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, device_num // 8, rank_id // 8]
        _logger.info(dict(zip(var_info, var_value)))

    return device_id, rank_id, device_num
