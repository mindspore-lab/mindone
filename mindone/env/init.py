import logging
import os
from typing import Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

_logger = logging.getLogger(__name__)


def init_train_env(
    mode: int = ms.GRAPH_MODE,
    device_target: Literal["Ascend", "GPU"] = "Ascend",
    debug: bool = False,
    seed: int = 42,
    cache_graph: bool = False,
    cache_path: str = "./cache",
    distributed: bool = False,
    ascend_config: Optional[dict] = None,
    enable_modelarts: bool = False,
    max_device_memory: str = None,
    num_workers: int = 1,
    json_data_path: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore training environment.

    Args:
        mode: MindSpore execution mode. Options: 0 (ms.GRAPH_MODE), 1 (ms.PYNATIVE_MODE). Default is 0 (ms.GRAPH_MODE).
        device_target: The target execution device. Options: "Ascend", "GPU". Default is "Ascend".
        debug: Whether to enable debug mode (forces PyNative mode). Default is False.
        seed: The seed value for reproducibility. Default is 42.
        cache_graph: (Experimental) Save or load the saved computation graph to significantly reduce the graph
                     compilation time during the first epoch. Use this feature with great caution, as any changes to the
                     Python scripts may cause inconsistencies in the results.
        cache_path: The path to save or load the saved computation graph.
        distributed: Whether to enable distributed training. Default is False.
        ascend_config: Parameters specific to the Ascend hardware platform.
        enable_modelarts: Whether to enable modelarts (OpenI) support. Default is False.
        max_device_memory (str, default: None): The maximum amount of memory that can be allocated on the Ascend device.
        num_workers: The number of modelarts workers. Used only when `enable_modelarts` is True. Default is 1.
        json_data_path: The path of num_samples.json containing a dictionary with 64 parts. Each part is a large
                        dictionary containing counts of samples of 533 tar packages.
                        Used only when `enable_modelarts` is True.

    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    ms.set_seed(seed)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        _logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE
    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)
    if distributed:
        device_id = int(os.getenv("DEVICE_ID"))
        ms.set_context(mode=mode, device_target=device_target, device_id=device_id, ascend_config=ascend_config or {})
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        _logger.debug(f"Device_id: {device_id}, rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, device_num // 8, rank_id // 8]
        _logger.info(dict(zip(var_info, var_value)))

        if enable_modelarts:
            # split_and_sync_data(json_data_path, num_workers, device_num, rank_id)
            raise NotImplementedError("ModelArts is not supported yet.")
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            ascend_config=ascend_config or {},
            pynative_synchronize=debug,
            enable_compile_cache=cache_graph,
            compile_cache_path=cache_path,
        )

    return device_id, rank_id, device_num
