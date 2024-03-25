import logging
import os
from typing import Optional, Tuple

from ldm.data.dataset_dist import split_and_sync_data
from ldm.modules.train.parallel_config import ParallelConfig
from ldm.modules.train.tools import set_random_seed

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

_logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    debug: bool = False,
    seed: int = 42,
    distributed: bool = False,
    device_target: Optional[str] = "Ascend",
    enable_modelarts: bool = False,
    num_workers: int = 1,
    json_data_path: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        debug: Whether to enable debug mode (forces PyNative mode). Default is False.
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
        enable_modelarts: Whether to enable modelarts (OpenI) support. Default is False.
        num_workers: The number of modelarts workers. Used only when `enable_modelarts` is True. Default is 1.
        json_data_path: The path of num_samples.json containing a dictionary with 64 parts. Each part is a large
                        dictionary containing counts of samples of 533 tar packages.
                        Used only when `enable_modelarts` is True.

    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        _logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    if distributed:
        device_id = int(os.getenv("DEVICE_ID"))
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # Only effective on Ascend 910*
        )
        init()
        device_num = get_group_size()
        ParallelConfig.dp = device_num
        rank_id = get_rank()
        _logger.debug(f"Device_id: {device_id}, rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        _logger.info(dict(zip(var_info, var_value)))

        if enable_modelarts:
            split_and_sync_data(json_data_path, num_workers, device_num, rank_id)
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # Only effective on Ascend 910*
            pynative_synchronize=debug,
        )

    return device_id, rank_id, device_num
