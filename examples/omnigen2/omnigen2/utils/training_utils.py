import logging
from math import sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

import numpy as np
from matplotlib import pyplot as plt

import mindspore as ms
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.mint import distributed as dist
from mindspore.parallel._utils import _get_parallel_mode

from mindone.trainers.zero import ZeroHelper, prepare_ema, prepare_network
from mindone.transformers.mindspore_adapter.train_onestep_wrapper import TrainOneStepWrapper

if TYPE_CHECKING:
    from mindspore import nn

_logger = logging.getLogger(__name__)


def init_env(
    debug: bool = False,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: Optional[str] = None,
    device_specific_seed: bool = False,
) -> tuple[int, int]:
    """
    Initialize MindSpore environment.

    Args:
        debug: Whether to enable debug mode. Default is False.
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed execution. Default is False.
        max_device_memory: The maximum amount of memory that can be allocated on the Ascend device.
        device_specific_seed: Whether to differ the seed on each device slightly with `local_rank`.

    Returns:
        A tuple containing the local rank and world size.
    """
    if max_device_memory:
        ms.set_memory(max_size=max_device_memory)
    if debug:
        ms.runtime.launch_blocking()

    local_rank, world_size = 0, 1
    if distributed:
        dist.init_process_group()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL  # , gradients_mean=True, device_num=device_num)
        )
        local_rank, world_size = dist.get_rank(), dist.get_world_size()

    if device_specific_seed:
        ms.set_seed(seed + local_rank)  # set different seeds per NPU for sampling different timesteps
        # keep MS.dataset's seed consistent as datasets first shuffled and then distributed
        ms.dataset.set_seed(seed)
    else:
        ms.set_seed(seed)

    return local_rank, world_size


# Copy of `mindone.trainers.zero.prepare_train_network` but with a replaced `TrainOneStepWrapper`
def prepare_train_network(
    network: "nn.Cell",
    optimizer: "nn.Optimizer",
    scale_sense: float = 1.0,
    ema: Optional["nn.Cell"] = None,
    drop_overflow_update: bool = True,
    gradient_accumulation_steps: int = 1,
    clip_grad: bool = False,
    clip_norm: float = 1.0,
    zero_stage: Literal[0, 1, 2, 3] = 0,
    optimizer_offload: bool = False,
    optimizer_parallel_group: Optional[str] = None,
    dp_group: Optional[str] = None,
    comm_fusion: Optional[dict] = None,
    parallel_modules=None,
) -> TrainOneStepWrapper:
    """
    Prepare network and optimizer for distributed training.

    Args:
        network (`nn.Cell`): train network, not include grad function,
            grad function must be built after rewrite train network.
        optimizer (`nn.Optimizer`): Must be the subclass of MindSpore Optimizer.
        scale_sense (Union[Tensor, Cell]): If this value is a Cell, it will be called
            to update loss scale. If this value is a Tensor, the loss scale can be modified by `set_sense_scale`,
            the shape should be :math:`()` or :math:`(1,)`.
        zero_stage (`int`, *optional*): Stage setting of ZeRO, default is 0.
        optimizer_offload (`bool`, *optional*): Only take effect when optimizer is AdamWeightDecay, default is False.
        optimizer_parallel_group (`str`, *optional*): The name of the optimizer parallel communication group, default is None.
        dp_group (`str`, *optional*): The name of the data parallel communication group, default is None.
        comm_fusion (`dict`, *optional*): A dict contains the types and configurations
            for setting the communication fusion, default is None, turn off the communication fusion. If set a dict,
            turn on the communication fusion.
            Examples: {"allreduce": {"openstate": True, "bucket_size": 5e8},
                       "reduce_scatter": {"openstate": True, "bucket_size": 5e8},
                       "allgather": {"openstate": False, "bucket_size": 5e8},}
        parallel_modules (`dict`, *optional*): A dict of Cells could split parameters in zero3, default is None.
            If None, use `PARALLEL_MODULES` from `mindone.models.modules.parallel`.
    """
    if zero_stage not in [0, 1, 2, 3]:
        raise ValueError("Not support zero_stage {zero_stage}")
    if optimizer_parallel_group is None:
        _logger.warning("Not set zero group, set it WORLD_COMM_GROUP.")
        optimizer_parallel_group = GlobalComm.WORLD_COMM_GROUP
    if optimizer_parallel_group != GlobalComm.WORLD_COMM_GROUP and dp_group is None:
        raise ValueError(
            "optimizer_parallel_group {optimizer_parallel_group} and dp_group {dp_group} not full network hccl group coverage"
        )

    is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
    if not is_parallel and zero_stage == 0:
        _logger.info("No need prepare train_network with zero.")
        zero_helper = None
    else:
        network = prepare_network(network, zero_stage, optimizer_parallel_group, parallel_modules=parallel_modules)
        zero_helper = ZeroHelper(
            optimizer, zero_stage, optimizer_parallel_group, dp_group, optimizer_offload, comm_fusion
        )

    if ema is not None:
        ema = prepare_ema(ema, zero_stage, optimizer_parallel_group)
    train_network = TrainOneStepWrapper(
        network,
        optimizer,
        ema,
        drop_overflow_step=drop_overflow_update,
        scaler="none",
        scaler_config={"scale_value": scale_sense},
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad="norm" if clip_grad else "none",
        clip_value=clip_norm,
        zero_helper=zero_helper,
    )
    return train_network


def log_time_distribution(transport, args, output_dir, rank: int = 0, world_size: int = 1):
    """Samples time steps from transport and plots their distribution."""
    dummy_tensor = ms.tensor(
        np.random.randn(64, 16, int(sqrt(args.data.max_output_pixels) / 8), int(sqrt(args.data.max_output_pixels) / 8)),
        dtype=ms.float32,
    )
    ts = np.concatenate([transport.sample(dummy_tensor, rank, world_size)[0].asnumpy() for _ in range(1000)])

    percentile_70 = np.percentile(ts, 70)

    plt.figure(figsize=(10, 6))
    plt.hist(ts, bins=50, edgecolor="black", alpha=0.7, label="Time Step Distribution")
    plt.axvline(
        percentile_70, color="red", linestyle="dashed", linewidth=2, label=f"70th Percentile = {percentile_70:.2f}"
    )
    plt.title("Distribution of Sampled Time Steps (t)")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = Path(output_dir) / "t_distribution.png"
    plt.savefig(save_path)
    plt.close()
    _logger.info(f"Time step distribution plot saved to {save_path}")
