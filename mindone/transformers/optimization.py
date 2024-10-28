from typing import Union, Optional
from mindspore.nn import Optimizer

from .trainer_utils import SchedulerType


def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    scheduler_specific_kwargs: Optional[dict] = None,
):
    # TODO: Add lr scheduler
    return None
