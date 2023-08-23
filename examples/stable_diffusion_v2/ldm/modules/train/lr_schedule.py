# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Learning Rate Scheduler Factory"""
import logging

from .dynamic_lr import cosine_decay_refined_lr, linear_refined_lr, multi_step_lr, polynomial_refined_lr

_logger = logging.getLogger(__name__)


def create_scheduler(
    steps_per_epoch: int,
    scheduler: str = "constant",
    lr: float = 0.01,
    min_lr: float = 1e-6,
    warmup_steps: int = 100,
    warmup_factor: float = 0.0,
    decay_steps: int = 100,
    decay_rate: float = 0.9,
    milestones: list = None,
    num_epochs: int = 20,
    num_cycles: int = 1,
    cycle_decay: float = 1.0,
):
    r"""Creates learning rate scheduler by name.

    Args:
        steps_per_epoch: number of steps per epoch.
        scheduler: scheduler name like 'constant', 'cosine_decay',
            'polynomial_decay', 'multi_step_decay'. Default: 'constant'.
        lr: learning rate value. Default: 0.01.
        min_lr: lower lr bound for 'cosine_decay' schedulers. Default: 1e-6.
        warmup_steps: steps to warmup LR, if scheduler supports. Default: 100.
        warmup_factor: the warmup phase of scheduler is a linearly increasing lr,
            the beginning factor is `warmup_factor`, i.e., the lr of the first step is lr*warmup_factor,
            and the ending lr in the warmup phase is lr. Default: 0.0
        decay_steps: decay LR to min_lr in `decay_steps`. Default: 100.
        decay_rate: LR decay rate (default: 0.9)
        milestones: list of steps milestones for 'multi_step_decay' scheduler. Must be increasing.
        num_epochs: number of total epochs.
    Returns:
        A list of float numbers indicating the learning rate at every step
    """
    # check params
    if milestones is None:
        milestones = []

    num_steps = num_epochs * steps_per_epoch
    if warmup_steps + decay_steps > num_steps:
        _logger.warning("warmup_steps + decay_steps > num_steps. Please check and reduce warmup_steps or decay_steps!")

    # lr warmup phase
    warmup_lr_scheduler = []
    if warmup_steps > 0:
        warmup_lr_scheduler = linear_refined_lr(
            start_factor=warmup_factor,
            end_factor=1.0,
            warmup_steps=warmup_steps,
            lr=lr,
            total_steps=warmup_steps,
        )

    # lr decay phase
    main_steps = num_steps - warmup_steps
    if scheduler == "cosine_decay":
        main_lr_scheduler = cosine_decay_refined_lr(
            decay_steps=decay_steps,
            eta_min=min_lr,
            eta_max=lr,
            total_steps=main_steps,
            num_cycles=num_cycles,
            cycle_decay=cycle_decay,
        )
    elif scheduler == "polynomial_decay":
        main_lr_scheduler = polynomial_refined_lr(
            decay_steps=decay_steps, power=decay_rate, lr=lr, end_lr=min_lr, total_steps=main_steps
        )
    elif scheduler == "multi_step_decay":
        main_lr_scheduler = multi_step_lr(milestones=milestones, gamma=decay_rate, lr=lr, total_steps=main_steps)
    elif scheduler == "constant":
        main_lr_scheduler = [lr for _ in range(main_steps)]
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")

    # combine
    lr_scheduler = warmup_lr_scheduler + main_lr_scheduler

    return lr_scheduler
