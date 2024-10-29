import logging
import os
import time
from typing import List, Optional

import numpy as np

import mindspore.ops as ops
from mindspore import Tensor
from mindspore.train import Callback, RunContext

from mindone.trainers.checkpoint import CheckpointManager

__all__ = ["LossMonitor", "SaveCkptCallback", "TimeMonitor"]

logger = logging.getLogger(__name__)


class LossMonitor(Callback):
    def __init__(self, log_interval: int = 1, log_overflow: bool = True) -> None:
        self.log_interval = log_interval
        self.log_overflow = log_overflow
        self.step_num = 0

    def on_train_step_begin(self, run_context: RunContext) -> None:
        self.step_num += 1

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        self.step_num = 0

    def on_train_step_end(self, run_context: RunContext) -> None:
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num

        if cur_step % self.log_interval == 0:
            cur_lr = self._fetch_optimizer_lr(cb_params)
            cur_loss = self._fetch_loss(cb_params)
            cur_loss_scale = self._fetch_loss_scale(cb_params)

            logger.info(
                "epoch: %d step: %d, lr: %.7f, loss: %.6f, loss scale: %d.",
                cb_params.cur_epoch_num,
                self.step_num,
                cur_lr.item(),
                cur_loss.item(),
                cur_loss_scale.item(),
            )

            if self.log_overflow:
                overflow = cb_params.net_outputs[1]
                if overflow:
                    logger.warning(f"overflow detected in epoch {cb_params.cur_epoch_num} step {self.step_num}.")

    def _get_optimizer_from_cbp(self, cb_params):
        if cb_params.optimizer is not None:
            optimizer = cb_params.optimizer
        elif cb_params.dataset_sink_mode:
            optimizer = cb_params.train_network.network.optimizer
        else:
            optimizer = cb_params.train_network.optimizer
        return optimizer

    def _fetch_loss_scale(self, cb_params) -> Tensor:
        if cb_params.dataset_sink_mode:
            return cb_params.train_network.network.scale_sense
        else:
            return cb_params.train_network.scale_sense

    def _fetch_optimizer_lr(self, cb_params) -> Tensor:
        opt = self._get_optimizer_from_cbp(cb_params)
        lr = opt.learning_rate
        if opt.dynamic_lr:
            lr = opt.learning_rate(ops.clip(opt.global_step - 1, min=0))[0]
        return lr

    def _fetch_loss(self, cb_params) -> Tensor:
        loss = cb_params.net_outputs[0]
        return loss


class SaveCkptCallback(Callback):
    def __init__(
        self,
        output_dir: str = "./output",
        ckpt_max_keep: int = 5,
        ckpt_save_interval: int = 1,
        rank_id: Optional[int] = None,
    ) -> None:
        self.rank_id = 0 if rank_id is None else rank_id
        if self.rank_id != 0:
            return

        self.ckpt_save_interval = ckpt_save_interval

        ckpt_save_dir = os.path.join(output_dir, f"rank_{rank_id}")
        if not os.path.isdir(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        self.ckpt_manager = CheckpointManager(ckpt_save_dir, ckpt_save_policy="latest_k", k=ckpt_max_keep)

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        if self.rank_id != 0:
            return

        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        if cur_epoch % self.ckpt_save_interval != 0 and cur_epoch != epoch_num:
            return

        ckpt_name = f"epoch_{cur_epoch}.ckpt"
        network = cb_params.train_network.network
        self.ckpt_manager.save(network=network.trainable_params(), ckpt_name=ckpt_name)


class TimeMonitor(Callback):
    def __init__(self) -> None:
        self.epoch_start_time = 0
        self.step_start_time = 0
        self.durations: List[int] = list()

    def on_train_epoch_begin(self, run_context: RunContext) -> None:
        self.epoch_start_time = time.time()

    def on_train_step_begin(self, run_context: RunContext) -> None:
        self.step_start_time = time.time()

    def on_train_step_end(self, run_context: RunContext) -> None:
        duration = time.time() - self.step_start_time
        self.durations.append(duration)

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        epoch_duration = time.time() - self.epoch_start_time
        avg_time = np.mean(self.durations)
        self.durations = list()
        logger.info(f"Total training time for single epoch: {epoch_duration:.3f} seconds")
        logger.info(f"Average step time: {avg_time:.3f} seconds")
