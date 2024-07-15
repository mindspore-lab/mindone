import logging
import os
import time
from typing import List, Optional

import numpy as np

import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, RunContext, Tensor
from mindspore.train import Callback

from mindone.trainers.checkpoint import CheckpointManager

__all__ = ["LossMonitor", "SaveCkptCallback", "TimeMonitor"]

logger = logging.getLogger(__name__)


class LossMonitor(Callback):
    def __init__(self, log_interval: int = 1) -> None:
        self.log_interval = log_interval

    def on_train_step_end(self, run_context: RunContext) -> None:
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        step_num = cb_params.batch_num * cb_params.epoch_num

        if (cur_step % self.log_interval == 0) or (cur_step == step_num):
            cur_lr = self._fetch_optimizer_lr(cb_params)
            cur_loss = self._fetch_loss(cb_params)
            cur_loss_scale = self._fetch_loss_scale(cb_params)

            logger.info(
                "epoch: %d step: %d, lr: %.7f, loss: %.6f, loss scale: %d.",
                cb_params.cur_epoch_num,
                (cb_params.cur_step_num - 1) % cb_params.batch_num + 1,
                cur_lr.item(),
                cur_loss.item(),
                cur_loss_scale.item(),
            )

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
        rank_id: Optional[int] = None,
        output_dir: str = "./output",
        ckpt_max_keep: int = 5,
        ckpt_save_interval: int = 1,
        save_ema: bool = False,
    ) -> None:
        self.rank_id = 0 if rank_id is None else rank_id
        if self.rank_id != 0:
            return

        self.ckpt_save_interval = ckpt_save_interval
        self.save_ema = save_ema

        ckpt_save_dir = os.path.join(output_dir, f"rank_{rank_id}")
        if not os.path.isdir(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        self.ckpt_manager = CheckpointManager(ckpt_save_dir, ckpt_save_policy="latest_k", k=ckpt_max_keep)

        if self.save_ema:
            self.ema_ckpt_manager = CheckpointManager(ckpt_save_dir, ckpt_save_policy="latest_k", k=ckpt_max_keep)

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        if self.rank_id != 0:
            return

        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        if (cur_epoch % self.ckpt_save_interval == 0) or (cur_epoch == epoch_num):
            ckpt_name = f"epoch_{cur_epoch}.ckpt"
            network_weight = cb_params.train_network.network
            self.ckpt_manager.save(network=network_weight, ckpt_name=ckpt_name)
            if self.save_ema:
                ckpt_name = f"epoch_{cur_epoch}_ema.ckpt"
                ema_weight = self._drop_ema_prefix(cb_params.train_network.ema.ema_weight)
                self.ema_ckpt_manager.save(network=ema_weight, ckpt_name=ckpt_name)

    def _drop_ema_prefix(self, weight: ParameterTuple) -> List[Parameter]:
        new_weight = list()
        for x in weight:
            x.name = x.name.replace("ema.", "")
            new_weight.append(x)
        return new_weight


class TimeMonitor(Callback):
    def __init__(self):
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
        logger.info(f"Total training time for single epoch: {epoch_duration:.3f} seconds")
        logger.info(f"Average step time: {avg_time:.3f} seconds")
