import logging
import os
import time
from typing import List, Optional

import numpy as np
import tqdm
from pixart.pipelines import PixArtInferPipeline

import mindspore.ops as ops
from mindspore import Parameter, ParameterTuple, RunContext, Tensor
from mindspore.train import Callback

from mindone.trainers.checkpoint import CheckpointManager

from .misc import organize_prompts
from .plot import create_save_func

__all__ = ["LossMonitor", "SaveCkptCallback", "TimeMonitor", "Visualizer", "TurnOffVAET5Train"]

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
        save_ema: bool = False,
        rank_id: Optional[int] = None,
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

        if cur_epoch % self.ckpt_save_interval != 0 and cur_epoch != epoch_num:
            return

        ckpt_name = f"epoch_{cur_epoch}.ckpt"
        network = cb_params.train_network.network
        self.ckpt_manager.save(network=network.trainable_params(), ckpt_name=ckpt_name)
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


class Visualizer(Callback):
    def __init__(
        self,
        infer_pipeline: PixArtInferPipeline,
        sample_size: int,
        validation_prompts: List[str],
        validation_negative_prompts: Optional[List[str]] = None,
        visualize_dir: str = "./output",
        visualize_interval: int = 1,
        rank_id: Optional[int] = None,
    ) -> None:
        self.infer_pipeline = infer_pipeline
        self.visualize_interval = visualize_interval

        # prepare the noise, keep it is same during whole training.
        # To save memory, inference one image at each time.
        self.visualize_dir = os.path.join(visualize_dir, f"rank_{rank_id}")
        self.noise = Tensor(
            np.random.default_rng(rank_id).standard_normal((1, 4, sample_size, sample_size), dtype=np.float32)
        )

        if not os.path.isdir(self.visualize_dir):
            os.makedirs(self.visualize_dir)

        self.prompts = organize_prompts(
            prompts=validation_prompts,
            negative_prompts=validation_negative_prompts,
            save_json=True,
            output_dir=self.visualize_dir,
        )

    def on_train_epoch_end(self, run_context: RunContext) -> None:
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        epoch_num = cb_params.epoch_num

        if cur_epoch % self.visualize_interval != 0 and cur_epoch != epoch_num:
            return

        assert not self.infer_pipeline.vae.training
        assert not self.infer_pipeline.text_encoder.training
        self.infer_pipeline.network.set_train(False)

        outputs = list()
        for record in tqdm.tqdm(self.prompts):
            output = self.infer_pipeline(self.noise, record["prompt"], record["negative_prompt"]).asnumpy()
            outputs.append(output)

        visualize_epoch_dir = os.path.join(self.visualize_dir, f"epoch_{cur_epoch}")
        save = create_save_func(output_dir=visualize_epoch_dir, imagegrid=False)
        for sample in outputs:
            save(sample)

        self.infer_pipeline.network.set_train(True)


class TurnOffVAET5Train(Callback):
    def on_train_begin(self, run_context: RunContext) -> None:
        cb_params = run_context.original_args()
        assert cb_params.train_network.network.network.training
        cb_params.train_network.network.vae.set_train(False)
        cb_params.train_network.network.text_encoder.set_train(False)
