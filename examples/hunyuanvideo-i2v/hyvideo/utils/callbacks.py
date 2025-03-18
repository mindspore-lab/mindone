import logging
import os
import time
from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd

from mindspore import Callback, Parameter, ReduceLROnPlateau, RunContext, Tensor
from mindspore import dtype as mstype
from mindspore import mint, nn, ops
from mindspore.communication import GlobalComm, get_group_size
from mindspore.dataset import BatchDataset, BucketBatchByLengthDataset, GeneratorDataset
from mindspore.ops import functional as F

from mindone.trainers.ema import EMA

__all__ = ["ValidationCallback", "PerfRecorderCallback", "ReduceLROnPlateauByStep"]

_logger = logging.getLogger(__name__)


class ValidationCallback(Callback):
    """
    A callback for performing validation during training on a per-step basis.

    Args:
        network (nn.Cell): The neural network model to be validated.
        dataset (BatchDataset, BucketBatchByLengthDataset, GeneratorDataset): The dataset to use for validation.
        alpha_smooth (float, optional): The smoothing factor for the loss. Defaults to 0.01.
        valid_frequency (int, optional): The frequency of validation in terms of training steps.
                                         Defaults to 100.
        ema (Optional[EMA], optional): An Exponential Moving Average object for the model weights.
                                       If provided, it will be used during validation. Defaults to None.

    Example:
        >>> model = MyModel()
        >>> val_dataset = MyValidationDataset()
        >>> val_callback = ValidationCallback(model, val_dataset, valid_frequency=500)
        >>> model.train(num_epochs, train_dataset, callbacks=[val_callback])
    """

    def __init__(
        self,
        network: nn.Cell,
        dataset: Union[BatchDataset, BucketBatchByLengthDataset, GeneratorDataset],
        alpha_smooth: float = 0.01,
        valid_frequency: int = 100,
        ema: Optional[EMA] = None,
    ):
        super().__init__()
        self.network = network
        self.dataset = dataset
        self.alpha_smooth = alpha_smooth
        self.valid_frequency = valid_frequency
        self.ema = ema
        self.reduce, self.rank_size = None, 1
        if GlobalComm.INITED:
            self.reduce = ops.AllReduce(op=ops.ReduceOp.SUM)
            self.rank_size = get_group_size()
        self.data = pd.Series(dtype=np.float32)

    def on_train_step_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cb_params.eval_results = {}  # Erase previous validation results
        cur_step = cb_params.cur_step_num

        if cur_step % self.valid_frequency == 0:
            if self.ema is not None:
                self.ema.swap_before_eval()
            self.network.set_train(False)

            loss = 0
            for data in self.dataset.create_tuple_iterator(num_epochs=1):
                loss += self.network(*data)
            loss = loss / self.dataset.get_dataset_size()
            if self.reduce is not None:
                loss = self.reduce(loss)
            loss = loss.item() / self.rank_size

            self.data = pd.concat([self.data, pd.Series(loss)], ignore_index=True)
            loss_smoothed = self.data.ewm(alpha=self.alpha_smooth).mean().iloc[-1]

            cb_params.eval_results = {"eval_loss": loss, "eval_loss_smoothed": loss_smoothed}
            _logger.info(f"Step: {cur_step}, Validation Loss: {loss}.")

            self.network.set_train(True)
            if self.ema is not None:
                self.ema.swap_after_eval()


class PerfRecorderCallback(Callback):
    """
    Improved version of `mindone.trainers.recorder.PerfRecorder` that tracks validation metrics as well.
    Used here first for testing.
    """

    def __init__(
        self,
        save_dir: str,
        file_name: str = "result.log",
        metric_names: List[str] = None,
        separator: str = "\t",
    ):
        super().__init__()
        self._sep = separator
        self._metrics = metric_names or []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self._log_file = os.path.join(save_dir, file_name)

        header = separator.join([f"{'step':<7}", f"{'loss':<10}", "train_time(s)"] + self._metrics)
        with open(self._log_file, "w", encoding="utf-8") as fp:
            fp.write(header + "\n")

    def on_train_step_begin(self, run_context: RunContext):
        self._step_time = time.perf_counter()

    def on_train_step_end(self, run_context: RunContext):
        step_time = time.perf_counter() - self._step_time
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        loss = cb_params.net_outputs
        loss = loss[0].asnumpy() if isinstance(loss, tuple) else np.mean(loss.asnumpy())
        eval_loss = cb_params.get("eval_results", [])
        metrics = (self._sep + self._sep.join([f"{eval_loss[m]:.6f}" for m in self._metrics])) if eval_loss else ""

        with open(self._log_file, "a", encoding="utf-8") as fp:
            fp.write(
                self._sep.join([f"{cur_step:<7}", f"{loss.item():<10.6f}", f"{step_time:<13.3f}"]) + metrics + "\n"
            )


class ReduceLROnPlateauByStep(ReduceLROnPlateau):
    """
    Extends ReduceLROnPlateau to reduce the learning rate at the end of a step and incorporates loss smoothing.
    """

    def __init__(
        self,
        optimizer,
        monitor: str = "eval_loss_smoothed",
        factor: float = 0.1,
        patience: int = 10,
        mode: Literal["auto", "min", "max"] = "auto",
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
    ):
        super().__init__(monitor, factor, patience, mode=mode, min_delta=min_delta, cooldown=cooldown, min_lr=min_lr)
        self.optimizer = optimizer
        self.min_lr = Tensor(self.min_lr, dtype=mstype.float32)

    def on_train_step_end(self, run_context):
        """
        monitors the training process and if no improvement is seen for a 'patience' number
        of epochs, the learning rate is reduced.

        Copy of the original `on_train_step_end()` with changes to add loss alpha smoothing.

        Args:
            run_context (RunContext): Context information of the model. For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        cb_params = run_context.original_args()
        cur_step = cb_params.cur_step_num
        lrs = self.optimizer.learning_rate.learning_rate
        if not isinstance(lrs, Parameter):
            raise ValueError("ReduceLROnPlateau does not support dynamic learning rate and group learning rate now.")

        current_monitor_value = cb_params.get("eval_results")
        if current_monitor_value:
            current_monitor_value = current_monitor_value[self.monitor]

            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                self.wait = 0

            if self.is_improvement(current_monitor_value, self.best):
                self.best = current_monitor_value
                self.wait = 0
            elif self.cooldown_counter <= 0:
                self.wait += 1
                if self.wait >= self.patience:
                    if lrs[cur_step] > self.min_lr:  # FIXME: doesn't hold for future LRs
                        new_lr = lrs * self.factor
                        min_lr = mint.tile(self.min_lr, lrs.shape)
                        new_lr = mint.where(new_lr < min_lr, min_lr, new_lr)
                        F.assign(self.optimizer.learning_rate.learning_rate, new_lr)
                        _logger.info(f"Step {cur_step}: reducing learning rate to {new_lr[cur_step]}.")
                    self.cooldown_counter = self.cooldown
                    self.wait = 0

    def on_train_epoch_end(self, run_context):
        # Use `on_train_step_end` instead
        pass
