import logging
import os
import time
from typing import List, Optional

import numpy as np

from mindspore import Callback, RunContext, nn, ops
from mindspore.communication import GlobalComm
from mindspore.dataset import GeneratorDataset

from mindone.trainers.ema import EMA

__all__ = ["ValidationCallback", "PerfRecorderCallback"]

_logger = logging.getLogger(__name__)


class ValidationCallback(Callback):
    """
    A callback for performing validation during training on a per-step basis.

    Args:
        network (nn.Cell): The neural network model to be validated.
        dataset (GeneratorDataset): The dataset to use for validation.
        rank_id (int): The rank ID of the current process. Defaults to 0.
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
        dataset: GeneratorDataset,
        rank_id: int = 0,
        valid_frequency: int = 100,
        ema: Optional[EMA] = None,
    ):
        super().__init__()
        self.network = network
        self.dataset = dataset
        self.rank_id = rank_id
        self.valid_frequency = valid_frequency
        self.ema = ema
        self.reduce = ops.AllReduce() if GlobalComm.INITED else None

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
            loss = loss.item()

            cb_params.eval_results = {"eval_loss": loss}
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
        resume: bool = False,
    ):
        super().__init__()
        self._sep = separator
        self._metrics = metric_names or []

        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        self._log_file = os.path.join(save_dir, file_name)
        if not resume:
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
