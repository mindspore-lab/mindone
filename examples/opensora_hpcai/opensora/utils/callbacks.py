import logging
import os
import time
from typing import Callable, List, Optional, Union

import numpy as np

from mindspore import Tensor
from mindspore.train import Callback, RunContext

from mindone.diffusers.utils.mindspore_utils import pynative_context

from ..pipelines.train_pipeline_v2 import no_grad
from ..utils.ema import EMA, EMA_

_logger = logging.getLogger(__name__)


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

        with open(self._log_file, "a", encoding="utf-8") as fp:
            fp.write(self._sep.join([f"{cur_step:<7}", f"{loss.item():<10.6f}", f"{step_time:<13.3f}"]) + "\n")

    def on_eval_end(self, run_context: RunContext):
        cb_params = run_context.original_args()
        cur_step = "-"
        step_time = "-"
        loss = "-"

        metrics = cb_params.metrics
        try:
            metrics = self._sep.join([f"{metrics[m]:.4f}" for m in self._metrics])
        except KeyError:
            raise KeyError(f"Metric ({self._metrics}) not found in eval result ({list(metrics.keys())}).")
        with open(self._log_file, "a", encoding="utf-8") as fp:
            fp.write(f"{cur_step:<8}{self._sep}{loss:<10}{self._sep}{step_time:<8}{self._sep}{metrics}\n")


class EMAEvalSwapCallback(Callback):
    """
    Callback that loads ema weights for model evaluation.
    """

    def __init__(self, ema: Optional[Union[EMA, EMA_]] = None):
        self._ema = ema

    def on_eval_begin(self, run_context: RunContext):
        if self._ema is not None:
            # swap ema weight and network weight
            self._ema.swap_before_eval()

    def on_eval_end(self, run_context: RunContext):
        if self._ema is not None:
            self._ema.swap_after_eval()


class VAEEmbedCallback(Callback):
    def __init__(self, vae_embed_func: Callable[[Tensor], tuple[Tensor, Optional[Tensor]]], return_cond: bool = False):
        """
        Callback that embeds input images/videos and conditions into latent space using VAE.
        As VAEs don't support dynamic graph shape in MindSpore, the input images/videos are embedded in this callback in
        Pynative mode.

        Args:
            vae_embed_func: function that takes input batch of images/videos and generates latent space representation
                            and conditioning.
            return_cond: whether to return conditioning or not.
        """
        super().__init__()
        self._vae_embed_func = vae_embed_func
        self._return_cond = return_cond

    @pynative_context()
    @no_grad()
    def on_train_step_begin(self, run_context: RunContext) -> None:
        batch = run_context.original_args().train_dataset_element  # is a mutable list
        batch[0], cond = self._vae_embed_func(batch[0])
        if self._return_cond:
            batch.append(cond)  # I/V conditioning
