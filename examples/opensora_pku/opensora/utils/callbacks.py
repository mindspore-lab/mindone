from typing import Optional, Union

from opensora.utils.ema import EMA, EMA_

from mindspore.train import Callback, RunContext


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
