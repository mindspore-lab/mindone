import logging
from typing import Dict, Union

import mindspore as ms
from mindspore import _no_grad, jit_class, nn

__all__ = ["load_ckpt_params", "no_grad"]

logger = logging.getLogger(__name__)


def load_ckpt_params(model: nn.Cell, ckpt: Union[str, Dict]) -> nn.Cell:
    if isinstance(ckpt, str):
        logger.info(f"Loading {ckpt} params into network...")
        param_dict = ms.load_checkpoint(ckpt)
    else:
        param_dict = ckpt

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
    if not (len(param_not_load) == len(ckpt_not_load) == 0):
        logger.warning(
            "Exist ckpt params not loaded: {} (total: {}), or net params not loaded: {} (total: {})".format(
                ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
            )
        )
    return model


@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)
