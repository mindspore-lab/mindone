import logging
from typing import Dict, Tuple, Union

import mindspore as ms
import mindspore.nn as nn

__all__ = ["load_ckpt_params", "count_params"]

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


def count_params(model: nn.Cell) -> Tuple[int, int]:
    total_params = sum([param.size for param in model.get_parameters()])
    trainable_params = sum([param.size for param in model.trainable_params()])
    return total_params, trainable_params
