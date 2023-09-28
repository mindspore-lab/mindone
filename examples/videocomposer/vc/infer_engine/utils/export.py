#!/usr/bin/env python
import logging
import os
from typing import List

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor

_logger = logging.getLogger(__name__)


def model_export(net: nn.Cell, inputs: List[Tensor], name: str, model_save_path: str = "./models/mindir") -> None:
    if not os.path.isdir(model_save_path):
        os.makedirs(model_save_path)

    if os.path.isfile(os.path.join(model_save_path, f"{name}.mindir")):
        _logger.warning(f"`{name}` mindir already exist, skip.")
        return

    if os.path.isfile(os.path.join(model_save_path, f"{name}_graph.mindir")):
        _logger.warning(f"`{name}` mindir already exist, skip.")
        return

    ms.export(net, *inputs, file_name=os.path.join(model_save_path, name), file_format="MINDIR")
    _logger.info(f"convert `{name}` mindir done")
