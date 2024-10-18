import logging
from collections import OrderedDict

import mindspore.mint as mint
import mindspore.nn as nn
from mindspore import Tensor

logger = logging.getLogger(__name__)


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor):
        return x * mint.sigmoid(1.702 * x)


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "quick_gelu": QuickGELU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}
ACT2FN = ClassInstantier(ACT2CLS)
