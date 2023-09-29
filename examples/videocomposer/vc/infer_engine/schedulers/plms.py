# TODO: To be implemented.
import numpy as np

import mindspore.nn as nn

__all__ = ["PLMS"]


class PLMS(nn.Cell):
    def __init__(self, model: nn.Cell, betas: np.ndarray) -> None:
        super().__init__()
        raise NotImplementedError("To be implemented")
