from typing import Optional

import numpy as np
import mindspore as ms
from mindspore import mint, nn, Parameter

from ..nn.vo_ops import build_kwargs_from_config

__all__ = ["LayerNorm2d", "build_norm", "set_norm_eps"]


class LayerNorm2d(mint.nn.LayerNorm):
    def construct(self, x: ms.tensor) -> ms.tensor:
        out = x - mint.mean(x, dim=1, keepdim=True)
        out = out / mint.sqrt(mint.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out



class RMSNorm2d(nn.Cell):
    def __init__(
        self, num_features: int, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(ms.tensor(np.empty(self.num_features)))
            self.bias = Parameter(ms.tensor(np.empty(self.num_features))) if bias else None
        else:
            self.weight, self.bias = None, None

    def construct(self, x: ms.tensor) -> ms.tensor:
        x = (x / mint.sqrt(mint.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class RMSNorm3d(RMSNorm2d):
    def construct(self, x: ms.tensor) -> ms.tensor:
        x = (x / mint.sqrt(mint.square(x.float()).mean(dim=1, keepdim=True) + self.eps)).to(x.dtype)
        if self.elementwise_affine:
            x = x * self.weight.view(1, -1, 1, 1, 1) + self.bias.view(1, -1, 1, 1, 1)
        return x


# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": mint.nn.BatchNorm2d,
    "ln": mint.nn.LayerNorm,
    "ln2d": LayerNorm2d,
    "rms2d": RMSNorm2d,
    "rms3d": RMSNorm3d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> Optional[nn.Cell]:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def set_norm_eps(model: nn.Cell, eps: Optional[float] = None) -> None:
    for m in model.cells():
        if isinstance(m, (mint.nn.GroupNorm, mint.nn.LayerNorm, mint.nn.BatchNorm1d, mint.nn.BatchNorm2d, mint.nn.BatchNorm3d)):
            if eps is not None:
                m.eps = eps
