# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from mindspore import Tensor, mint, nn


class SwiGLUFFN(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Cell] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Dense(in_features, 2 * hidden_features, has_bias=bias)
        self.w3 = nn.Dense(hidden_features, out_features, has_bias=bias)

    def construct(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = mint.chunk(x12, 2, dim=-1)
        hidden = mint.nn.functional.silu(x1) * x2
        return self.w3(hidden)


SwiGLU = SwiGLUFFN


# https://github.com/facebookresearch/xformers/blob/6e10bd21ac6fc878657b24684723ccd05e41d385/xformers/ops/swiglu_op.py#L433
class SwiGLUFFNFused(SwiGLU):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Cell] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            bias=bias,
        )
