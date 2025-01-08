# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Modified from: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L103-L110

from typing import Union

import mindspore as ms
from mindspore import Tensor, mint, nn


class LayerScale(nn.Cell):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = ms.Parameter(init_values * mint.ones(dim))

    def construct(self, x: Tensor) -> Tensor:
        return mint.mul(x, self.gamma) if self.inplace else x * self.gamma
