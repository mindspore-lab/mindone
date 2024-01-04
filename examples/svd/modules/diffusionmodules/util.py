from typing import Optional, Tuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import mindspore as ms
from mindspore import nn, ops


class AlphaBlender(nn.Cell):
    """
    Blend different network branches with learnable alpha value:
    blended = branch1 * alpha + branch2 * (1 - alpha)
    """

    def __init__(
        self,
        alpha: float,
        merge_strategy: Literal["fixed", "learned", "learned_with_images"] = "learned_with_images",
        reshape_pattern: Tuple[int, ...] = (-1, 1, 1),
    ):
        super().__init__()
        self.merge_strategy = merge_strategy
        self.reshape_pattern = reshape_pattern

        if self.merge_strategy == "fixed":
            self.mix_factor = ms.Tensor([alpha])
        elif self.merge_strategy == "learned" or self.merge_strategy == "learned_with_images":
            self.mix_factor = ms.Parameter([alpha])
        else:
            raise ValueError(f"unknown merge strategy {self.merge_strategy}")

    def get_alpha(self, image_only_indicator: ms.Tensor) -> ms.Tensor:
        if self.merge_strategy == "fixed":
            alpha = self.mix_factor
        elif self.merge_strategy == "learned":
            alpha = ops.sigmoid(self.mix_factor)
        elif self.merge_strategy == "learned_with_images":
            alpha = ops.where(
                image_only_indicator.bool(),
                ops.ones((1, 1)),
                ops.expand_dims(ops.sigmoid(self.mix_factor), -1),
            )
            alpha = alpha.reshape(*self.reshape_pattern)
        else:
            raise NotImplementedError()
        return alpha

    def construct(
        self, x_spatial: ms.Tensor, x_temporal: ms.Tensor, image_only_indicator: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        alpha = self.get_alpha(image_only_indicator)
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x
