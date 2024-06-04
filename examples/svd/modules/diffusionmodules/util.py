from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import mindspore as ms
from mindspore import nn, ops


class AlphaBlender(nn.Cell):
    """
    Blend spatial and temporal network branches using a learnable alpha value:
    blended = spatial * alpha + temporal * (1 - alpha)

    Args:
        alpha: a blending coefficient between 0 and 1.
        merge_strategy: merge strategy to use for spatial and temporal blending.
                        Options: "fixed" - alpha remains constant, "learned" - alpha is learned during training,
                        "learned_with_images" - alpha is learned for video frames only during hybrid (images and videos)
                        training. Default: "learned".
    """

    def __init__(
        self,
        alpha: float,
        merge_strategy: Literal["fixed", "learned", "learned_with_images"] = "learned",
    ):
        super().__init__()
        self.merge_strategy = merge_strategy

        if self.merge_strategy == "fixed":
            self.mix_factor = ms.Tensor([alpha])
        elif self.merge_strategy in ["learned", "learned_with_images"]:
            self.mix_factor = ms.Parameter([alpha])
        else:
            raise ValueError(f"Unknown branch merge strategy {self.merge_strategy}")

    def _get_alpha(self, image_only_indicator: ms.Tensor, ndim: int) -> ms.Tensor:
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

            if ndim == 5:  # apply alpha on the frame axis
                alpha = alpha[:, None, :, None, None]
            elif ndim == 3:  # apply alpha on the batch x frame axis
                alpha = alpha.reshape(-1)[:, None, None]
            else:
                raise ValueError(f"Unexpected ndim {ndim}. Dimensions should be 3 or 5")

        else:
            raise NotImplementedError(f"Unknown branch merge strategy {self.merge_strategy}")
        return alpha

    def construct(
        self, x_spatial: ms.Tensor, x_temporal: ms.Tensor, image_only_indicator: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        alpha = self._get_alpha(image_only_indicator, x_spatial.ndim)
        x = alpha.to(x_spatial.dtype) * x_spatial + (1.0 - alpha).to(x_spatial.dtype) * x_temporal
        return x
