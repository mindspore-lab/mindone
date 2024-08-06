import logging
from typing import Set, Tuple

from mindspore import Tensor
from mindspore.train import Loss

_logger = logging.getLogger(__name__)


class BucketLoss(Loss):
    """
    Updates the loss only for the corresponding sample bucket.

    Args:
        name: bucket name, i.e "720p".
        resolutions: set of resolutions that belong to the bucket.
        num_frames: number of frames in the bucket.
    """

    def __init__(self, name: str, resolutions: Set[Tuple[int, int]], num_frames: int):
        super().__init__()
        self._name = name
        self._res = resolutions
        self._num_frames = num_frames

    def update(self, loss: Tensor, height: Tensor, width: Tensor, num_frames: Tensor):
        if (int(height[0].item()), int(width[0].item())) in self._res and int(num_frames[0].item()) == self._num_frames:
            self._sum_loss += loss.item()
            self._total_num += 1

    def eval(self) -> float:
        if self._total_num == 0:
            _logger.warning(f"No samples with {self._name} resolution and {self._num_frames} frames.")
            return 0.0
        return self._sum_loss / self._total_num
