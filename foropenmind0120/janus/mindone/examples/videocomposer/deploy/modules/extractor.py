from typing import Optional

import cv2
import numpy as np

__all__ = ["CannyExtractor"]


class CannyExtractor:
    def __init__(self) -> None:
        self.canny_detector = CannyDetector()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        b, f, c, h, w = x.shape
        x = np.reshape(x, (b * f, c, h, w))
        x = np.transpose(x, (0, 2, 3, 1))
        x = np.stack([self.canny_detector(a) for a in x])  # (b f) h w c
        x = np.reshape(x, (b, f, h, w, c))
        x = np.transpose(x, (0, 4, 1, 2, 3))  # b c f h w
        # for classifier-free guidance
        x = np.tile(x, (2, 1, 1, 1, 1))
        return x


class CannyDetector:
    def __call__(
        self,
        img: np.ndarray,
        low_threshold: Optional[float] = None,
        high_threshold: Optional[float] = None,
        random_threshold: Optional[float] = True,
    ) -> np.ndarray:
        #  Select the threshold
        if (low_threshold is None) and (high_threshold is None):
            median_intensity = np.median(img)
            if random_threshold is False:
                low_threshold = int(max(0, (1 - 0.33) * median_intensity))
                high_threshold = int(min(255, (1 + 0.33) * median_intensity))
            else:
                random_canny = np.random.uniform(0.1, 0.4)
                # Might try other values
                low_threshold = int(max(0, (1 - random_canny) * median_intensity))
                high_threshold = 2 * low_threshold

        # Detect canny edge
        canny_edge = cv2.Canny(img, low_threshold, high_threshold)

        canny_condition = np.expand_dims(canny_edge, axis=-1).astype(np.float32) / 255.0

        return canny_condition
