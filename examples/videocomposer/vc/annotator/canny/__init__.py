import cv2
import numpy as np

import mindspore as ms


class CannyDetector:
    def __call__(self, img, low_threshold=None, high_threshold=None, random_threshold=True):
        #  Convert to numpy
        if isinstance(img, ms.Tensor):  # (h, w, c)
            img = img.asnumpy()
            img_np = cv2.convertScaleAbs((img * 255.0))
        elif isinstance(img, np.ndarray):  # (h, w, c)
            img_np = img  # we assume values are in the range from 0 to 255.
        else:
            raise TypeError(f"The input 'img' should be a 'mindspore.Tensor' or 'numpy.ndarray', but got {type(img)}.")

        #  Select the threshold
        if (low_threshold is None) and (high_threshold is None):
            median_intensity = np.median(img_np)
            if random_threshold is False:
                low_threshold = int(max(0, (1 - 0.33) * median_intensity))
                high_threshold = int(min(255, (1 + 0.33) * median_intensity))
            else:
                random_canny = np.random.uniform(0.1, 0.4)
                # Might try other values
                low_threshold = int(max(0, (1 - random_canny) * median_intensity))
                high_threshold = 2 * low_threshold

        # Detect canny edge
        canny_edge = cv2.Canny(img_np, low_threshold, high_threshold)

        canny_condition = ms.Tensor(canny_edge.copy()).unsqueeze(-1).float() / 255.0

        return canny_condition
