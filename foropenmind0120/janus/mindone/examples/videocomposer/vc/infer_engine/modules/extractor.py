from typing import List, Optional, Union

import cv2
import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from ...annotator.depth import midas_v3_dpt_large
from ...annotator.sketch import pidinet_bsd, sketch_simplification_gan
from ...utils import get_abspath_of_weights

__all__ = ["DepthExtractor", "CannyExtractor", "SketchExtractor", "ConditionExtractor"]


class ConditionExtractor(nn.Cell):
    """A wrapper of extracting the conditions of each frame from the input video. Currently, it supports
    `DepthExtractor` and `SketchExtractor` only.

    Args:
        extractor: The extractor for extracting the conditions,
            the extractor should accept a input with a 4-Dimensional (b, c, h, w) tensor
        chunk_size: Perform the a single inference step for a given `chunk_size` of frames. Default: 1
    """

    def __init__(self, extractor: nn.Cell, chunk_size: int = 1) -> None:
        super().__init__()
        self.extractor = extractor
        self.chunk_size = chunk_size

    def construct(self, x: Tensor) -> Tensor:
        b, f, c, h, w = x.shape
        x = ops.reshape(x, (b * f, c, h, w))

        n = (b * f) // self.chunk_size

        if n > 1:
            xs = ops.chunk(x, n, axis=0)
            cond_data = []
            for x in xs:
                cond = self.extractor(x)
                cond_data.append(cond)

            x = ops.concat(cond_data, axis=0)
        else:
            x = self.extractor(x)

        _, c, h, w = x.shape
        # (b f) c h w -> b f c h w -> b c f h w
        x = ops.reshape(x, (b, f, c, h, w))
        x = ops.transpose(x, (0, 2, 1, 3, 4))
        # for classifier-free guidance
        x = ops.tile(x, (2, 1, 1, 1, 1))
        return x


class DepthExtractor(nn.Cell):
    def __init__(
        self, ckpt_path: str, depth_std: float = 20.0, depth_clamp: float = 10.0, use_fp16: bool = False
    ) -> None:
        super().__init__()

        dtype = ms.float16 if use_fp16 else ms.float32
        self.midas = (
            midas_v3_dpt_large(pretrained=True, ckpt_path=get_abspath_of_weights(ckpt_path))
            .set_train(False)
            .to_float(dtype)
        )
        for _, param in self.midas.parameters_and_names():
            param.requires_grad = False

        self.depth_std = depth_std
        self.depth_clamp = depth_clamp

    def construct(self, x: Tensor) -> Tensor:
        x = (x - 0.5) / 0.5
        x = self.midas(x)
        x = (x / self.depth_std).clamp(0, self.depth_clamp)
        return x


class CannyExtractor:
    def __init__(self) -> None:
        self.canny_detector = CannyDetector()

    def __call__(self, x: Union[np.ndarray, Tensor]) -> np.ndarray:
        if isinstance(x, Tensor):
            x = x.asnumpy()

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


class SketchExtractor(nn.Cell):
    def __init__(
        self,
        pidinet_ckpt_path: str,
        sketch_simplification_ckpt_path: str,
        sketch_mean: List[float] = [0.485, 0.456, 0.406],
        sketch_std: List[float] = [0.229, 0.224, 0.225],
        use_fp16: bool = False,
    ) -> None:
        super().__init__()

        dtype = ms.float16 if use_fp16 else ms.float32
        self.pidinet = (
            pidinet_bsd(pretrained=True, vanilla_cnn=True, ckpt_path=get_abspath_of_weights(pidinet_ckpt_path))
            .set_train(False)
            .to_float(dtype)
        )

        for _, param in self.pidinet.parameters_and_names():
            param.requires_grad = False

        self.cleaner = (
            sketch_simplification_gan(
                pretrained=True, ckpt_path=get_abspath_of_weights(sketch_simplification_ckpt_path)
            )
            .set_train(False)
            .to_float(dtype)
        )
        for _, param in self.cleaner.parameters_and_names():
            param.requires_grad = False

        self.pidi_mean = Tensor(np.array(sketch_mean).reshape(1, -1, 1, 1))
        self.pidi_std = Tensor(np.array(sketch_std).reshape(1, -1, 1, 1))

    def construct(self, x: Tensor) -> Tensor:
        x = self.pidinet((x - self.pidi_mean) / self.pidi_std)
        x = 1.0 - self.cleaner(1.0 - x)
        return x
