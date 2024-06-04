import cv2
import numpy as np


class TokenizerWrapper:
    def __init__(self, tokenizer: callable):
        super().__init__()
        self._tokenizer = tokenizer

    def __call__(self, text: np.ndarray) -> np.ndarray:
        text = text.item()  # extract string from the numpy array
        SOT_TEXT = self._tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self._tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = self._tokenizer.context_length

        sot_token = self._tokenizer.encoder[SOT_TEXT]
        eot_token = self._tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN])
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result.astype(np.int32)


class CannyRandomThreshold:
    """
    Apply Canny edge detector to an image with random low and high thresholds. Receives an RGB image and returns a
    grayscale image with detected edges.

    Args:
        low_threshold: lower bound for filtering edges. Default: 100.
        high_threshold: upper bound for filtering edges. Default: 200.
        shift_range: random shift range applied to low and high thresholds. Default: 50.
        seed: random seed. Default: 42.
    """

    def __init__(self, low_threshold: int = 100, high_threshold: int = 200, shift_range: int = 50, seed: int = 42):
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._rng = np.random.default_rng(seed)
        self._shift_range = shift_range

    def __call__(self, image: np.ndarray) -> np.ndarray:
        low_threshold = self._low_threshold + self._rng.integers(-self._shift_range, self._shift_range, endpoint=True)
        high_threshold = self._high_threshold + self._rng.integers(-self._shift_range, self._shift_range, endpoint=True)
        return np.expand_dims(cv2.Canny(image, low_threshold, high_threshold), axis=2)
