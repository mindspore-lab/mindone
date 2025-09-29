# This code is copied from https://github.com/Picsart-AI-Research/Text2Video-zero

import cv2

__all__ = ["CannyDetector"]


class CannyDetector:
    def __call__(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)
