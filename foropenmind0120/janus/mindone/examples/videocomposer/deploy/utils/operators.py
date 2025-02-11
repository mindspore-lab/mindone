import numpy as np


def swap_c_t_and_tile(x: np.ndarray) -> np.ndarray:
    """Swap the second and third dimension, and duplicated along the first dimension for
    classifier-free guidance
    """
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = np.tile(x, (2, 1, 1, 1, 1))
    return x


def make_masked_images(imgs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Making masked image for condition"""
    imgs = (imgs - 0.5) / 0.5
    masked_imgs = np.concatenate([imgs * (1 - masks), (1 - masks)], axis=2)
    return masked_imgs
