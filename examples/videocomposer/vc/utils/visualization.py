import os
from typing import Dict, Tuple

import imageio
import numpy as np

import mindspore as ms
import mindspore.ops as ops
from mindspore import JitConfig, Tensor

try:
    _config = JitConfig(jit_level="O3") if os.environ.get("MS_ENABLE_GE", 0) else None
except ValueError:  # for MS > 2.1
    _config = JitConfig(jit_level="O2")

__all__ = ["save_video_multiple_conditions"]


def rearrange_tensor(x: np.ndarray, nrow: int = 1) -> np.ndarray:
    """b c f h w -> f (i h) (j w) c, where i = nrow, j=ncol, i x j = b"""
    b, c, f, h, w = x.shape
    nrow = min(nrow, b)
    x = np.reshape(x, (nrow, b // nrow, c, f, h, w))  # i j c f h w
    x = np.transpose(x, (2, 3, 0, 4, 1, 5))  # c f i h j w
    x = np.reshape(x, (c, f, nrow * h, -1))  # c f (i h) (j w)
    x = np.transpose(x, (1, 2, 3, 0))
    return x


def swap_c_t(x: np.ndarray) -> np.ndarray:
    """Swap the second and third dimension"""
    x = np.transpose(x, (0, 2, 1, 3, 4))
    return x


def unormalize_tensor(x: np.ndarray, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> np.ndarray:
    """Convert from [-1, 1] to [0, 1]
    Args:
        x: b c f h w
        mean: (3, )
        std: (3, )
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    mean = np.reshape(mean, (1, -1, 1, 1, 1))
    std = np.reshape(std, (1, -1, 1, 1, 1))
    x = x * std + mean
    x = np.clip(x, 0, 1)
    return x


@ms.jit(jit_config=_config)
def resize_tensor_for_visual(x: Tensor, n: int, h: int, w: int) -> Tensor:
    """Resize the tensor to (-1, -1, n, h, w) shape
    Args:
        x: b c f h w
    """
    # TODO: change to ops.interpolate(mode="trilinear") once it is ok on 910B
    b, c, f, _, _ = x.shape
    x = ops.reshape(x, (-1, f, x.shape[3], x.shape[4]))
    x = ops.interpolate(x, size=(h, w), mode="bilinear")
    x = ops.reshape(x, (-1, f, h * w))
    x = ops.transpose(x, (0, 2, 1))
    x = ops.adaptive_avg_pool1d(x, n)
    x = ops.transpose(x, (0, 2, 1))
    x = ops.reshape(x, (b, c, n, h, w))
    return x


def video_tensor_to_gif(images: np.ndarray, path: str, duration: int = 120, save_frames: bool = False) -> None:
    """images: f x h x w x c"""
    images = (images * 255).round().clip(0, 255).astype(np.uint8)
    images = [x for x in images]
    imageio.mimwrite(path, images, duration=duration)
    if save_frames:
        root = os.path.join(os.path.splitext(path)[0], "frames")
        if not os.path.isdir(root):
            os.makedirs(root)
        for i, x in enumerate(images):
            imageio.imwrite(os.path.join(root, f"{i:04d}.jpg"), x)


def save_video_multiple_conditions(
    filename: str,
    video_tensor: np.ndarray,
    model_kwargs: Dict[str, np.ndarray],
    source_imgs: np.ndarray,
    mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    save_origin_video: bool = True,
    save_frames: bool = False,
    nrow: int = 1,
) -> None:
    """Save video gifs"""

    # prepare the output image
    video_tensor = unormalize_tensor(video_tensor, mean, std)
    _, _, n, h, w = video_tensor.shape
    vid_gif = rearrange_tensor(video_tensor, nrow=nrow)

    # prepare the conditional image
    model_kwargs_channel3 = dict()
    for key, conditions in model_kwargs.items():
        if conditions.shape[-1] == 1024:  # Skip for style embedding
            continue

        c = conditions.shape[1]
        if c == 1:
            conditions = np.tile(conditions, (1, 3, 1, 1, 1))
        elif c == 2:
            # TODO: use HSV for u, v vector
            conditions = np.concatenate([conditions, conditions[:, :1]], axis=1)
        elif c == 3:
            pass
        elif c == 4:
            color = (conditions[:, :3] + 1.0) / 2.0
            alpha = conditions[:, 3:4]
            conditions = color * alpha + (1.0 - alpha)
        else:
            raise ValueError(f"Unsupported dimension `{c}`")

        model_kwargs_channel3[key] = resize_tensor_for_visual(Tensor(conditions, dtype=ms.float32), n, h, w).asnumpy()

    cons_list = [rearrange_tensor(con, nrow=nrow) for con in model_kwargs_channel3.values()]

    if save_origin_video:
        source_imgs = swap_c_t(source_imgs)
        source_imgs = resize_tensor_for_visual(Tensor(source_imgs, dtype=ms.float32), n, h, w).asnumpy()
        source_imgs = rearrange_tensor(source_imgs, nrow=nrow)
        vid_gif = np.concatenate([source_imgs, *cons_list, vid_gif], axis=2)
    else:
        vid_gif = np.concatenate([*cons_list, vid_gif], axis=2)

    video_tensor_to_gif(vid_gif, filename, save_frames=save_frames)
