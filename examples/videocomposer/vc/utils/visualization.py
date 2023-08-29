import logging

import imageio
import numpy as np

import mindspore as ms
from mindspore import ops

from .misc import rand_name

__all__ = [
    "video_tensor_to_gif",
    "save_video_multiple_conditions",
]

_logger = logging.getLogger(__name__)


# @torch.no_grad()
def video_tensor_to_gif(tensor, path, duration=120, loop=0, optimize=True):
    tensor = tensor.permute(1, 2, 3, 0)
    images = tensor.unbind(dim=0)
    images = [(image.numpy() * 255).astype("uint8") for image in images]
    imageio.mimwrite(path, images, duration=duration)
    return images


# @torch.no_grad()
def save_video_multiple_conditions(
    filename,
    video_tensor,
    model_kwargs,
    source_imgs,
    palette,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    nrow=8,
    save_origin_video=True,
):
    mean = ms.Tensor(mean).view(1, -1, 1, 1, 1)  # ncfhw
    std = ms.Tensor(std).view(1, -1, 1, 1, 1)  # ncfhw
    video_tensor = video_tensor * std + mean  # unnormalize back to [0,1]
    try:
        video_tensor = ops.clamp(video_tensor, 0, 1)
    except:  # noqa
        video_tensor = ops.clamp(video_tensor.float(), 0, 1)

    b, c, n, h, w = video_tensor.shape
    source_imgs = ops.adaptive_avg_pool3d(source_imgs, (n, h, w))

    model_kwargs_channel3 = {}
    for key, conditions in model_kwargs[0].items():
        if conditions.shape[-1] == 1024:  # Skip for style embedding
            continue
        if len(conditions.shape) == 3:  # which means that it is histogram.
            conditions_np = conditions.numpy()
            conditions = []
            for i in conditions_np:
                vis_i = []
                for j in i:
                    vis_i.append(palette.get_palette_image(j, percentile=90, width=256, height=256))
                conditions.append(np.stack(vis_i))
            conditions = ms.Tensor(np.stack(conditions))  # (8, 16, 256, 256, 3)
            # b n h w c -> b c n h w
            conditions = ops.transpose(conditions, (0, 4, 1, 2, 3))
        else:
            if conditions.shape[1] == 1:
                conditions = ops.cat([conditions, conditions, conditions], axis=1)
                conditions = ops.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.shape[1] == 2:
                conditions = ops.cat([conditions, conditions[:, :1]], axis=1)
                conditions = ops.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.shape[1] == 3:
                conditions = ops.adaptive_avg_pool3d(conditions, (n, h, w))
            elif conditions.shape[1] == 4:  # means it is a mask.
                color = (conditions[:, 0:3] + 1.0) / 2.0  # .astype(np.float32)
                alpha = conditions[:, 3:4]  # .astype(np.float32)
                conditions = color * alpha + 1.0 * (1.0 - alpha)
                conditions = ops.adaptive_avg_pool3d(conditions, (n, h, w))
        model_kwargs_channel3[key] = conditions

    if not filename:
        filename = "output/output_" + rand_name(suffix=".gif")
    try:
        # (i j) c f h w -> i j c f h w -> c f i h j w -> c f (i h) (j w), num_sample_rows=8
        def rearrange_tensor(x):
            x = ops.reshape(x, (nrow, x.shape[0] // nrow, *x.shape[1:]))
            x = ops.transpose(x, (2, 3, 0, 4, 1, 5))
            x = ops.reshape(x, (*x.shape[:2], x.shape[2] * x.shape[3], -1))
            return x

        vid_gif = rearrange_tensor(video_tensor)
        cons_list = [rearrange_tensor(con) for _, con in model_kwargs_channel3.items()]
        source_imgs = rearrange_tensor(source_imgs)

        if save_origin_video:
            vid_gif = ops.cat([source_imgs, *cons_list, vid_gif], axis=3)
        else:
            vid_gif = ops.cat([*cons_list, vid_gif], axis=3)

        video_tensor_to_gif(vid_gif, filename)
    except Exception as e:
        _logger.warning("save video to {} failed, error: {}".format(filename, e))
