import PIL

import mindspore as ms
from mindspore import ops

from ..util import save_videos_grid


class TuningFreePipeline:
    def __init__(self, sd, unet, scheduler, depth_estimator):
        super().__init__()

        self.sd = sd
        self.unet = unet
        self.scheduler = scheduler
        self.depth_estimator = depth_estimator

        self.vae_scale_factor = 2 ** (self.sd.first_stage_model.encoder.num_resolutions - 1)

    def prepare_depth_map(self, image, depth_map, batch_size, do_classifier_free_guidance):
        if isinstance(image, PIL.Image.Image):
            image = [image]
        else:
            image = [img for img in image]

        if depth_map is None:
            if len(image) < 20:
                depth_map = ms.Tensor(self.depth_estimator(image))
            else:
                depth_map = []

                for i in range(0, len(image), 20):
                    depth_map.append(ms.Tensor(self.depth_estimator(image[i : i + 20])))

                depth_map = ops.cat(depth_map)

        depth_map = ops.interpolate(
            depth_map.unsqueeze(1),
            size=(512 // self.vae_scale_factor, 512 // self.vae_scale_factor),
            mode="bicubic",
            align_corners=False,
        )

        depth_min = ops.amin(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_max = ops.amax(depth_map, axis=[1, 2, 3], keepdims=True)
        depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0

        save_videos_grid((depth_map + 1) / 2, "./depth_map.gif")

        # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
        if depth_map.shape[0] < batch_size:
            depth_map = depth_map.tile((batch_size, 1, 1, 1))

        depth_map = ops.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map  # classifier free guidance

        return depth_map
