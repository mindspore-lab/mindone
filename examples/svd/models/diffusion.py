import os
import sys
from typing import List, Optional

from gm.models.diffusion import DiffusionEngine
from gm.util.util import append_dims

import mindspore as ms
from mindspore import Tensor, nn, ops

# FIXME: remove in future when mindone is ready for install
sys.path.append(os.path.abspath(os.path.join(__file__, "../../../../")))
from mindone.utils.version_control import MS_VERSION, is_910b


def net_to_dtype(
    net: nn.Cell,
    dtype: ms.dtype,
    exclude_layers: Optional[List[nn.Cell]] = None,
    exclude_dtype: ms.dtype = ms.float16,
):
    """
    Converts the data type of a neural network except for the layers specified in `filter_layers`.

    Args:
        net: The network to be converted.
        dtype: The data type to convert the neural network to.
        exclude_layers: A list of specific layers to exclude from the conversion. Default is None.
        exclude_dtype: The data type to convert excluded layers to. Default is None, which means no conversion.
    """
    if net.cells():
        for cell in net.cells():
            net_to_dtype(cell, dtype, exclude_layers)
    else:
        if exclude_layers is None or type(net) not in exclude_layers:
            net.to_float(dtype)
        else:
            net.to_float(exclude_dtype)


class VideoDiffusionEngine(DiffusionEngine):
    def __init__(self, *args, en_and_decode_n_samples_a_time: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.weighting = self.denoiser.weighting

        # There is a bug in MindSpore 2.2 that causes the execution of `nn.Conv3d` in FP32 mode to fail.
        if self.disable_first_stage_amp and MS_VERSION < "2.3" and is_910b():
            net_to_dtype(self.first_stage_model, ms.float32, exclude_layers=[nn.Conv3d], exclude_dtype=ms.float16)

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z

        n_samples = self.en_and_decode_n_samples_a_time or z.shape[0]
        n_samples = min(z.shape[0], n_samples)
        if z.shape[0] % n_samples:
            raise ValueError("Total number of frames must be divisible by number of decoded frames at a time.")

        all_out = [
            self.first_stage_model.decode(z[n : n + n_samples], timesteps=n_samples)
            for n in range(0, z.shape[0], n_samples)
        ]
        return ops.cat(all_out, axis=0)

    def encode_first_stage(self, x):
        n_samples = self.en_and_decode_n_samples_a_time or x.shape[0]
        all_out = [self.first_stage_model.encode(x[n : n + n_samples]) for n in range(0, x.shape[0], n_samples)]

        z = ops.cat(all_out, axis=0)
        z = self.scale_factor * z
        return z

    def construct(
        self,
        batch: Tensor,
        cond_frames_without_noise: Tensor,
        cond_frames: Tensor,
        cond_aug: Tensor,
        fps_id: Tensor,
        motion_bucket_id: Tensor,
    ):
        num_frames = batch.shape[1]

        # merge the batch dimension with the frame dimension b t c h w -> (b t) c h w
        batch = batch.reshape(-1, *batch.shape[2:])
        fps_id = fps_id.reshape(-1, *fps_id.shape[2:])
        motion_bucket_id = motion_bucket_id.reshape(-1, *motion_bucket_id.shape[2:])
        cond_aug = cond_aug.reshape(-1, *cond_aug.shape[2:])

        x = self.encode_first_stage(batch)

        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.weighting(sigmas), x.ndim)

        tokens = (cond_frames_without_noise, fps_id, motion_bucket_id, cond_frames, cond_aug)

        vector, crossattn, concat = self.conditioner(*tokens)
        crossattn = crossattn.repeat(num_frames, axis=0)
        concat = concat.repeat(num_frames, axis=0)

        c_skip, c_out, c_in, c_noise = self.denoiser(sigmas, noised_input.ndim)
        model_output = self.model(
            noised_input * c_in,
            c_noise,
            concat=concat,
            context=crossattn,
            y=vector,
            num_frames=num_frames,
        )
        model_output = model_output * c_out + noised_input * c_skip
        loss = self.loss_fn(model_output, x, w)
        return loss.mean()
