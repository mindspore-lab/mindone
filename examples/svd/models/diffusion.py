from typing import List, Optional

from gm.models.diffusion import DiffusionEngine
from gm.util.util import get_obj_from_str
from omegaconf import DictConfig

import mindspore as ms
from mindspore import nn, ops

from gm.util.util import append_dims


def net_to_dtype(net: nn.Cell, dtype: ms.dtype, exclude_layers: Optional[List[nn.Cell]] = None):
    """
    Converts the data type of a neural network except for the layers specified in `filter_layers`.

    Args:
        net: The network to be converted.
        dtype: The data type to convert the neural network to.
        exclude_layers: A list of specific layers to exclude from the conversion. Default is None.
    """
    if net.cells():
        for cell in net.cells():
            net_to_dtype(cell, dtype, exclude_layers)
    else:
        if exclude_layers is None or type(net) not in exclude_layers:
            net.to_float(dtype)


class VideoDiffusionEngine(DiffusionEngine):
    def __init__(self, *args, en_and_decode_n_samples_a_time: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        self._force_fp16 = None
        if isinstance(self.disable_first_stage_amp, DictConfig):
            if "force_fp16" in self.disable_first_stage_amp:
                self._force_fp16 = [get_obj_from_str(item) for item in self.disable_first_stage_amp["force_fp16"]]
            self.disable_first_stage_amp = self.disable_first_stage_amp["enable"]

    def decode_first_stage(self, z):
        if self.disable_first_stage_amp:
            net_to_dtype(self.first_stage_model, ms.float32, exclude_layers=self._force_fp16)
            z = z.astype(ms.float32)

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
        if self.disable_first_stage_amp:
            self.first_stage_model.to_float(ms.float32)
            x = x.astype(ms.float32)

        n_samples = self.en_and_decode_n_samples_a_time or x.shape[0]
        all_out = [self.first_stage_model.encode(x[n : n + n_samples]) for n in range(0, x.shape[0], n_samples)]

        z = ops.cat(all_out, axis=0)
        z = self.scale_factor * z
        return z

    # def construct(self, batch: ms.Tensor, caption: ms.Tensor, fps_id, motion_bucket_id, cond_aug, num_frames):
    def construct(
        self,
        batch: ms.Tensor,
        cond_frames_without_noise: ms.Tensor,
        fps_id,
        motion_bucket_id,
        cond_frames,
        cond_aug,
    ):
        num_frames = batch.shape[1]
        # merge the batch dimension with the frame dimension b t c h w -> (b t) c h w
        batch = batch.reshape(-1, *batch.shape[2:])
        # cond_frames_without_noise = cond_frames_without_noise.reshape(-1, *cond_frames_without_noise.shape[2:])
        fps_id = fps_id.reshape(-1, *fps_id.shape[2:])
        motion_bucket_id = motion_bucket_id.reshape(-1, *motion_bucket_id.shape[2:])
        # cond_frames = cond_frames.reshape(-1, *cond_frames.shape[2:])
        cond_aug = cond_aug.reshape(-1, *cond_aug.shape[2:])

        # get latent target
        x = self.encode_first_stage(batch)

        # get noise and sigma
        sigmas = self.sigma_sampler(x.shape[0])
        noise = ops.randn_like(x)
        noised_input = self.loss_fn.get_noise_input(x, noise, sigmas)
        w = append_dims(self.denoiser.w(sigmas), x.ndim)

        tokens = (cond_frames_without_noise, fps_id, motion_bucket_id, cond_frames, cond_aug)

        # compute loss
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
            num_video_frames=num_frames,
        )
        model_output = model_output * c_out + noised_input * c_skip
        loss = self.loss_fn(model_output, x, w)
        return loss.mean()
