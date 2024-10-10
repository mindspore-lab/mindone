from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import logging

from tqdm import tqdm

from mindspore import Tensor, dtype, ops
from mindspore.communication import get_rank

from ..acceleration.parallel_states import get_sequence_parallel_group
from ..utils.distributions import LogisticNormal
from .iddpm.diffusion_utils import mean_flat

logger = logging.getLogger(__name__)


class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def __call__(
        self,
        model,
        shape,
        z: Tensor,
        model_kwargs,
        frames_mask=None,
        clip_denoised=False,
        progress=True,
    ):
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [Tensor([t] * z.shape[0]) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [
                timestep_transform(
                    t,
                    model_kwargs["height"],
                    model_kwargs["width"],
                    model_kwargs["num_frames"],
                    num_timesteps=self.num_timesteps,
                )
                for t in timesteps
            ]

        if frames_mask is not None:
            noise_added = (ops.zeros_like(frames_mask) + frames_mask).astype(dtype.bool_)

        for i, t in tqdm(enumerate(timesteps), total=self.num_sampling_steps):
            # mask for adding noise
            if frames_mask is not None:
                mask_t = frames_mask * self.num_timesteps
                x0 = z.copy()
                x_noise = self.scheduler.add_noise(x0, ops.randn_like(x0), t)
                # x_noise = self.scheduler.add_noise(x0, ms.Tensor(np.random.randn(*x0.shape), dtype=ms.float32), t)

                model_kwargs["frames_mask"] = mask_t_upper = mask_t >= t.unsqueeze(1)
                mask_add_noise = (mask_t_upper * (1 - noise_added)).astype(dtype.bool_)

                z = ops.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            pred = model(z, t, **model_kwargs)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + pred * dt[:, None, None, None, None]

            if frames_mask is not None:
                z = ops.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z


def timestep_transform(
    t,
    height: Tensor,
    width: Tensor,
    num_frames: Tensor,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # FIXME: avoid calculations on tensors outside `construct`
    t = t / num_timesteps
    resolution = height.astype(dtype.float32) * width.astype(dtype.float32)
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    if num_frames[0] == 1:  # image
        num_frames = ops.ones_like(num_frames, dtype=dtype.float32)
    else:
        num_frames = num_frames.astype(dtype.float32) // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        sample_method: Literal["discrete-uniform", "uniform", "logit-normal"] = "uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps

        if sample_method == "discrete-uniform":
            self._sample_func = self._discrete_sample
        elif sample_method == "uniform":
            self._sample_func = self._uniform_sample
        elif sample_method == "logit-normal":
            self.distribution = LogisticNormal(loc, scale)
            self._sample_func = self._logit_normal_sample
        else:
            raise ValueError(f"Unknown sample method: {sample_method}")

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

        self.sp_group = get_sequence_parallel_group()
        if self.sp_group is not None:
            logging.info(
                f"Broadcasting all random variables from rank (0) to current rank ({get_rank(self.sp_group)}) in group `{self.sp_group}`."
            )
            self.broadcast = ops.Broadcast(0, group=self.sp_group)

    def _discrete_sample(self, size: int) -> Tensor:
        return ops.randint(0, self.num_timesteps, (size,), dtype=dtype.int32)

    def _broadcast(self, x: Tensor) -> Tensor:
        if self.sp_group is None:
            return x
        return self.broadcast((x,))[0]

    def _uniform_sample(self, size: int) -> Tensor:
        return ops.rand((size,), dtype=dtype.float32) * self.num_timesteps

    def _logit_normal_sample(self, size: int) -> Tensor:
        return self.distribution.sample((size,))[0] * self.num_timesteps  # noqa

    def training_losses(
        self,
        model,
        x_start: Tensor,
        text_embed: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            t = self._sample_func(x_start.shape[0])
            t = self._broadcast(t)
            if self.use_timestep_transform:
                t = timestep_transform(
                    t, height, width, num_frames, scale=self.transform_scale, num_timesteps=self.num_timesteps
                )

        noise = ops.randn_like(x_start)
        noise = self._broadcast(noise)

        x_t = self.add_noise(x_start, noise, t)

        # frames mask branch
        t0 = ops.zeros_like(t)
        x_t0 = self.add_noise(x_start, noise, t0)
        x_t = ops.where(frames_mask[:, None, :, None, None], x_t, x_t0)

        text_embed = text_embed[:, None, :]
        model_output = model(
            x_t, t, text_embed, mask, frames_mask=frames_mask, fps=fps, height=height, width=width, **kwargs
        )
        velocity_pred = model_output.chunk(2, axis=1)[0]
        loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), frames_mask=frames_mask)

        return loss.mean()

    def add_noise(
        self,
        original_samples: Tensor,
        noise: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """
        Compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints[:, None, None, None, None]
        timepoints = timepoints.tile((1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4]))

        return timepoints * original_samples + (1 - timepoints) * noise
