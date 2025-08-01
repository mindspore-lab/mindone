"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/schedulers/scheduling_consistency_decoder.py."""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import mint

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput
from ..utils.mindspore_utils import randn_tensor
from .scheduling_utils import SchedulerMixin


# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(
    num_diffusion_timesteps,
    max_beta=0.999,
    alpha_transform_type="cosine",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)

    else:
        raise ValueError(f"Unsupported alpha_transform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return ms.tensor(betas, dtype=ms.float32)


@dataclass
class ConsistencyDecoderSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function.

    Args:
        prev_sample (`ms.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: ms.Tensor


class ConsistencyDecoderScheduler(SchedulerMixin, ConfigMixin):
    order = 1

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1024,
        sigma_data: float = 0.5,
    ):
        betas = betas_for_alpha_bar(num_train_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = mint.cumprod(alphas, dim=0)

        self.sqrt_alphas_cumprod = mint.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mint.sqrt(1.0 - alphas_cumprod)

        sigmas = mint.sqrt(1.0 / alphas_cumprod - 1)

        sqrt_recip_alphas_cumprod = mint.sqrt(1.0 / alphas_cumprod)

        self.c_skip = sqrt_recip_alphas_cumprod * sigma_data**2 / (sigmas**2 + sigma_data**2)
        self.c_out = sigmas * sigma_data / (sigmas**2 + sigma_data**2) ** 0.5
        self.c_in = sqrt_recip_alphas_cumprod / (sigmas**2 + sigma_data**2) ** 0.5

        # setable values
        self.timesteps = ms.tensor([1008, 512])

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
    ):
        if num_inference_steps != 2:
            raise ValueError("Currently more than 2 inference steps are not supported.")

        self.timesteps = ms.tensor([1008, 512])

    @property
    def init_noise_sigma(self):
        return self.sqrt_one_minus_alphas_cumprod[self.timesteps[0]]

    def scale_model_input(self, sample: ms.Tensor, timestep: Optional[int] = None) -> ms.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`ms.Tensor`):
                The input sample.
            timestep (`int`, *optional*):
                The current timestep in the diffusion chain.

        Returns:
            `ms.Tensor`:
                A scaled input sample.
        """
        return (sample * self.c_in[timestep]).to(sample.dtype)

    def step(
        self,
        model_output: ms.Tensor,
        timestep: Union[float, ms.Tensor],
        sample: ms.Tensor,
        generator: Optional[np.random.Generator] = None,
        return_dict: bool = False,
    ) -> Union[ConsistencyDecoderSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`ms.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`float`):
                The current timestep in the diffusion chain.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`np.random.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a
                [`~schedulers.scheduling_consistency_models.ConsistencyDecoderSchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_consistency_models.ConsistencyDecoderSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_consistency_models.ConsistencyDecoderSchedulerOutput`] is returned, otherwise
                a tuple is returned where the first element is the sample tensor.
        """
        x_0 = (
            self.c_out[timestep].to(model_output.dtype) * model_output + self.c_skip[timestep].to(sample.dtype) * sample
        )

        timestep_idx = (self.timesteps == timestep).nonzero()[0]

        if timestep_idx == len(self.timesteps) - 1:
            prev_sample = x_0
        else:
            noise = randn_tensor(x_0.shape, generator=generator, dtype=x_0.dtype)
            prev_sample = (
                self.sqrt_alphas_cumprod[self.timesteps[timestep_idx + 1]].to(x_0.dtype) * x_0
                + self.sqrt_one_minus_alphas_cumprod[self.timesteps[timestep_idx + 1]].to(x_0.dtype) * noise
            )

        if not return_dict:
            return (prev_sample,)

        return ConsistencyDecoderSchedulerOutput(prev_sample=prev_sample)
