# Copyright 2025 Katherine Crowson, The HuggingFace Team and hlky. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import mint

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils import BaseOutput, is_scipy_available
from ..utils.mindspore_utils import randn_tensor
from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

if is_scipy_available():
    import scipy.stats


@dataclass
# Copied from diffusers.schedulers.scheduling_ddpm.DDPMSchedulerOutput with DDPM->KDPM2AncestralDiscrete
class KDPM2AncestralDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.
    Args:
        prev_sample (`ms.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
        pred_original_sample (`ms.Tensor` of shape `(batch_size, num_channels, height, width)` for images):
            The predicted denoised sample `(x_{0})` based on the model output from the current timestep.
            `pred_original_sample` can be used to preview progress or for guidance.
    """

    prev_sample: ms.Tensor
    pred_original_sample: Optional[ms.Tensor] = None


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


class KDPM2AncestralDiscreteScheduler(SchedulerMixin, ConfigMixin):
    """
    KDPM2DiscreteScheduler with ancestral sampling is inspired by the DPMSolver2 and Algorithm 2 from the [Elucidating
    the Design Space of Diffusion-Based Generative Models](https://huggingface.co/papers/2206.00364) paper.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        beta_start (`float`, defaults to 0.00085):
            The starting `beta` value of inference.
        beta_end (`float`, defaults to 0.012):
            The final `beta` value.
        beta_schedule (`str`, defaults to `"linear"`):
            The beta schedule, a mapping from a beta range to a sequence of betas for stepping the model. Choose from
            `linear` or `scaled_linear`.
        trained_betas (`np.ndarray`, *optional*):
            Pass an array of betas directly to the constructor to bypass `beta_start` and `beta_end`.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        use_exponential_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use exponential sigmas for step sizes in the noise schedule during the sampling process.
        use_beta_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use beta sigmas for step sizes in the noise schedule during the sampling process. Refer to [Beta
            Sampling is All You Need](https://huggingface.co/papers/2407.12173) for more information.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 2

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,  # sensible defaults
        beta_end: float = 0.012,
        beta_schedule: str = "linear",
        trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
        prediction_type: str = "epsilon",
        timestep_spacing: str = "linspace",
        steps_offset: int = 0,
    ):
        if self.config.use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        if sum([self.config.use_beta_sigmas, self.config.use_exponential_sigmas, self.config.use_karras_sigmas]) > 1:
            raise ValueError(
                "Only one of `config.use_beta_sigmas`, `config.use_exponential_sigmas`, `config.use_karras_sigmas` can be used."
            )
        if trained_betas is not None:
            self.betas = ms.tensor(trained_betas, dtype=ms.float32)
        elif beta_schedule == "linear":
            self.betas = ms.tensor(np.linspace(beta_start, beta_end, num_train_timesteps), dtype=ms.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            self.betas = (
                ms.tensor(np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps), dtype=ms.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            self.betas = betas_for_alpha_bar(num_train_timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} is not implemented for {self.__class__}")

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = mint.cumprod(self.alphas, dim=0)

        #  set all values
        self.set_timesteps(num_train_timesteps, num_train_timesteps)
        self._step_index = None
        self._begin_index = None

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        if self.config.timestep_spacing in ["linspace", "trailing"]:
            return self.sigmas.max()

        return (self.sigmas.max() ** 2 + 1) ** 0.5

    @property
    def step_index(self):
        """
        The index counter for current timestep. It will increase 1 after each scheduler step.
        """
        return self._step_index

    @property
    def begin_index(self):
        """
        The index for the first timestep. It should be set from pipeline with `set_begin_index` method.
        """
        return self._begin_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.set_begin_index
    def set_begin_index(self, begin_index: int = 0):
        """
        Sets the begin index for the scheduler. This function should be run from pipeline before the inference.

        Args:
            begin_index (`int`):
                The begin index for the scheduler.
        """
        self._begin_index = begin_index

    def scale_model_input(
        self,
        sample: ms.Tensor,
        timestep: Union[float, ms.Tensor],
    ) -> ms.Tensor:
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
        if self.step_index is None:
            self._init_step_index(timestep)

        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
        else:
            sigma = self.sigmas_interpol[self.step_index - 1]

        sample = (sample / ((sigma**2 + 1) ** 0.5)).to(sample.dtype)
        return sample

    def set_timesteps(
        self,
        num_inference_steps: int,
        num_train_timesteps: Optional[int] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """
        self.num_inference_steps = num_inference_steps

        num_train_timesteps = num_train_timesteps or self.config.num_train_timesteps

        # "linspace", "leading", "trailing" corresponds to annotation of Table 2. of https://huggingface.co/papers/2305.08891
        if self.config.timestep_spacing == "linspace":
            timesteps = np.linspace(0, num_train_timesteps - 1, num_inference_steps, dtype=np.float32)[::-1].copy()
        elif self.config.timestep_spacing == "leading":
            step_ratio = num_train_timesteps // self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].copy().astype(np.float32)
            timesteps += self.config.steps_offset
        elif self.config.timestep_spacing == "trailing":
            step_ratio = num_train_timesteps / self.num_inference_steps
            # creates integer timesteps by multiplying by ratio
            # casting to int to avoid issues when num_inference_step is power of 3
            timesteps = (np.arange(num_train_timesteps, 0, -step_ratio)).round().copy().astype(np.float32)
            timesteps -= 1
        else:
            raise ValueError(
                f"{self.config.timestep_spacing} is not supported. Please make sure to choose one of 'linspace', 'leading' or 'trailing'."
            )

        sigmas = (((1 - self.alphas_cumprod) / self.alphas_cumprod) ** 0.5).asnumpy()
        log_sigmas = np.log(sigmas)

        sigmas = np.interp(timesteps, np.arange(0, len(sigmas)), sigmas)

        if self.config.use_karras_sigmas:
            sigmas = self._convert_to_karras(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas]).round().astype(np.float32)
        elif self.config.use_exponential_sigmas:
            sigmas = self._convert_to_exponential(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])
        elif self.config.use_beta_sigmas:
            sigmas = self._convert_to_beta(in_sigmas=sigmas, num_inference_steps=num_inference_steps)
            timesteps = np.array([self._sigma_to_t(sigma, log_sigmas) for sigma in sigmas])

        self.log_sigmas = ms.tensor(log_sigmas)
        sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        sigmas = ms.tensor(sigmas)

        # compute up and down sigmas
        sigmas_next = ms.tensor(np.roll(sigmas.asnumpy(), -1))
        sigmas_next[-1] = 0.0
        sigmas_up = (sigmas_next**2 * (sigmas**2 - sigmas_next**2) / sigmas**2) ** 0.5
        sigmas_down = (sigmas_next**2 - sigmas_up**2) ** 0.5
        sigmas_down[-1] = 0.0

        # compute interpolated sigmas
        sigmas_interpol = sigmas.log().lerp(sigmas_down.log(), 0.5).exp()
        sigmas_interpol[-2:] = 0.0

        # set sigmas
        self.sigmas = mint.cat([sigmas[:1], sigmas[1:].repeat_interleave(2), sigmas[-1:]])
        self.sigmas_interpol = mint.cat(
            [sigmas_interpol[:1], sigmas_interpol[1:].repeat_interleave(2), sigmas_interpol[-1:]]
        )
        self.sigmas_up = mint.cat([sigmas_up[:1], sigmas_up[1:].repeat_interleave(2), sigmas_up[-1:]])
        self.sigmas_down = mint.cat([sigmas_down[:1], sigmas_down[1:].repeat_interleave(2), sigmas_down[-1:]])

        timesteps = ms.tensor(timesteps)

        log_sigmas = self.log_sigmas
        timesteps_interpol = np.array(
            [self._sigma_to_t(sigma_interpol, log_sigmas.asnumpy()) for sigma_interpol in sigmas_interpol.asnumpy()]
        )

        timesteps_interpol = ms.tensor(timesteps_interpol, dtype=timesteps.dtype)
        interleaved_timesteps = mint.stack((timesteps_interpol[:-2, None], timesteps[1:, None]), dim=-1).flatten()

        self.timesteps = mint.cat([timesteps[:1], interleaved_timesteps])

        self.sample = None

        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma, log_sigmas):
        # get log sigma
        log_sigma = np.log(np.maximum(sigma, 1e-10))

        # get distribution
        dists = log_sigma - log_sigmas[:, np.newaxis]

        # get sigmas range
        low_idx = np.cumsum((dists >= 0), axis=0).argmax(axis=0).clip(max=log_sigmas.shape[0] - 2)
        high_idx = low_idx + 1

        low = log_sigmas[low_idx]
        high = log_sigmas[high_idx]

        # interpolate sigmas
        w = (low - log_sigma) / (low - high)
        w = np.clip(w, 0, 1)

        # transform interpolation to time range
        t = (1 - w) * low_idx + w * high_idx
        t = t.reshape(sigma.shape)
        return t

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_karras
    def _convert_to_karras(self, in_sigmas: ms.Tensor, num_inference_steps) -> ms.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        rho = 7.0  # 7.0 is the value used in the paper
        ramp = np.linspace(0, 1, num_inference_steps)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_exponential
    def _convert_to_exponential(self, in_sigmas: ms.Tensor, num_inference_steps: int) -> ms.Tensor:
        """Constructs an exponential noise schedule."""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.exp(np.linspace(math.log(sigma_max), math.log(sigma_min), num_inference_steps))
        return sigmas

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._convert_to_beta
    def _convert_to_beta(
        self, in_sigmas: ms.Tensor, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> ms.Tensor:
        """From "Beta Sampling is All You Need" [arXiv:2407.12173] (Lee et. al, 2024)"""

        # Hack to make sure that other schedulers which copy this function don't break
        # TODO: Add this logic to the other schedulers
        if hasattr(self.config, "sigma_min"):
            sigma_min = self.config.sigma_min
        else:
            sigma_min = None

        if hasattr(self.config, "sigma_max"):
            sigma_max = self.config.sigma_max
        else:
            sigma_max = None

        sigma_min = sigma_min if sigma_min is not None else in_sigmas[-1].item()
        sigma_max = sigma_max if sigma_max is not None else in_sigmas[0].item()

        sigmas = np.array(
            [
                sigma_min + (ppf * (sigma_max - sigma_min))
                for ppf in [
                    scipy.stats.beta.ppf(timestep, alpha, beta)
                    for timestep in 1 - np.linspace(0, 1, num_inference_steps)
                ]
            ]
        )
        return sigmas

    @property
    def state_in_first_order(self):
        return self.sample is None

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.index_for_timestep
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        if (schedule_timesteps == timestep).sum() > 1:
            pos = 1
        else:
            pos = 0

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        indices = (schedule_timesteps == timestep).nonzero()

        return int(indices[pos])

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler._init_step_index
    def _init_step_index(self, timestep):
        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: Union[ms.Tensor, np.ndarray],
        timestep: Union[float, ms.Tensor],
        sample: Union[ms.Tensor, np.ndarray],
        generator: Optional[np.random.Generator] = None,
        return_dict: bool = False,
    ) -> Union[KDPM2AncestralDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`ms.Tensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`np.random.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_k_dpm_2_ancestral_discrete.KDPM2AncestralDiscreteSchedulerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """
        if self.step_index is None:
            self._init_step_index(timestep)

        if self.state_in_first_order:
            sigma = self.sigmas[self.step_index]
            sigma_interpol = self.sigmas_interpol[self.step_index]
            sigma_up = self.sigmas_up[self.step_index]
            sigma_down = self.sigmas_down[self.step_index - 1]
        else:
            # 2nd order / KPDM2's method
            sigma = self.sigmas[self.step_index - 1]
            sigma_interpol = self.sigmas_interpol[self.step_index - 1]
            sigma_up = self.sigmas_up[self.step_index - 1]
            sigma_down = self.sigmas_down[self.step_index - 1]

        # currently only gamma=0 is supported. This usually works best anyways.
        # We can support gamma in the future but then need to scale the timestep before
        # passing it to the model which requires a change in API
        gamma = 0
        sigma_hat = sigma * (gamma + 1)  # Note: sigma_hat == sigma for now

        # 1. compute predicted original sample (x_0) from sigma-scaled predicted noise
        if self.config.prediction_type == "epsilon":
            sigma_input = sigma_hat if self.state_in_first_order else sigma_interpol
            pred_original_sample = sample - sigma_input.to(model_output.dtype) * model_output
        elif self.config.prediction_type == "v_prediction":
            sigma_input = sigma_hat if self.state_in_first_order else sigma_interpol
            pred_original_sample = (model_output * (-sigma_input / (sigma_input**2 + 1) ** 0.5)).to(
                model_output.dtype
            ) + (sample / (sigma_input**2 + 1)).to(sample.dtype)
        elif self.config.prediction_type == "sample":
            raise NotImplementedError("prediction_type not implemented yet: sample")
        else:
            raise ValueError(
                f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, or `v_prediction`"
            )

        if self.state_in_first_order:
            # 2. Convert to an ODE derivative for 1st order
            derivative = ((sample - pred_original_sample) / sigma_hat).to(sample.dtype)
            # 3. delta timestep
            dt = sigma_interpol - sigma_hat

            # store for 2nd order step
            self.sample = sample
            self.dt = dt
            prev_sample = sample + (derivative * dt).to(derivative.dtype)
        else:
            # DPM-Solver-2
            # 2. Convert to an ODE derivative for 2nd order
            derivative = ((sample - pred_original_sample) / sigma_interpol).to(sample.dtype)
            # 3. delta timestep
            dt = sigma_down - sigma_hat

            sample = self.sample
            self.sample = None

            prev_sample = sample + (derivative * dt).to(derivative.dtype)
            noise = randn_tensor(model_output.shape, dtype=model_output.dtype, generator=generator)
            prev_sample = prev_sample + (noise * sigma_up).to(noise.dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (
                prev_sample,
                pred_original_sample,
            )

        return KDPM2AncestralDiscreteSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)

    # Copied from diffusers.schedulers.scheduling_euler_discrete.EulerDiscreteScheduler.add_noise
    def add_noise(
        self,
        original_samples: ms.Tensor,
        noise: ms.Tensor,
        timesteps: ms.Tensor,
    ) -> ms.Tensor:
        broadcast_shape = original_samples.shape
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(dtype=original_samples.dtype)
        schedule_timesteps = self.timesteps

        # self.begin_index is None when scheduler is used for training, or pipeline does not implement set_begin_index
        if self.begin_index is None:
            step_indices = [self.index_for_timestep(t, schedule_timesteps) for t in timesteps]
        elif self.step_index is not None:
            # add_noise is called after first denoising step (for inpainting)
            step_indices = [self.step_index] * timesteps.shape[0]
        else:
            # add noise is called before first denoising step to create initial latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        # while len(sigma.shape) < len(original_samples.shape):
        #     sigma = sigma.unsqueeze(-1)
        sigma = mint.reshape(sigma, (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
