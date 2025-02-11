# Copyright 2024 TSAIL Team and The HuggingFace Team. All rights reserved.
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

# DISCLAIMER: This file is strongly influenced by https://github.com/LuChengTHU/dpm-solver and https://github.com/NVlabs/edm

import math
from typing import List, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import ops

from ..configuration_utils import ConfigMixin, register_to_config
from ..utils.mindspore_utils import randn_tensor
from .scheduling_utils import SchedulerMixin, SchedulerOutput


class EDMDPMSolverMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    Implements DPMSolverMultistepScheduler in EDM formulation as presented in Karras et al. 2022 [1].
    `EDMDPMSolverMultistepScheduler` is a fast dedicated high-order solver for diffusion ODEs.

    [1] Karras, Tero, et al. "Elucidating the Design Space of Diffusion-Based Generative Models."
    https://arxiv.org/abs/2206.00364

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        sigma_min (`float`, *optional*, defaults to 0.002):
            Minimum noise magnitude in the sigma schedule. This was set to 0.002 in the EDM paper [1]; a reasonable
            range is [0, 10].
        sigma_max (`float`, *optional*, defaults to 80.0):
            Maximum noise magnitude in the sigma schedule. This was set to 80.0 in the EDM paper [1]; a reasonable
            range is [0.2, 80.0].
        sigma_data (`float`, *optional*, defaults to 0.5):
            The standard deviation of the data distribution. This is set to 0.5 in the EDM paper [1].
        sigma_schedule (`str`, *optional*, defaults to `karras`):
            Sigma schedule to compute the `sigmas`. By default, we the schedule introduced in the EDM paper
            (https://arxiv.org/abs/2206.00364). Other acceptable value is "exponential". The exponential schedule was
            incorporated in this model: https://huggingface.co/stabilityai/cosxl.
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        solver_order (`int`, defaults to 2):
            The DPMSolver order which can be `1` or `2` or `3`. It is recommended to use `solver_order=2` for guided
            sampling, and `solver_order=3` for unconditional sampling.
        prediction_type (`str`, defaults to `epsilon`, *optional*):
            Prediction type of the scheduler function; can be `epsilon` (predicts the noise of the diffusion process),
            `sample` (directly predicts the noisy sample`) or `v_prediction` (see section 2.4 of [Imagen
            Video](https://imagen.research.google/video/paper.pdf) paper).
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True` and
            `algorithm_type="dpmsolver++"`.
        algorithm_type (`str`, defaults to `dpmsolver++`):
            Algorithm type for the solver; can be `dpmsolver++` or `sde-dpmsolver++`. The `dpmsolver++` type implements
            the algorithms in the [DPMSolver++](https://huggingface.co/papers/2211.01095) paper. It is recommended to
            use `dpmsolver++` or `sde-dpmsolver++` with `solver_order=2` for guided sampling like in Stable Diffusion.
        solver_type (`str`, defaults to `midpoint`):
            Solver type for the second-order solver; can be `midpoint` or `heun`. The solver type slightly affects the
            sample quality, especially for a small number of steps. It is recommended to use `midpoint` solvers.
        lower_order_final (`bool`, defaults to `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        euler_at_final (`bool`, defaults to `False`):
            Whether to use Euler's method in the final step. It is a trade-off between numerical stability and detail
            richness. This can stabilize the sampling of the SDE variant of DPMSolver for small number of inference
            steps, but sometimes may result in blurring.
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
    """

    _compatibles = []
    order = 1

    @register_to_config
    def __init__(
        self,
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        sigma_data: float = 0.5,
        sigma_schedule: str = "karras",
        num_train_timesteps: int = 1000,
        prediction_type: str = "epsilon",
        rho: float = 7.0,
        solver_order: int = 2,
        thresholding: bool = False,
        dynamic_thresholding_ratio: float = 0.995,
        sample_max_value: float = 1.0,
        algorithm_type: str = "dpmsolver++",
        solver_type: str = "midpoint",
        lower_order_final: bool = True,
        euler_at_final: bool = False,
        final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
    ):
        # settings for DPM-Solver
        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"]:
            if algorithm_type == "deis":
                self.register_to_config(algorithm_type="dpmsolver++")
            else:
                raise NotImplementedError(f"{algorithm_type} is not implemented for {self.__class__}")

        if solver_type not in ["midpoint", "heun"]:
            if solver_type in ["logrho", "bh1", "bh2"]:
                self.register_to_config(solver_type="midpoint")
            else:
                raise NotImplementedError(f"{solver_type} is not implemented for {self.__class__}")

        if algorithm_type not in ["dpmsolver++", "sde-dpmsolver++"] and final_sigmas_type == "zero":
            raise ValueError(
                f"`final_sigmas_type` {final_sigmas_type} is not supported for `algorithm_type` {algorithm_type}. Please choose `sigma_min` instead."
            )

        ramp = ms.tensor(np.linspace(0, 1, num_train_timesteps), dtype=ms.float32)
        if sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(ramp)
        elif sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(ramp)

        self.timesteps = self.precondition_noise(sigmas)

        self.sigmas = ops.cat([sigmas, ops.zeros(1, dtype=sigmas.dtype)])

        # setable values
        self.num_inference_steps = None
        self.model_outputs = [None] * solver_order
        self.lower_order_nums = 0
        self._step_index = None
        self._begin_index = None
        self.sigma_data = self.config.sigma_data

    @property
    def init_noise_sigma(self):
        # standard deviation of the initial noise distribution
        return (self.config.sigma_max**2 + 1) ** 0.5

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

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_inputs
    def precondition_inputs(self, sample, sigma):
        c_in = 1 / ((sigma**2 + self.sigma_data**2) ** 0.5)
        scaled_sample = (sample * c_in).to(sample.dtype)
        return scaled_sample

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_noise
    def precondition_noise(self, sigma):
        if not isinstance(sigma, ms.Tensor):
            sigma = ms.tensor(sigma)

        c_noise = 0.25 * ops.log(sigma)

        return c_noise

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.precondition_outputs
    def precondition_outputs(self, sample, model_output, sigma):
        sigma_data = self.sigma_data
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)

        if self.config.prediction_type == "epsilon":
            c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        elif self.config.prediction_type == "v_prediction":
            c_out = -sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
        else:
            raise ValueError(f"Prediction type {self.config.prediction_type} is not supported.")

        denoised = c_skip.to(sample.dtype) * sample + c_out.to(sample.dtype) * model_output

        return denoised

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler.scale_model_input
    def scale_model_input(self, sample: ms.Tensor, timestep: Union[float, ms.Tensor]) -> ms.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep. Scales the denoising model input by `(sigma**2 + 1) ** 0.5` to match the Euler algorithm.

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

        sigma = self.sigmas[self.step_index]
        sample = self.precondition_inputs(sample, sigma)

        self.is_scale_input_called = True
        return sample

    def set_timesteps(self, num_inference_steps: int = None):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model.
        """

        self.num_inference_steps = num_inference_steps

        ramp = ms.tensor(np.linspace(0, 1, self.num_inference_steps))
        if self.config.sigma_schedule == "karras":
            sigmas = self._compute_karras_sigmas(ramp)
        elif self.config.sigma_schedule == "exponential":
            sigmas = self._compute_exponential_sigmas(ramp)

        sigmas = sigmas.to(ms.float32)
        self.timesteps = self.precondition_noise(sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = self.config.sigma_min
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        self.sigmas = ops.cat([sigmas, ms.tensor([sigma_last], dtype=ms.float32)])

        self.model_outputs = [
            None,
        ] * self.config.solver_order
        self.lower_order_nums = 0

        # add an index counter for schedulers that allow duplicated timesteps
        self._step_index = None
        self._begin_index = None

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_karras_sigmas
    def _compute_karras_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> ms.Tensor:
        """Constructs the noise schedule of Karras et al. (2022)."""

        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max

        rho = self.config.rho
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return sigmas

    # Copied from diffusers.schedulers.scheduling_edm_euler.EDMEulerScheduler._compute_exponential_sigmas
    def _compute_exponential_sigmas(self, ramp, sigma_min=None, sigma_max=None) -> ms.Tensor:
        """Implementation closely follows k-diffusion.

        https://github.com/crowsonkb/k-diffusion/blob/6ab5146d4a5ef63901326489f31f1d8e7dd36b48/k_diffusion/sampling.py#L26
        """
        sigma_min = sigma_min or self.config.sigma_min
        sigma_max = sigma_max or self.config.sigma_max
        sigmas = ms.tensor(np.linspace(math.log(sigma_min), math.log(sigma_max), len(ramp))).exp().flip((0,))
        return sigmas

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: ms.Tensor) -> ms.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (ms.float32, ms.float64):
            sample = sample.float()  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims).item())

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = ms.Tensor.from_numpy(np.quantile(abs_sample.asnumpy(), self.config.dynamic_thresholding_ratio, axis=1))
        s = ops.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = ops.clamp(sample, -s, s) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

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

    def _sigma_to_alpha_sigma_t(self, sigma):
        alpha_t = ms.tensor(1, dtype=ms.float32)  # Inputs are pre-scaled before going into unet, so alpha_t = 1
        sigma_t = sigma

        return alpha_t, sigma_t

    def convert_model_output(
        self,
        model_output: ms.Tensor,
        sample: ms.Tensor = None,
    ) -> ms.Tensor:
        """
        Convert the model output to the corresponding type the DPMSolver/DPMSolver++ algorithm needs. DPM-Solver is
        designed to discretize an integral of the noise prediction model, and DPM-Solver++ is designed to discretize an
        integral of the data prediction model.

        <Tip>

        The algorithm and model type are decoupled. You can use either DPMSolver or DPMSolver++ for both noise
        prediction and data prediction models.

        </Tip>

        Args:
            model_output (`ms.Tensor`):
                The direct output from the learned diffusion model.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `ms.Tensor`:
                The converted model output.
        """
        sigma = self.sigmas[self.step_index]
        x0_pred = self.precondition_outputs(sample, model_output, sigma)

        if self.config.thresholding:
            x0_pred = self._threshold_sample(x0_pred)

        return x0_pred

    def dpm_solver_first_order_update(
        self,
        model_output: ms.Tensor,
        sample: ms.Tensor = None,
        noise: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """
        One step for the first-order DPMSolver (equivalent to DDIM).

        Args:
            model_output (`ms.Tensor`):
                The direct output from the learned diffusion model.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `ms.Tensor`:
                The sample tensor at the previous timestep.
        """
        dtype = sample.dtype
        sigma_t, sigma_s = self.sigmas[self.step_index + 1], self.sigmas[self.step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s, sigma_s = self._sigma_to_alpha_sigma_t(sigma_s)
        lambda_t = ops.log(alpha_t) - ops.log(sigma_t)
        lambda_s = ops.log(alpha_s) - ops.log(sigma_s)

        h = lambda_t - lambda_s
        if self.config.algorithm_type == "dpmsolver++":
            x_t = (sigma_t / sigma_s).to(dtype) * sample - (alpha_t * (ops.exp(-h) - 1.0)).to(dtype) * model_output
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            x_t = (
                (sigma_t / sigma_s * ops.exp(-h)).to(dtype) * sample
                + (alpha_t * (1 - ops.exp(-2.0 * h))).to(dtype) * model_output
                + (sigma_t * ops.sqrt(1.0 - ops.exp(-2 * h))).to(dtype) * noise
            )

        return x_t

    def multistep_dpm_solver_second_order_update(
        self,
        model_output_list: List[ms.Tensor],
        sample: ms.Tensor = None,
        noise: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        """
        One step for the second-order multistep DPMSolver.

        Args:
            model_output_list (`List[ms.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process.

        Returns:
            `ms.Tensor`:
                The sample tensor at the previous timestep.
        """
        dtype = sample.dtype
        sigma_t, sigma_s0, sigma_s1 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)

        lambda_t = ops.log(alpha_t) - ops.log(sigma_t)
        lambda_s0 = ops.log(alpha_s0) - ops.log(sigma_s0)
        lambda_s1 = ops.log(alpha_s1) - ops.log(sigma_s1)

        m0, m1 = model_output_list[-1], model_output_list[-2]

        h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
        r0 = h_0 / h
        D0, D1 = m0, (1.0 / r0).to(dtype) * (m0 - m1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2211.01095 for detailed derivations
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0).to(dtype) * sample
                    - (alpha_t * (ops.exp(-h) - 1.0)).to(dtype) * D0
                    - (0.5 * (alpha_t * (ops.exp(-h) - 1.0))).to(dtype) * D1
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0).to(dtype) * sample
                    - (alpha_t * (ops.exp(-h) - 1.0)).to(dtype) * D0
                    + (alpha_t * ((ops.exp(-h) - 1.0) / h + 1.0)).to(dtype) * D1
                )
        elif self.config.algorithm_type == "sde-dpmsolver++":
            assert noise is not None
            if self.config.solver_type == "midpoint":
                x_t = (
                    (sigma_t / sigma_s0 * ops.exp(-h)).to(dtype) * sample
                    + (alpha_t * (1 - ops.exp(-2.0 * h))).to(dtype) * D0
                    + (0.5 * (alpha_t * (1 - ops.exp(-2.0 * h)))).to(dtype) * D1
                    + (sigma_t * ops.sqrt(1.0 - ops.exp(-2 * h))).to(dtype) * noise
                )
            elif self.config.solver_type == "heun":
                x_t = (
                    (sigma_t / sigma_s0 * ops.exp(-h)).to(dtype) * sample
                    + (alpha_t * (1 - ops.exp(-2.0 * h))).to(dtype) * D0
                    + (alpha_t * ((1.0 - ops.exp(-2.0 * h)) / (-2.0 * h) + 1.0)).to(dtype) * D1
                    + (sigma_t * ops.sqrt(1.0 - ops.exp(-2 * h))).to(dtype) * noise
                )

        return x_t

    def multistep_dpm_solver_third_order_update(
        self,
        model_output_list: List[ms.Tensor],
        sample: ms.Tensor = None,
    ) -> ms.Tensor:
        """
        One step for the third-order multistep DPMSolver.

        Args:
            model_output_list (`List[ms.Tensor]`):
                The direct outputs from learned diffusion model at current and latter timesteps.
            sample (`ms.Tensor`):
                A current instance of a sample created by diffusion process.

        Returns:
            `ms.Tensor`:
                The sample tensor at the previous timestep.
        """
        dtype = sample.dtype
        sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
            self.sigmas[self.step_index + 1],
            self.sigmas[self.step_index],
            self.sigmas[self.step_index - 1],
            self.sigmas[self.step_index - 2],
        )

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)
        alpha_s1, sigma_s1 = self._sigma_to_alpha_sigma_t(sigma_s1)
        alpha_s2, sigma_s2 = self._sigma_to_alpha_sigma_t(sigma_s2)

        lambda_t = ops.log(alpha_t) - ops.log(sigma_t)
        lambda_s0 = ops.log(alpha_s0) - ops.log(sigma_s0)
        lambda_s1 = ops.log(alpha_s1) - ops.log(sigma_s1)
        lambda_s2 = ops.log(alpha_s2) - ops.log(sigma_s2)

        m0, m1, m2 = model_output_list[-1], model_output_list[-2], model_output_list[-3]

        h, h_0, h_1 = lambda_t - lambda_s0, lambda_s0 - lambda_s1, lambda_s1 - lambda_s2
        r0, r1 = h_0 / h, h_1 / h
        D0 = m0
        D1_0, D1_1 = (1.0 / r0).to(dtype) * (m0 - m1), (1.0 / r1).to(dtype) * (m1 - m2)
        D1 = D1_0 + (r0 / (r0 + r1)).to(dtype) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)).to(dtype) * (D1_0 - D1_1)
        if self.config.algorithm_type == "dpmsolver++":
            # See https://arxiv.org/abs/2206.00927 for detailed derivations
            x_t = (
                (sigma_t / sigma_s0).to(dtype) * sample
                - (alpha_t * (ops.exp(-h) - 1.0)).to(dtype) * D0
                + (alpha_t * ((ops.exp(-h) - 1.0) / h + 1.0)).to(dtype) * D1
                - (alpha_t * ((ops.exp(-h) - 1.0 + h) / h**2 - 0.5)).to(dtype) * D2
            )

        return x_t

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.index_for_timestep
    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        index_candidates_num = (schedule_timesteps == timestep).sum()

        if index_candidates_num == 0:
            step_index = len(self.timesteps) - 1
        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        else:
            if index_candidates_num > 1:
                pos = 1
            else:
                pos = 0
            step_index = int((schedule_timesteps == timestep).nonzero()[pos])

        return step_index

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler._init_step_index
    def _init_step_index(self, timestep):
        """
        Initialize the step_index counter for the scheduler.
        """

        if self.begin_index is None:
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
        self,
        model_output: ms.Tensor,
        timestep: int,
        sample: ms.Tensor,
        generator=None,
        return_dict: bool = False,
    ) -> Union[SchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep DPMSolver.

        Args:
            model_output (`ms.Tensor`):
                The direct output from learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`ms.Tensor`):
                A current instance of a sample created by the diffusion process.
            generator (`np.random.Generator`, *optional*):
                A random number generator.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Improve numerical stability for small number of steps
        lower_order_final = (self.step_index == len(self.timesteps) - 1) and (
            self.config.euler_at_final
            or (self.config.lower_order_final and len(self.timesteps) < 15)
            or self.config.final_sigmas_type == "zero"
        )
        lower_order_second = (
            (self.step_index == len(self.timesteps) - 2) and self.config.lower_order_final and len(self.timesteps) < 15
        )

        model_output = self.convert_model_output(model_output, sample=sample)
        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

        if self.config.algorithm_type == "sde-dpmsolver++":
            noise = randn_tensor(model_output.shape, generator=generator, dtype=model_output.dtype)
        else:
            noise = None

        if self.config.solver_order == 1 or self.lower_order_nums < 1 or lower_order_final:
            prev_sample = self.dpm_solver_first_order_update(model_output, sample=sample, noise=noise)
        elif self.config.solver_order == 2 or self.lower_order_nums < 2 or lower_order_second:
            prev_sample = self.multistep_dpm_solver_second_order_update(self.model_outputs, sample=sample, noise=noise)
        else:
            prev_sample = self.multistep_dpm_solver_third_order_update(self.model_outputs, sample=sample)

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

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
            # add noise is called bevore first denoising step to create inital latent(img2img)
            step_indices = [self.begin_index] * timesteps.shape[0]

        sigma = sigmas[step_indices].flatten()
        # while len(sigma.shape) < len(original_samples.shape):
        #     sigma = sigma.unsqueeze(-1)
        sigma = ops.reshape(sigma, (timesteps.shape[0],) + (1,) * (len(broadcast_shape) - 1))

        noisy_samples = original_samples + noise * sigma
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
