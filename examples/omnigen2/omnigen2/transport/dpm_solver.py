# Adapted from https://github.com/VectorSpaceLab/OmniGen2/blob/main/omnigen2/transport/dpm_solver.py
# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import os

from tqdm import tqdm

import mindspore as ms
from mindspore import mint, ops


class NoiseScheduleFlow:
    def __init__(self, schedule="discrete_flow"):
        """Create a wrapper class for the forward SDE (EDM type)."""
        self.T = 1
        self.t0 = 0.001
        self.schedule = schedule  # ['continuous', 'discrete_flow']
        self.total_N = 1000

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        return mint.log(self.marginal_alpha(t))

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return 1 - t

    @staticmethod
    def marginal_std(t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return t

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = mint.log(self.marginal_std(t))
        return log_mean_coeff - log_std

    @staticmethod
    def inverse_lambda(lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        return mint.exp(-lamb)


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    interval_guidance=[0, 1.0],
    classifier_fn=None,
    classifier_kwargs={},
):
    """Create a wrapper function for the noise prediction model.

    DPM-Solver needs to solve the continuous-time diffusion ODEs. For DPMs trained on discrete-time labels, we need to
    firstly wrap the model function to a noise prediction model that accepts the continuous time as the input.

    We support four types of the diffusion model by setting `model_type`:

        1. "noise": noise prediction model. (Trained by predicting noise).

        2. "x_start": data prediction model. (Trained by predicting the data x_0 at time 0).

        3. "v": velocity prediction model. (Trained by predicting the velocity).
            The "v" prediction is derivation detailed in Appendix D of [1], and is used in Imagen-Video [2].

            [1] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
                arXiv preprint arXiv:2202.00512 (2022).
            [2] Ho, Jonathan, et al. "Imagen Video: High Definition Video Generation with Diffusion Models."
                arXiv preprint arXiv:2210.02303 (2022).

        4. "score": marginal score function. (Trained by denoising score matching).
            Note that the score function and the noise prediction model follows a simple relationship:
            ```
                noise(x_t, t) = -sigma_t * score(x_t, t)
            ```

    We support three types of guided sampling by DPMs by setting `guidance_type`:
        1. "uncond": unconditional sampling by DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

        2. "classifier": classifier guidance sampling [3] by DPMs and another classifier.
            The input `model` has the following format:
            ``
                model(x, t_input, **model_kwargs) -> noise | x_start | v | score
            ``

            The input `classifier_fn` has the following format:
            ``
                classifier_fn(x, t_input, cond, **classifier_kwargs) -> logits(x, t_input, cond)
            ``

            [3] P. Dhariwal and A. Q. Nichol, "Diffusion models beat GANs on image synthesis,"
                in Advances in Neural Information Processing Systems, vol. 34, 2021, pp. 8780-8794.

        3. "classifier-free": classifier-free guidance sampling by conditional DPMs.
            The input `model` has the following format:
            ``
                model(x, t_input, cond, **model_kwargs) -> noise | x_start | v | score
            ``
            And if cond == `unconditional_condition`, the model output is the unconditional DPM output.

            [4] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance."
                arXiv preprint arXiv:2207.12598 (2022).


    The `t_input` is the time label of the model, which may be discrete-time labels (i.e. 0 to 999)
    or continuous-time labels (i.e. epsilon to T).

    We wrap the model function to accept only `x` and `t_continuous` as inputs, and outputs the predicted noise:
    ``
        def model_fn(x, t_continuous) -> noise:
            t_input = get_model_input_time(t_continuous)
            return noise_pred(model, x, t_input, **model_kwargs)
    ``
    where `t_continuous` is the continuous time labels (i.e. epsilon to T). And we use `model_fn` for DPM-Solver.

    ===============================================================

    Args:
        model: A diffusion model with the corresponding format described above.
        noise_schedule: A noise schedule object, such as NoiseScheduleVP.
        model_type: A `str`. The parameterization type of the diffusion model.
                    "noise" or "x_start" or "v" or "score".
        model_kwargs: A `dict`. A dict for the other inputs of the model function.
        guidance_type: A `str`. The type of the guidance for sampling.
                    "uncond" or "classifier" or "classifier-free".
        condition: A mindspore tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A mindspore tensor. The condition for the unconditional sampling.
                    Only used for "classifier-free" guidance type.
        guidance_scale: A `float`. The scale for the guided sampling.
        classifier_fn: A classifier function. Only used for the classifier guidance.
        classifier_kwargs: A `dict`. A dict for the other inputs of the classifier function.
    Returns:
        A noise prediction model that accepts the noised data and the continuous time as the inputs.
    """

    def get_model_input_time(t_continuous):
        """
        Convert the continuous-time `t_continuous` (in [epsilon, T]) to the model input time.
        For discrete-time DPMs, we convert `t_continuous` in [1 / N, 1] to `t_input` in [0, 1000 * (N - 1) / N].
        For continuous-time DPMs, we just use `t_continuous`.
        """
        if noise_schedule.schedule == "discrete":
            return (t_continuous - 1.0 / noise_schedule.total_N) * noise_schedule.total_N
        elif noise_schedule.schedule == "discrete_flow":
            return t_continuous * noise_schedule.total_N
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return (x - expand_dims(alpha_t, x.dim()) * output) / expand_dims(sigma_t, x.dim())
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            return expand_dims(alpha_t, x.dim()) * output + expand_dims(sigma_t, x.dim()) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            return -expand_dims(sigma_t, x.dim()) * output
        elif model_type == "flow":
            _, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            try:
                noise = (1 - expand_dims(sigma_t, x.dim()).to(x.dtype)) * output + x
            except Exception:
                noise = (1 - expand_dims(sigma_t, x.dim()).to(x.dtype)) * output[0] + x
            return noise

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        # with torch.enable_grad():
        #     x_in = x.detach().requires_grad_(True)
        #     log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
        #     return torch.autograd.grad(log_prob.sum(), x_in)[0]
        raise NotImplementedError

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        guidance_tp = guidance_type
        if guidance_tp == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_tp == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, x.dim()) * cond_grad
        elif guidance_tp == "classifier-free":
            if (
                guidance_scale == 1.0
                or unconditional_condition is None
                or not (interval_guidance[0] < t_continuous[0] < interval_guidance[1])
            ):
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = mint.cat([x] * 2)
                t_in = mint.cat([t_continuous] * 2)
                c_in = mint.cat([unconditional_condition, condition])
                try:
                    noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in).chunk(2)
                except Exception:
                    noise_uncond, noise = noise_pred_fn(x_in, t_in, cond=c_in)[0].chunk(2)
                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v", "score", "flow"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class DPM_Solver:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        algorithm_type="dpmsolver++",
        correcting_x0_fn=None,
        correcting_xt_fn=None,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
    ):
        """Construct a DPM-Solver.

        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

        We also support the "dynamic thresholding" method in Imagen[1]. For pixel-space diffusion models, you
        can set both `algorithm_type="dpmsolver++"` and `correcting_x0_fn="dynamic_thresholding"` to use the
        dynamic thresholding. The "dynamic thresholding" can greatly improve the sample quality for pixel-space
        DPMs with large guidance scales. Note that the thresholding method is **unsuitable** for latent-space
        DPMs (such as stable-diffusion).

        To support advanced algorithms in image-to-image applications, we also support corrector functions for
        both x0 and xt.

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            noise_schedule: A noise schedule object, such as NoiseScheduleVP.
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".
            correcting_x0_fn: A `str` or a function with the following format:
                ```
                def correcting_x0_fn(x0, t):
                    x0_new = ...
                    return x0_new
                ```
                This function is to correct the outputs of the data prediction model at each sampling step. e.g.,
                ```
                x0_pred = data_pred_model(xt, t)
                if correcting_x0_fn is not None:
                    x0_pred = correcting_x0_fn(x0_pred, t)
                xt_1 = update(x0_pred, xt, t)
                ```
                If `correcting_x0_fn="dynamic_thresholding"`, we use the dynamic thresholding proposed in Imagen[1].
            correcting_xt_fn: A function with the following format:
                ```
                def correcting_xt_fn(xt, t, step):
                    x_new = ...
                    return x_new
                ```
                This function is to correct the intermediate samples xt at each sampling step. e.g.,
                ```
                xt = ...
                xt = correcting_xt_fn(xt, t, step)
                ```
            thresholding_max_val: A `float`. The max value for thresholding.
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.
            dynamic_thresholding_ratio: A `float`. The ratio for dynamic thresholding (see Imagen[1] for details).
                Valid only when use `dpmsolver++` and `correcting_x0_fn="dynamic_thresholding"`.

        [1] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily Denton, Seyed Kamyar Seyed Ghasemipour,
            Burcu Karagol Ayan, S Sara Mahdavi, Rapha Gontijo Lopes, et al. Photorealistic text-to-image diffusion models
            with deep language understanding. arXiv preprint arXiv:2205.11487, 2022b.
        """
        self.model = lambda x, t: model_fn(x, t.expand(x.shape[0]))
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type
        if correcting_x0_fn == "dynamic_thresholding":
            self.correcting_x0_fn = self.dynamic_thresholding_fn
        else:
            self.correcting_x0_fn = correcting_x0_fn
        self.correcting_xt_fn = correcting_xt_fn
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio
        self.thresholding_max_val = thresholding_max_val
        self.register_progress_bar()

    def register_progress_bar(self, progress_fn=None):
        """
        Register a progress bar callback function

        Args:
            progress_fn: Callback function that takes current step and total steps as parameters
        """
        self.progress_fn = progress_fn if progress_fn is not None else lambda step, total: None

    def update_progress(self, step, total_steps):
        """
        Update sampling progress

        Args:
            step: Current step number
            total_steps: Total number of steps
        """
        if hasattr(self, "progress_fn"):
            try:
                self.progress_fn(step / total_steps, desc=f"Generating {step}/{total_steps}")
            except Exception:
                self.progress_fn(step, total_steps)

        else:
            # If no progress_fn registered, use default empty function
            pass

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.dim()
        p = self.dynamic_thresholding_ratio
        s = ops.quantile(mint.abs(x0).reshape((x0.shape[0], -1)), p, axis=1)
        s = expand_dims(mint.maximum(s, self.thresholding_max_val * mint.ones_like(s)), dims)
        x0 = mint.clamp(x0, -s, s) / s
        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with corrector).
        """
        noise = self.noise_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t
        if self.correcting_x0_fn is not None:
            x0 = self.correcting_x0_fn(x0, t)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, shift=1.0):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
        Returns:
            A mindspore tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(ms.tensor(t_T))
            lambda_0 = self.noise_schedule.marginal_lambda(ms.tensor(t_0))
            logSNR_steps = mint.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return mint.linspace(t_T, t_0, N + 1)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = mint.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order)
            return t
        elif skip_type == "time_uniform_flow":
            betas = mint.linspace(t_T, t_0, N + 1)
            sigmas = 1.0 - betas
            sigmas = (shift * sigmas / (1 + (shift - 1) * sigmas)).flip(dims=[0])
            return sigmas
        else:
            raise ValueError(
                f"Unsupported skip_type {skip_type}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'"
            )

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3] * (K - 1) + [1]
            else:
                orders = [3] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2] * K
            else:
                K = steps // 2 + 1
                orders = [2] * (K - 1) + [1]
        elif order == 1:
            K = 1
            orders = [1] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps)[mint.cumsum(ms.tensor([0] + orders), 0)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A mindspore tensor. The initial value at time `s`.
            s: A mindspore tensor. The starting time, with the shape (1,).
            t: A mindspore tensor. The ending time, with the shape (1,).
            model_s: A mindspore tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A mindspore tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.marginal_log_mean_coeff(s), ns.marginal_log_mean_coeff(t)
        sigma_s, sigma_t = ns.marginal_std(s), ns.marginal_std(t)
        alpha_t = mint.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = mint.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t
        else:
            phi_1 = mint.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = mint.exp(log_alpha_t - log_alpha_s) * x - (sigma_t * phi_1) * model_s
            if return_intermediate:
                return x_t, {"model_s": model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(
        self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type="dpmsolver"
    ):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A mindspore tensor. The initial value at time `s`.
            s: A mindspore tensor. The starting time, with the shape (1,).
            t: A mindspore tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A mindspore tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A mindspore tensor. The approximated solution at time `t`.
        """
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError(f"'solver_type' must be either 'dpmsolver' or 'taylor', got {solver_type}")
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_t = ns.marginal_std(s), ns.marginal_std(s1), ns.marginal_std(t)
        alpha_s1, alpha_t = mint.exp(log_alpha_s1), mint.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = mint.expm1(-r1 * h)
            phi_1 = mint.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpmsolver":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1.0 / r1) * (alpha_t * (phi_1 / h + 1.0)) * (model_s1 - model_s)
                )
        else:
            phi_11 = mint.expm1(r1 * h)
            phi_1 = mint.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = mint.exp(log_alpha_s1 - log_alpha_s) * x - (sigma_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpmsolver":
                x_t = (
                    mint.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    mint.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1.0 / r1) * (sigma_t * (phi_1 / h - 1.0)) * (model_s1 - model_s)
                )
        if return_intermediate:
            return x_t, {"model_s": model_s, "model_s1": model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(
        self,
        x,
        s,
        t,
        r1=1.0 / 3.0,
        r2=2.0 / 3.0,
        model_s=None,
        model_s1=None,
        return_intermediate=False,
        solver_type="dpmsolver",
    ):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A mindspore tensor. The initial value at time `s`.
            s: A mindspore tensor. The starting time, with the shape (1,).
            t: A mindspore tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A mindspore tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A mindspore tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A mindspore tensor. The approximated solution at time `t`.
        """
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError(f"'solver_type' must be either 'dpmsolver' or 'taylor', got {solver_type}")
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.marginal_lambda(s), ns.marginal_lambda(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = (
            ns.marginal_log_mean_coeff(s),
            ns.marginal_log_mean_coeff(s1),
            ns.marginal_log_mean_coeff(s2),
            ns.marginal_log_mean_coeff(t),
        )
        sigma_s, sigma_s1, sigma_s2, sigma_t = (
            ns.marginal_std(s),
            ns.marginal_std(s1),
            ns.marginal_std(s2),
            ns.marginal_std(t),
        )
        alpha_s1, alpha_s2, alpha_t = mint.exp(log_alpha_s1), mint.exp(log_alpha_s2), mint.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = mint.expm1(-r1 * h)
            phi_12 = mint.expm1(-r2 * h)
            phi_1 = mint.expm1(-h)
            phi_22 = mint.expm1(-r2 * h) / (r2 * h) + 1.0
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpmsolver":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1.0 / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (model_s1 - model_s)
                D1_1 = (1.0 / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = mint.expm1(r1 * h)
            phi_12 = mint.expm1(r2 * h)
            phi_1 = mint.expm1(h)
            phi_22 = mint.expm1(r2 * h) / (r2 * h) - 1.0
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (mint.exp(log_alpha_s1 - log_alpha_s)) * x - (sigma_s1 * phi_11) * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (mint.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpmsolver":
                x_t = (
                    (mint.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1.0 / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (model_s1 - model_s)
                D1_1 = (1.0 / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (mint.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        if return_intermediate:
            return x_t, {"model_s": model_s, "model_s1": model_s1, "model_s2": model_s2}
        else:
            return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A mindspore tensor. The initial value at time `s`.
            model_prev_list: A list of mindspore tensor. The previous computed model values.
            t_prev_list: A list of mindspore tensor. The previous times, each time has the shape (1,)
            t: A mindspore tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A mindspore tensor. The approximated solution at time `t`.
        """
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError(f"'solver_type' must be either 'dpmsolver' or 'taylor', got {solver_type}")
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = mint.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = mint.expm1(-h)
            if solver_type == "dpmsolver":
                x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.0)) * D1_0
                )
        else:
            phi_1 = mint.expm1(h)
            if solver_type == "dpmsolver":
                x_t = (
                    (mint.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == "taylor":
                x_t = (
                    (mint.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.0)) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A mindspore tensor. The initial value at time `s`.
            model_prev_list: A list of mindspore tensor. The previous computed model values.
            t_prev_list: A list of mindspore tensor. The previous times, each time has the shape (1,)
            t: A mindspore tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A mindspore tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.marginal_lambda(t_prev_2),
            ns.marginal_lambda(t_prev_1),
            ns.marginal_lambda(t_prev_0),
            ns.marginal_lambda(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = mint.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = mint.expm1(-h)
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = mint.expm1(h)
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (mint.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        return x_t

    def singlestep_dpm_solver_update(
        self, x, s, t, order, return_intermediate=False, solver_type="dpmsolver", r1=None, r2=None
    ):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A mindspore tensor. The initial value at time `s`.
            s: A mindspore tensor. The starting time, with the shape (1,).
            t: A mindspore tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            return_intermediate: A `bool`. If true, also return the model value at time `s`, `s1` and `s2` (the intermediate times).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A mindspore tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t, return_intermediate=return_intermediate)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(
                x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1
            )
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(
                x, s, t, return_intermediate=return_intermediate, solver_type=solver_type, r1=r1, r2=r2
            )
        else:
            raise ValueError(f"Solver order must be 1 or 2 or 3, got {order}")

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type="dpmsolver"):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A mindspore tensor. The initial value at time `s`.
            model_prev_list: A list of mindspore tensor. The previous computed model values.
            t_prev_list: A list of mindspore tensor. The previous times, each time has the shape (1,)
            t: A mindspore tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A mindspore tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError(f"Solver order must be 1 or 2 or 3, got {order}")

    def dpm_solver_adaptive(
        self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type="dpmsolver"
    ):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A mindspore tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078, followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A mindspore tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas,
        "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * mint.ones((1,)).to(x.dtype)
        lambda_s = ns.marginal_lambda(s)
        lambda_0 = ns.marginal_lambda(t_0 * mint.ones_like(s).to(x.dtype))
        h = h_init * mint.ones_like(s).to(x.dtype)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(
                x, s, t, r1=r1, solver_type=solver_type, **kwargs
            )
        elif order == 3:
            r1, r2 = 1.0 / 3.0, 2.0 / 3.0
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(
                x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type
            )
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(
                x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs
            )
        else:
            raise ValueError(f"For adaptive step size solver, order must be 2 or 3, got {order}")
        while mint.abs(s - t_0).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = mint.max(mint.ones_like(x).to(x.dtype) * atol, rtol * mint.max(mint.abs(x_lower), mint.abs(x_prev)))
            norm_fn = lambda v: mint.sqrt(mint.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if mint.all(E <= 1.0):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.marginal_lambda(s)
            h = mint.min(theta * h * mint.float_power(E, -1.0 / order).float(), lambda_0 - lambda_s)
            nfe += order
        print("adaptive solver nfe", nfe)
        return x

    def add_noise(self, x, t, noise=None):
        """
        Compute the noised input xt = alpha_t * x + sigma_t * noise.

        Args:
            x: A `ms.Tensor` with shape `(batch_size, *shape)`.
            t: A `ms.Tensor` with shape `(t_size,)`.
        Returns:
            xt with shape `(t_size, batch_size, *shape)`.
        """
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        if noise is None:
            noise = mint.randn((t.shape[0], *x.shape))
        x = x.reshape((-1, *x.shape))
        xt = expand_dims(alpha_t, x.dim()) * x + expand_dims(sigma_t, x.dim()) * noise
        if t.shape[0] == 1:
            return xt.squeeze(0)
        else:
            return xt

    def inverse(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
    ):
        """
        Inverse the sample `x` from time `t_start` to `t_end` by DPM-Solver.
        For discrete-time DPMs, we use `t_start=1/N`, where `N` is the total time steps during training.
        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_start is None else t_start
        t_T = self.noise_schedule.T if t_end is None else t_end
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        return self.sample(
            x,
            steps=steps,
            t_start=t_0,
            t_end=t_T,
            order=order,
            skip_type=skip_type,
            method=method,
            lower_order_final=lower_order_final,
            denoise_to_zero=denoise_to_zero,
            solver_type=solver_type,
            atol=atol,
            rtol=rtol,
            return_intermediate=return_intermediate,
        )

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        return_intermediate=False,
        flow_shift=1.0,
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver.
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.

        =====================================================
        Args:
            x: A mindspore tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            return_intermediate: A `bool`. Whether to save the xt at each step.
                When set to `True`, method returns a tuple (x0, intermediates); when set to False, method returns only x0.
        Returns:
            x_end: A mindspore tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        if return_intermediate:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when saving intermediate values"
        if self.correcting_xt_fn is not None:
            assert method in [
                "multistep",
                "singlestep",
                "singlestep_fixed",
            ], "Cannot use adaptive solver when correcting_xt_fn is not None"
        intermediates = []

        # with torch.no_grad(): TODO
        if method == "adaptive":
            x = self.dpm_solver_adaptive(
                x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type
            )
        elif method == "multistep":
            assert steps >= order
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, shift=flow_shift)
            assert timesteps.shape[0] - 1 == steps
            # Init the initial values.
            step = 0
            t = timesteps[step]
            t_prev_list = [t]
            model_prev_list = [self.model_fn(x, t)]
            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step)
            if return_intermediate:
                intermediates.append(x)
            self.update_progress(step + 1, len(timesteps))
            # Init the first `order` values by lower order multistep DPM-Solver.
            for step in range(1, order):
                t = timesteps[step]
                x = self.multistep_dpm_solver_update(x, model_prev_list, t_prev_list, t, step, solver_type=solver_type)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                t_prev_list.append(t)
                model_prev_list.append(self.model_fn(x, t))
                # update progress bar
                self.update_progress(step + 1, len(timesteps))
            # Compute the remaining values by `order`-th order multistep DPM-Solver.
            for step in tqdm(range(order, steps + 1), disable=os.getenv("DPM_TQDM", "False") == "True"):
                t = timesteps[step]
                # We only use lower order for steps < 10
                # if lower_order_final and steps < 10:
                if lower_order_final:  # recommended by Shuchen Xue
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                x = self.multistep_dpm_solver_update(
                    x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type
                )
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = t
                # We do not need to evaluate the final model value.
                if step < steps:
                    model_prev_list[-1] = self.model_fn(x, t)
                # update progress bar
                self.update_progress(step + 1, len(timesteps))
        elif method in ["singlestep", "singlestep_fixed"]:
            if method == "singlestep":
                timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(
                    steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0
                )
            elif method == "singlestep_fixed":
                K = steps // order
                orders = [order] * K
                timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K)
            for step, order in enumerate(orders):
                s, t = timesteps_outer[step], timesteps_outer[step + 1]
                timesteps_inner = self.get_time_steps(skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order)
                lambda_inner = self.noise_schedule.marginal_lambda(timesteps_inner)
                h = lambda_inner[-1] - lambda_inner[0]
                r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
                if self.correcting_xt_fn is not None:
                    x = self.correcting_xt_fn(x, t, step)
                if return_intermediate:
                    intermediates.append(x)
                self.update_progress(step + 1, len(timesteps_outer))
        else:
            raise ValueError(f"Got wrong method {method}")
        if denoise_to_zero:
            t = mint.ones((1,)) * t_0
            x = self.denoise_to_zero_fn(x, t)
            if self.correcting_xt_fn is not None:
                x = self.correcting_xt_fn(x, t, step + 1)
            if return_intermediate:
                intermediates.append(x)
        if return_intermediate:
            return x, intermediates
        else:
            return x


#############################################################
# other utility functions
#############################################################


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to define the linear function.)

    Args:
        x: MindSpore tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for DPM-Solver).
        xp: MindSpore tensor with shape [C, K], where K is the number of keypoints.
        yp: MindSpore tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    N, K = x.shape[0], xp.shape[1]
    all_x = mint.cat([x.unsqueeze(2), xp.unsqueeze(0).repeat((N, 1, 1))], dim=2)
    sorted_all_x, x_indices = mint.sort(all_x, dim=2)
    x_idx = mint.argmin(x_indices, dim=2)
    cand_start_idx = x_idx - 1
    start_idx = mint.where(
        mint.eq(x_idx, 0), ms.tensor(1), mint.where(mint.eq(x_idx, K), ms.tensor(K - 2), cand_start_idx)
    )
    end_idx = mint.where(mint.eq(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = mint.gather(sorted_all_x, dim=2, index=start_idx.unsqueeze(2)).squeeze(2)
    end_x = mint.gather(sorted_all_x, dim=2, index=end_idx.unsqueeze(2)).squeeze(2)
    start_idx2 = mint.where(
        mint.eq(x_idx, 0), ms.tensor(0), mint.where(mint.eq(x_idx, K), ms.tensor(K - 2), cand_start_idx)
    )
    y_positions_expanded = yp.unsqueeze(0).expand(N, -1, -1)
    start_y = mint.gather(y_positions_expanded, dim=2, index=start_idx2.unsqueeze(2)).squeeze(2)
    end_y = mint.gather(y_positions_expanded, dim=2, index=(start_idx2 + 1).unsqueeze(2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.

    Args:
        `v`: a MindSpore tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a MindSpore tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
