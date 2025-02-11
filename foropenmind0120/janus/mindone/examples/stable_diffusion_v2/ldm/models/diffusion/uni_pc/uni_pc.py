# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import math

from ldm.util import is_old_ms_version

import mindspore as ms
from mindspore import ops, scipy


class NoiseScheduleVP:
    def __init__(
        self,
        schedule="discrete",
        betas=None,
        alphas_cumprod=None,
        continuous_beta_0=0.1,
        continuous_beta_1=20.0,
    ):
        """
        Create a wrapper class for the forward SDE (VP type).
        ***
        Update: We support discrete-time diffusion models by implementing a picewise linear interpolation for
                log_alpha_t.
                We recommend to use schedule='discrete' for the discrete-time diffusion models, especially for
                high-resolution images.
        ***
        The forward SDE ensures that the condition distribution q_{t|0}(x_t | x_0) = N ( alpha_t * x_0, sigma_t^2 * I ).
        We further define lambda_t = log(alpha_t) - log(sigma_t), which is the half-logSNR (described in the DPM-Solver
            paper).
        Therefore, we implement the functions for computing alpha_t, sigma_t and lambda_t. For t in [0, T], we have:
            log_alpha_t = self.marginal_log_mean_coeff(t)
            sigma_t = self.marginal_std(t)
            lambda_t = self.marginal_lambda(t)
        Moreover, as lambda(t) is an invertible function, we also support its inverse function:
            t = self.inverse_lambda(lambda_t)
        ===============================================================
        We support both discrete-time DPMs (trained on n = 0, 1, ..., N-1) and continuous-time DPMs (trained on t in
            [t_0, T]).
        1. For discrete-time DPMs:
            For discrete-time DPMs trained on n = 0, 1, ..., N-1, we convert the discrete steps to continuous time steps
                by: t_i = (i + 1) / N
            e.g. for N = 1000, we have t_0 = 1e-3 and T = t_{N-1} = 1.
            We solve the corresponding diffusion ODE from time T = 1 to time t_0 = 1e-3.
            Args:
                betas: A `mindspore.Tensor`. The beta array for the discrete-time DPM.
                    (See the original DDPM paper for details)
                alphas_cumprod: A `mindspore.Tensor`. The cumprod alphas for the discrete-time DPM.
                    (See the original DDPM paper for details)

            Note that we always have alphas_cumprod = cumprod(betas). Therefore, we only need to set one of `betas` and
            `alphas_cumprod`.
            **Important**:  Please pay special attention for the args for `alphas_cumprod`:
                The `alphas_cumprod` is the \hat{alpha_n} arrays in the notations of DDPM. Specifically, DDPMs assume
                    that q_{t_n | 0}(x_{t_n} | x_0) = N ( \sqrt{\hat{alpha_n}} * x_0, (1 - \hat{alpha_n}) * I ).
                Therefore, the notation \hat{alpha_n} is different from the notation alpha_t in DPM-Solver. In fact,
                    we have alpha_{t_n} = \sqrt{\hat{alpha_n}},
                and
                    log(alpha_{t_n}) = 0.5 * log(\hat{alpha_n}).
        2. For continuous-time DPMs:
            We support two types of VPSDEs: linear (DDPM) and cosine (improved-DDPM). The hyperparameters for the noise
            schedule are the default settings in DDPM and improved-DDPM:
            Args:
                beta_min: A `float` number. The smallest beta for the linear schedule.
                beta_max: A `float` number. The largest beta for the linear schedule.
                cosine_s: A `float` number. The hyperparameter in the cosine schedule.
                cosine_beta_max: A `float` number. The hyperparameter in the cosine schedule.
                T: A `float` number. The ending time of the forward process.
        ===============================================================
        Args:
            schedule: A `str`. The noise schedule of the forward SDE. 'discrete' for discrete-time DPMs,
                    'linear' or 'cosine' for continuous-time DPMs.
        Returns:
            A wrapper object of the forward SDE (VP type).
        ===============================================================
        Example:
        # For discrete-time DPMs, given betas (the beta array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', betas=betas)
        # For discrete-time DPMs, given alphas_cumprod (the \hat{alpha_n} array for n = 0, 1, ..., N - 1):
        >>> ns = NoiseScheduleVP('discrete', alphas_cumprod=alphas_cumprod)
        # For continuous-time DPMs (VPSDE), linear schedule:
        >>> ns = NoiseScheduleVP('linear', continuous_beta_0=0.1, continuous_beta_1=20.)
        """

        self.log = ops.Log()
        self.cast = ops.Cast()
        self.cos = ops.Cos()
        self.sqrt = ops.Sqrt()

        if schedule not in ["discrete", "linear", "cosine"]:
            raise ValueError(
                "Unsupported noise schedule {}. The schedule needs to be 'discrete' or 'linear' or 'cosine'".format(
                    schedule
                )
            )

        self.schedule = schedule
        if schedule == "discrete":
            if betas is not None:
                log_alphas = 0.5 * self.log(1 - betas).cumsum(dim=0)
            else:
                assert alphas_cumprod is not None
                log_alphas = 0.5 * self.log(alphas_cumprod)
            self.total_N = len(log_alphas)
            self.T = 1.0
            self.t_array = self.cast(
                ops.linspace(ms.Tensor(0.0, ms.float32), ms.Tensor(1.0, ms.float32), self.total_N + 1)[1:].reshape(
                    (1, -1)
                ),
                ms.float16,
            )
            self.log_alpha_array = log_alphas.reshape(
                (
                    1,
                    -1,
                )
            )
        else:
            self.total_N = 1000
            self.beta_0 = continuous_beta_0
            self.beta_1 = continuous_beta_1
            self.cosine_s = 0.008
            self.cosine_beta_max = 999.0
            self.cosine_t_max = (
                math.atan(self.cosine_beta_max * (1.0 + self.cosine_s) / math.pi)
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            self.cosine_log_alpha_0 = math.log(math.cos(self.cosine_s / (1.0 + self.cosine_s) * math.pi / 2.0))
            self.schedule = schedule
            if schedule == "cosine":
                # For the cosine schedule, T = 1 will have numerical issues. So we manually set the ending time T.
                # Note that T = 0.9946 may be not the optimal setting. However, we find it works well.
                self.T = 0.9946
            else:
                self.T = 1.0

    def marginal_log_mean_coeff(self, t):
        """
        Compute log(alpha_t) of a given continuous-time label t in [0, T].
        """
        if self.schedule == "discrete":
            return interpolate_fn(t.reshape((-1, 1)), self.t_array, self.log_alpha_array).reshape((-1))
        elif self.schedule == "linear":
            return -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        elif self.schedule == "cosine":

            def log_alpha_fn(s):
                return self.log(self.cos((s + self.cosine_s) / (1.0 + self.cosine_s) * math.pi / 2.0))

            log_alpha_t = log_alpha_fn(t) - self.cosine_log_alpha_0
            return log_alpha_t

    def marginal_alpha(self, t):
        """
        Compute alpha_t of a given continuous-time label t in [0, T].
        """
        return ops.exp(self.marginal_log_mean_coeff(t))

    def marginal_std(self, t):
        """
        Compute sigma_t of a given continuous-time label t in [0, T].
        """
        return self.sqrt(1.0 - ops.exp(2.0 * self.marginal_log_mean_coeff(t)))

    def marginal_lambda(self, t):
        """
        Compute lambda_t = log(alpha_t) - log(sigma_t) of a given continuous-time label t in [0, T].
        """
        log_mean_coeff = self.marginal_log_mean_coeff(t)
        log_std = 0.5 * ops.log(1.0 - ops.exp(2.0 * log_mean_coeff))
        return log_mean_coeff - log_std

    def inverse_lambda(self, lamb):
        """
        Compute the continuous-time label t in [0, T] of a given half-logSNR lambda_t.
        """
        if self.schedule == "linear":
            tmp = 2.0 * (self.beta_1 - self.beta_0) * ops.log(ops.exp(-2.0 * lamb) + ops.exp(ops.Zeros()((1,))))
            Delta = self.beta_0**2 + tmp
            return tmp / (ops.Sqrt()(Delta) + self.beta_0) / (self.beta_1 - self.beta_0)
        elif self.schedule == "discrete":
            log_alpha = -0.5 * ops.log(ops.exp(ops.Zeros()((1,))) + ops.exp(-2.0 * lamb))
            t = interpolate_fn(
                log_alpha.reshape((-1, 1)),
                ops.ReverseV2(axis=[1])(self.log_alpha_array),
                ops.ReverseV2(axis=[1])(self.t_array),
            )
            return t.reshape((-1,))
        else:
            log_alpha = -0.5 * ops.log(ops.exp(-2.0 * lamb) + ops.exp(ops.Zeros()((1,))))
            t_fn = (
                lambda log_alpha_t: ops.ACos()(ops.exp(log_alpha_t + self.cosine_log_alpha_0))
                * 2.0
                * (1.0 + self.cosine_s)
                / math.pi
                - self.cosine_s
            )
            t = t_fn(log_alpha)
            return t


def model_wrapper(
    model,
    noise_schedule,
    model_type="noise",
    model_kwargs={},
    guidance_type="uncond",
    condition=None,
    unconditional_condition=None,
    guidance_scale=1.0,
    classifier_fn=None,
    classifier_kwargs={},
):
    """
    Create a wrapper function for the noise prediction model.
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
        condition: A MindSpore tensor. The condition for the guided sampling.
                    Only used for "classifier" or "classifier-free" guidance type.
        unconditional_condition: A MindSpore tensor. The condition for the unconditional sampling.
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
            return (t_continuous - 1.0 / noise_schedule.total_N) * 1000.0
        else:
            return t_continuous

    def noise_pred_fn(x, t_continuous, cond=None):
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = ops.broadcast_to(t_continuous, (x.shape[0],))
        t_input = get_model_input_time(t_continuous)
        if cond is None:
            output = model(x, t_input, **model_kwargs)
        else:
            output = model(x, t_input, cond, **model_kwargs)
        if model_type == "noise":
            return output
        elif model_type == "x_start":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.ndim
            return (x - expand_dims(alpha_t, dims) * output) / expand_dims(sigma_t, dims)
        elif model_type == "v":
            alpha_t, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            dims = x.ndim
            return expand_dims(alpha_t, dims) * output + expand_dims(sigma_t, dims) * x
        elif model_type == "score":
            sigma_t = noise_schedule.marginal_std(t_continuous)
            dims = x.ndim
            return -expand_dims(sigma_t, dims) * output

    def sum_results(x_in, t_input, condition, **classifier_kwargs):
        log_prob = classifier_fn(x_in, t_input, condition, **classifier_kwargs)
        return log_prob.sum()

    def cond_grad_fn(x, t_input):
        """
        Compute the gradient of the classifier, i.e. nabla_{x} log p_t(cond | x_t).
        """
        x_in = x
        grad_fn = ops.value_and_grad(sum_results, 0, weights=None)
        return grad_fn(x_in, t_input, condition, **classifier_kwargs)[1]

    def model_fn(x, t_continuous):
        """
        The noise predicition model function that is used for DPM-Solver.
        """
        if t_continuous.reshape((-1,)).shape[0] == 1:
            t_continuous = ops.broadcast_to(t_continuous, (x.shape[0],))
        if guidance_type == "uncond":
            return noise_pred_fn(x, t_continuous)
        elif guidance_type == "classifier":
            assert classifier_fn is not None
            t_input = get_model_input_time(t_continuous)
            cond_grad = cond_grad_fn(x, t_input)
            sigma_t = noise_schedule.marginal_std(t_continuous)
            noise = noise_pred_fn(x, t_continuous)
            return noise - guidance_scale * expand_dims(sigma_t, dims=cond_grad.ndim) * cond_grad
        elif guidance_type == "classifier-free":
            if guidance_scale == 1.0 or unconditional_condition is None:
                return noise_pred_fn(x, t_continuous, cond=condition)
            else:
                x_in = ops.concat([x] * 2)
                t_in = ops.concat([t_continuous] * 2)
                c_in = ops.concat([unconditional_condition, condition])
                noise_output = noise_pred_fn(x_in, t_in, cond=c_in)
                if is_old_ms_version():
                    noise_uncond, noise = ops.split(noise_output, output_num=2)
                else:
                    noise_uncond, noise = ops.split(noise_output, split_size_or_sections=noise_output.shape[0] // 2)

                return noise_uncond + guidance_scale * (noise - noise_uncond)

    assert model_type in ["noise", "x_start", "v"]
    assert guidance_type in ["uncond", "classifier", "classifier-free"]
    return model_fn


class UniPC:
    def __init__(
        self,
        model_fn,
        noise_schedule,
        predict_x0=True,
        thresholding=False,
        thresholding_max_val=1.0,
        dynamic_thresholding_ratio=0.995,
        variant="bh1",
    ):
        """
        Construct a UniPC.
        We support both data_prediction and noise_prediction.
        """
        self.model = model_fn
        self.noise_schedule = noise_schedule
        self.variant = variant
        self.predict_x0 = predict_x0
        self.thresholding = thresholding
        self.thresholding_max_val = thresholding_max_val
        self.dynamic_thresholding_ratio = dynamic_thresholding_ratio

    def dynamic_thresholding_fn(self, x0, t):
        """
        The dynamic thresholding method.
        """
        dims = x0.ndim
        p = self.dynamic_thresholding_ratio

        temp = ops.Sort(axis=1)(ops.abs(x0).reshape((x0.shape[0], -1)))
        left_index = int((temp.shape[1] - 1) * p)
        right_index = left_index + 1
        left_column = temp[:, left_index]
        right_column = temp[:, right_index]
        s = left_column + (right_column - left_column) * p
        s = expand_dims(ops.maximum(s, self.thresholding_max_val * ops.ones_like(s)), dims)
        x0 = ops.clip_by_value(x0, -s, s) / s

        return x0

    def noise_prediction_fn(self, x, t):
        """
        Return the noise prediction model.
        """
        return self.model(x, t)

    def data_prediction_fn(self, x, t):
        """
        Return the data prediction model (with thresholding).
        """
        noise = self.noise_prediction_fn(x, t)
        dims = x.dim()
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - expand_dims(sigma_t, dims) * noise) / expand_dims(alpha_t, dims)
        if self.thresholding:
            x0 = self.dynamic_thresholding_fn(self, x0, t=None)
        return x0

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        if self.predict_x0:
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N):
        """
        Compute the intermediate time steps for sampling.
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.marginal_lambda(ms.Tensor(t_T, ms.float16))
            lambda_0 = self.noise_schedule.marginal_lambda(ms.Tensor(t_0, ms.float16))
            logSNR_steps = ops.linspace(lambda_T, lambda_0, N + 1)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return ops.Cast()(ops.linspace(ms.Tensor(t_T, ms.float32), ms.Tensor(t_0, ms.float32), N + 1), ms.float16)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = ops.pow(
                ops.linspace(ms.Tensor(t_T ** (1.0 / t_order)), ms.Tensor(t_0 ** (1.0 / t_order)), N + 1), t_order
            )
            return t
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic'".format(skip_type)
            )

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [
                    3,
                ] * (
                    K - 2
                ) + [2, 1]
            elif steps % 3 == 1:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [1]
            else:
                orders = [
                    3,
                ] * (
                    K - 1
                ) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [
                    2,
                ] * K
            else:
                K = steps // 2 + 1
                orders = [
                    2,
                ] * (
                    K - 1
                ) + [1]
        elif order == 1:
            K = steps
            orders = [
                1,
            ] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == "logSNR":
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps)[
                ops.CumSum()(
                    ms.tensor(
                        [
                            0,
                        ]
                        + orders
                    ),
                    0,
                )
            ]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order
        discretization.
        """
        return self.data_prediction_fn(x, s)

    def multistep_uni_pc_update(self, x, model_prev_list, t_prev_list, t, order, **kwargs):
        if len(t.shape) == 0:
            t = t.view(-1)
        if "bh" in self.variant:
            return self.multistep_uni_pc_bh_update(x, model_prev_list, t_prev_list, t, order, **kwargs)
        else:
            assert self.variant == "vary_coeff"
            return self.multistep_uni_pc_vary_update(x, model_prev_list, t_prev_list, t, order, **kwargs)

    def multistep_uni_pc_vary_update(self, x, model_prev_list, t_prev_list, t, order, use_corrector=True):
        # print(f'using unified predictor-corrector with order {order} (solver type: vary coeff)')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        alpha_t = ops.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = (lambda_prev_i - lambda_prev_0) / h
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.0)
        rks = ms.Tensor(rks)

        K = len(rks)
        # build C matrix
        C = []

        col = ops.ones_like(rks)
        for k in range(1, K + 1):
            C.append(col)
            col = col * rks / (k + 1)
        C = ops.stack(C, axis=1)

        if len(D1s) > 0:
            D1s = ops.stack(D1s, axis=1)  # (B, K)
            C_inv_p = scipy.linalg.inv(C[:-1, :-1])
            A_p = C_inv_p

        if use_corrector:
            # print('using corrector')
            C_inv = scipy.linalg.inv(C)
            A_c = C_inv

        hh = -h if self.predict_x0 else h
        h_phi_1 = ops.expm1(hh)
        h_phi_ks = []
        factorial_k = 1
        h_phi_k = h_phi_1
        for k in range(1, K + 2):
            h_phi_ks.append(h_phi_k)
            h_phi_k = h_phi_k / hh - 1 / factorial_k
            factorial_k *= k + 1

        model_t = None
        if self.predict_x0:
            x_t_ = sigma_t / sigma_prev_0 * x - alpha_t * h_phi_1 * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    temp = ops.matmul(D1s.transpose(0, 2, 3, 4, 1), A_p[k])
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * temp
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    temp = ops.matmul(D1s.transpose(0, 2, 3, 4, 1), A_c[k][:-1])
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * temp
                x_t = x_t - alpha_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        else:
            log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
            x_t_ = (ops.exp(log_alpha_t - log_alpha_prev_0)) * x - (sigma_t * h_phi_1) * model_prev_0
            # now predictor
            x_t = x_t_
            if len(D1s) > 0:
                # compute the residuals for predictor
                for k in range(K - 1):
                    temp = ops.matmul(D1s.transpose(0, 2, 3, 4, 1), A_c[k][:-1])
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * temp
            # now corrector
            if use_corrector:
                model_t = self.model_fn(x_t, t)
                D1_t = model_t - model_prev_0
                x_t = x_t_
                k = 0
                for k in range(K - 1):
                    temp = ops.matmul(D1s.transpose(0, 2, 3, 4, 1), A_c[k][:-1])
                    x_t = x_t - alpha_t * h_phi_ks[k + 1] * temp
                x_t = x_t - sigma_t * h_phi_ks[K] * (D1_t * A_c[k][-1])
        return x_t, model_t

    def multistep_uni_pc_bh_update(self, x, model_prev_list, t_prev_list, t, order, x_t=None, use_corrector=True):
        # print(f'using unified predictor-corrector with order {order} (solver type: B(h))')
        ns = self.noise_schedule
        assert order <= len(model_prev_list)
        dims = x.ndim

        expandd = ops.ExpandDims()
        # cast = ops.Cast()

        # first compute rks
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = ns.marginal_lambda(t_prev_0)
        lambda_t = ns.marginal_lambda(t)
        model_prev_0 = model_prev_list[-1]
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        log_alpha_prev_0, log_alpha_t = ns.marginal_log_mean_coeff(t_prev_0), ns.marginal_log_mean_coeff(t)
        alpha_t = ops.exp(log_alpha_t)

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = ns.marginal_lambda(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        if len(rks) == 0:
            rks = ms.Tensor(1.0)
        else:
            rks.append(ms.Tensor(1.0, ms.float16))
            rks_temp = []
            for loop_rks_temp in range(len(rks)):
                rks_temp.append(expandd(rks[loop_rks_temp], 0))
            rks = ops.concat(rks_temp)

        R = []
        b = []

        hh = -h[0] if self.predict_x0 else h[0]
        h_phi_1 = ops.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == "bh1":
            B_h = hh
        elif self.variant == "bh2":
            B_h = ops.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(ops.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = ops.stack(R)
        # b = ms.Tensor(b)
        if len(b) == 1:
            b = b[0]
        else:
            b_temp = []
            for loop_b_temp in range(len(b)):
                b_temp.append(expandd(b[loop_b_temp], 0))
            b = ops.concat(b_temp)

        # now predictor
        use_predictor = len(D1s) > 0 and x_t is None
        if len(D1s) > 0:
            D1s = ops.stack(D1s, axis=1)  # (B, K)
            if x_t is None:
                # for order 2, we use a simplified version
                if order == 2:
                    rhos_p = ms.Tensor([0.5])
                else:
                    R_m1_inv = inv(R[:-1, :-1])
                    rhos_p = ops.matmul(R_m1_inv, b[:-1])
        else:
            D1s = None

        if use_corrector:
            # print('using corrector')
            # for order 1, we use a simplified version
            if order == 1:
                rhos_c = ms.Tensor([0.5])
            else:
                R_inv = inv(R)
                rhos_c = ops.matmul(R_inv, b)

        model_t = None
        if self.predict_x0:
            x_t_ = expand_dims(sigma_t / sigma_prev_0, dims) * x - expand_dims(alpha_t * h_phi_1, dims) * model_prev_0

            if x_t is None:
                if use_predictor:
                    b, k, c, h, w = D1s.shape
                    D1s_transposed = D1s.transpose(1, 0, 2, 3, 4).reshape(k, -1)
                    pred_res = ops.matmul(rhos_p, D1s_transposed).reshape(b, c, h, w)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    b, k, c, h, w = D1s.shape
                    D1s_transposed = D1s.transpose(1, 0, 2, 3, 4).reshape(k, -1)
                    corr_res = ops.matmul(rhos_c[:-1], D1s_transposed).reshape(b, c, h, w)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - expand_dims(alpha_t * B_h, dims) * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = (
                expand_dims(ops.exp(log_alpha_t - log_alpha_prev_0), dims) * x
                - expand_dims(sigma_t * h_phi_1, dims) * model_prev_0
            )
            if x_t is None:
                if use_predictor:
                    b, k, c, h, w = D1s.shape
                    D1s_transposed = D1s.transpose(1, 0, 2, 3, 4).reshape(k, -1)
                    pred_res = ops.matmul(rhos_p, D1s_transposed).reshape(b, c, h, w)
                else:
                    pred_res = 0
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * pred_res

            if use_corrector:
                model_t = self.model_fn(x_t, t)
                if D1s is not None:
                    b, k, c, h, w = D1s.shape
                    D1s_transposed = D1s.transpose(1, 0, 2, 3, 4).reshape(k, -1)
                    corr_res = ops.matmul(rhos_c[:-1], D1s_transposed).reshape(b, c, h, w)
                else:
                    corr_res = 0
                D1_t = model_t - model_prev_0
                x_t = x_t_ - expand_dims(sigma_t * B_h, dims) * (corr_res + rhos_c[-1] * D1_t)
        return x_t, model_t

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=3,
        skip_type="time_uniform",
        method="singlestep",
        lower_order_final=True,
        denoise_to_zero=False,
    ):
        if steps > 30:
            print("The selected sampling timesteps are not appropriate for UniPC sampler")
        print(f"Running UniPC sampling with {steps} timesteps")
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        if method == "multistep":
            assert steps >= order
            timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps)
            assert timesteps.shape[0] - 1 == steps
            vec_t = ops.broadcast_to(timesteps[0], (x.shape[0], 1)).squeeze(-1)
            model_prev_list = [self.model_fn(x, vec_t)]
            t_prev_list = [vec_t]
            # Init the first `order` values by lower order multistep DPM-Solver.
            for init_order in range(1, order):
                vec_t = ops.broadcast_to(timesteps[init_order], (x.shape[0], 1)).squeeze(-1)
                x, model_x = self.multistep_uni_pc_update(
                    x, model_prev_list, t_prev_list, vec_t, init_order, use_corrector=True
                )
                if model_x is None:
                    model_x = self.model_fn(x, vec_t)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            for step in range(order, steps + 1):
                vec_t = ops.broadcast_to(timesteps[step], (x.shape[0], 1)).squeeze(-1)
                if lower_order_final:
                    step_order = min(order, steps + 1 - step)
                else:
                    step_order = order
                # print('this step order:', step_order)
                if step == steps:
                    # print('do not run corrector at the last step')
                    use_corrector = False
                else:
                    use_corrector = True
                x, model_x = self.multistep_uni_pc_update(
                    x, model_prev_list, t_prev_list, vec_t, step_order, use_corrector=use_corrector
                )
                for i in range(order - 1):
                    t_prev_list[i] = t_prev_list[i + 1]
                    model_prev_list[i] = model_prev_list[i + 1]
                t_prev_list[-1] = vec_t
                # We do not need to evaluate the final model value.
                if step < steps:
                    if model_x is None:
                        model_x = self.model_fn(x, vec_t)
                    model_prev_list[-1] = model_x
        else:
            raise NotImplementedError()
        if denoise_to_zero:
            x = self.denoise_to_zero_fn(x, ms.numpy.ones((x.shape[0],)) * t_0)
        return x


#############################################################
# other utility functions
#############################################################


def interpolate_fn(x, xp, yp):
    """
    A piecewise linear function y = f(x), using xp and yp as keypoints.
    We implement f(x) in a differentiable way (i.e. applicable for autograd).
    The function f(x) is well-defined for all x-axis. (For x beyond the bounds of xp, we use the outmost points of xp to
    define the linear function.)
    ===============================================================
    Args:
        x: MindSpore tensor with shape [N, C], where N is the batch size, C is the number of channels (we use C = 1 for
          DPM-Solver).
        xp: MindSpore tensor with shape [C, K], where K is the number of keypoints.
        yp: MindSpore tensor with shape [C, K].
    Returns:
        The function values f(x), with shape [N, C].
    """
    expandd = ops.ExpandDims()
    equal = ops.Equal()
    gatherd = ops.GatherD()
    cast = ops.Cast()

    N, K = x.shape[0], xp.shape[1]
    all_x = ops.concat([expandd(x, 2), ms.numpy.tile(expandd(xp, 0), (N, 1, 1))], axis=2)
    sorted_all_x, x_indices = ops.Sort(axis=2)(all_x)
    x_idx = ops.Argmin(axis=2)(cast(x_indices, ms.float16))
    cand_start_idx = x_idx - 1

    start_idx = ms.numpy.where(
        equal(x_idx, 0),
        ms.Tensor(1),
        ms.numpy.where(
            equal(x_idx, K),
            ms.Tensor(K - 2),
            cand_start_idx,
        ),
    )
    end_idx = ms.numpy.where(equal(start_idx, cand_start_idx), start_idx + 2, start_idx + 1)
    start_x = gatherd(sorted_all_x, 2, expandd(start_idx, 2)).squeeze(2)
    end_x = gatherd(sorted_all_x, 2, expandd(end_idx, 2)).squeeze(2)
    start_idx2 = ms.numpy.where(
        equal(x_idx, 0),
        ms.Tensor(0),
        ms.numpy.where(
            equal(x_idx, K),
            ms.Tensor(K - 2),
            cand_start_idx,
        ),
    )
    y_positions_expanded = ops.broadcast_to(expandd(yp, 0), (N, -1, -1))
    start_y = gatherd(y_positions_expanded, 2, expandd(start_idx2, 2)).squeeze(2)
    end_y = gatherd(y_positions_expanded, 2, expandd((start_idx2 + 1), 2)).squeeze(2)
    cand = start_y + (x - start_x) * (end_y - start_y) / (end_x - start_x)
    return cand


def inv(x):
    """
    Compute the inverse matrix of tensor `x`.
    ===============================================================
    Args:
        `x`: a MindSpore tensor with shape [2, 2].
        `dim`: a `int`.
    Returns:
        a MindSpore tensor with shape [2, 2].
    """
    x_inv = ms.numpy.zeros_like(x)
    a, b, c, d = x[0, 0], x[0, 1], x[1, 0], x[1, 1]
    x_inv[0, 0] = d / (a * d - b * c)
    x_inv[0, 1] = -b / (a * d - b * c)
    x_inv[1, 0] = -c / (a * d - b * c)
    x_inv[1, 1] = a / (a * d - b * c)
    return x_inv


def expand_dims(v, dims):
    """
    Expand the tensor `v` to the dim `dims`.
    ===============================================================
    Args:
        `v`: a MindSpore tensor with shape [N].
        `dim`: a `int`.
    Returns:
        a MindSpore tensor with shape [N, 1, 1, ..., 1] and the total dimension is `dims`.
    """
    return v[(...,) + (None,) * (dims - 1)]
