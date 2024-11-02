import math

import numpy as np

import mindspore as ms
from mindspore import ops

from ..utils.tools import assert_shape
from .diffusion_utils import discretized_gaussian_log_likelihood, normal_kl


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(axis=list(range(1, len(tensor.shape))))


class ModelMeanType:
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = 1  # the model predicts x_{t-1}
    START_X = 2  # the model predicts x_0
    EPSILON = 3  # the model predicts epsilon
    VELOCITY = 4  # the model predicts v


predict_type_dict = {
    "epsilon": ModelMeanType.EPSILON,
    "sample": ModelMeanType.START_X,
    "v_prediction": ModelMeanType.VELOCITY,
}


class ModelVarType:
    """
    What is used as the model's output variance.
    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = 1
    FIXED_SMALL = 2
    FIXED_LARGE = 3
    LEARNED_RANGE = 4


class LossType:
    MSE = 1  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = 2  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = 3  # use the variational lower-bound
    RESCALED_KL = 4  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start**0.5,
                beta_end**0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "warmup10":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == "warmup50":
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert_shape(betas, (num_diffusion_timesteps,))
    return betas


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps, beta_start=0.0001, beta_end=0.02):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * beta_start,  # DDPM
            beta_end=scale * beta_end,  # DDPM
            num_diffusion_timesteps=num_diffusion_timesteps,  # DDPM
        )
    elif schedule_name == "scaled_linear":
        return get_beta_schedule(
            "quad",
            beta_start=beta_start,  # StableDiffusion, should be 0.00085
            beta_end=beta_end,  # StableDiffusion, should be 0.012
            num_diffusion_timesteps=num_diffusion_timesteps,  # StableDiffusion
        )
    elif schedule_name == "squaredcos_cap_v2":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
        self,
        *,
        model,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
        mse_loss_weight_type="constant",
        noise_offset=0.0,
    ):
        self.model = model
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        self.mse_loss_weight_type = mse_loss_weight_type
        self.noise_offset = noise_offset

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)

        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert_shape(self.alphas_cumprod_prev, (self.num_timesteps,))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = ms.tensor(np.sqrt(self.alphas_cumprod), dtype=ms.float32)
        self.sqrt_one_minus_alphas_cumprod = ms.tensor(np.sqrt(1.0 - self.alphas_cumprod), dtype=ms.float32)
        self.log_one_minus_alphas_cumprod = ms.tensor(np.log(1.0 - self.alphas_cumprod), dtype=ms.float32)
        self.sqrt_recip_alphas_cumprod = ms.tensor(np.sqrt(1.0 / self.alphas_cumprod), dtype=ms.float32)
        self.sqrt_recipm1_alphas_cumprod = ms.tensor(np.sqrt(1.0 / self.alphas_cumprod - 1), dtype=ms.float32)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = (
            ms.tensor(np.log(np.append(posterior_variance[1], posterior_variance[1:])), dtype=ms.float32)
            if len(posterior_variance) > 1
            else ms.tensor(np.array([]), dtype=ms.float32)
        )
        self.posterior_variance = ms.tensor(posterior_variance, dtype=ms.float32)

        self.posterior_mean_coef1 = ms.tensor(
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod), dtype=ms.float32
        )
        self.posterior_mean_coef2 = ms.tensor(
            (1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod), dtype=ms.float32
        )

        self.log_2 = float(np.log(2.0))
        self.log_betas = ms.tensor(np.log(self.betas), dtype=ms.float32)

        self.sampler = {
            "ddpm": self.p_sample_loop,
            "ddim": self.ddim_sample_loop,
            "plms": self.plms_sample_loop,
        }

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = ops.randn_like(x_start, dtype=x_start.dtype)
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        model_var_type=None,
        frozen_out=None,
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param model_var_type: if not None, overlap the default self.model_var_type.
            It is useful when training with learned var but sampling with fixed var.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        if model_var_type is None:
            model_var_type = self.model_var_type

        B, C = x.shape[:2]
        if frozen_out is None:
            out_dict = self.model(x, t, **model_kwargs)
            model_output = out_dict[0]
        else:
            model_output = frozen_out

        # self.model_var_type corresponds to model output
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            model_output, model_var_values = ops.split(model_output, C, axis=1)
        else:
            model_var_values = None

        # model_var_type corresponds to reverse diffusion process
        if model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            if model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = ops.exp(model_log_variance)
            else:
                min_log = _extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
                max_log = _extract_into_tensor(self.log_betas, t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = ops.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    ms.tensor(self.posterior_variance[1].item(), self.betas[1:].item()),
                    ops.log(ms.tensor(self.posterior_variance[1].item(), self.betas[1:].item())),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON, ModelMeanType.VELOCITY]:
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            elif self.model_mean_type == ModelMeanType.EPSILON:
                pred_xstart = process_xstart(self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            else:
                pred_xstart = process_xstart(self._predict_xstart_from_v(x_t=x, t=t, v=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        return (model_mean, model_variance, model_log_variance, pred_xstart)

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_v(self, x_t, t, v):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        return (  # (xprev - coef2*x_t) / coef1
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _velocity_from_xstart_and_noise(self, x_start, t, noise):
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * noise
            - _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def _vb_terms_bpd(self, x_start, x_t, t, clip_denoised=True, model_kwargs=None, frozen_out=None):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)
        out = self.p_mean_variance(
            x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, frozen_out=frozen_out
        )
        kl = normal_kl(true_mean, true_log_variance_clipped, out[0], out[2])
        kl = mean_flat(kl) / self.log_2

        decoder_nll = -discretized_gaussian_log_likelihood(x_start, means=out[0], log_scales=0.5 * out[2])
        decoder_nll = mean_flat(decoder_nll) / self.log_2

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = ops.where((t == 0), decoder_nll, kl)
        return (output, out[3])

    def training_losses(self, x_start, model_kwargs=None, controlnet=None, noise=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        # Time steps
        t = ops.randint(0, self.num_timesteps, (x_start.shape[0],))
        # Noise
        if noise is None:
            noise = ops.randn_like(x_start, dtype=x_start.dtype)
        if self.noise_offset > 0:
            # Add channel wise noise offset
            # https://www.crosslabs.org/blog/diffusion-with-offset-noise
            noise = noise + self.noise_offset * ops.randn((*x_start.shape[:2], 1, 1), dtype=x_start.dtype)
        x_t = self.q_sample(x_start, t, noise=noise)
        terms = {}

        if self.mse_loss_weight_type == "constant":
            mse_loss_weight = ops.ones_like(t)
        elif self.mse_loss_weight_type.startswith("min_snr_"):
            alpha = _extract_into_tensor(self.sqrt_alphas_cumprod, t, t.shape)
            sigma = _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
            snr = (alpha / sigma) ** 2

            k = float(self.mse_loss_weight_type.split("min_snr_")[-1])
            # min{snr, k}
            mse_loss_weight = ops.stack([snr, k * ops.ones_like(t)], axis=1).min(axis=1)[0] / snr
        else:
            raise ValueError(self.mse_loss_weight_type)

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            out_dict = self._vb_terms_bpd(
                x_start=x_start,
                x_t=x_t,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            terms["loss"] = out_dict[0]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            if controlnet is not None:
                controls = controlnet(x_t, t, **model_kwargs)
                model_kwargs.pop("condition")
                model_kwargs.update(controls)
            out_dict = self.model(x_t, t, **model_kwargs)
            model_output = out_dict[0]

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                model_output, model_var_values = ops.split(model_output, C, axis=1)
                # Learn the variance using the variational bound, but don't let
                # it affect our mean prediction.
                frozen_out = ops.cat([ops.stop_gradient(model_output), model_var_values], axis=1)
                terms["vb"] = self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    frozen_out=frozen_out,
                )[0]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            if self.model_mean_type == ModelMeanType.VELOCITY:
                target = self._velocity_from_xstart_and_noise(x_start, t, noise)
            else:
                if self.model_mean_type == ModelMeanType.START_X:
                    target = x_start
                elif self.model_mean_type == ModelMeanType.EPSILON:
                    target = noise
                else:
                    target = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0]
            raw_mse = mean_flat((target - model_output) ** 2)
            raw_mse = ops.stop_gradient(raw_mse)
            terms["mse"] = mse_loss_weight * mean_flat((target - model_output) ** 2)
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
                terms["raw_loss"] = raw_mse + ops.stop_gradient(terms["vb"])
            else:
                terms["loss"] = terms["mse"]
                terms["raw_loss"] = raw_mse
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, t, **model_kwargs)
        new_mean = p_mean_var[0].float() + p_mean_var[1] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.
        See condition_mean() for details on cond_fn.
        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self._predict_eps_from_xstart(x, t, p_mean_var[3])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, t, **model_kwargs)

        out = p_mean_var.copy()
        out[3] = self._predict_xstart_from_eps(x, t, eps)
        out[0], _, _ = self.q_posterior_mean_variance(x_start=out[3], x_t=x, t=t)
        return out

    def p_sample(
        self,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        model_var_type=None,
        **kwargs,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param model_var_type: if not None, overlap the default self.model_var_type.
            It is useful when training with learned var but sampling with fixed var.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            model_var_type=model_var_type,
        )
        noise = ops.randn_like(x, dtype=x.dtype)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        if cond_fn is not None:
            out[0] = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)

        sample = out[0] + nonzero_mask * ops.exp(0.5 * out[2]) * noise

        return (sample, out[3])

    def p_sample_loop(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        model_var_type=None,
        progress=False,
        progress_leave=True,
        **kwargs,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param model_var_type: if not None, overlap the default self.model_var_type.
            It is useful when training with learned var but sampling with fixed var.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            model_var_type=model_var_type,
            progress=progress,
            progress_leave=progress_leave,
            **kwargs,
        ):
            final = sample
        return final["sample"]

    def p_sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        model_var_type=None,
        progress=False,
        progress_leave=True,
        **kwargs,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        dtype = next(self.model.get_parameters()).dtype
        if noise is not None:
            img = noise
        else:
            img = ops.randn(*shape, dtype=dtype)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, leave=progress_leave)

        for i in indices:
            t = ms.tensor([i] * shape[0])
            out = ops.stop_gradient(
                self.p_sample(
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    model_var_type=model_var_type,
                    **kwargs,
                )
            )
            yield out
            img = out[0]

    def ddim_sample(
        self,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM.

        Same usage as p_sample().
        """
        out = self.p_mean_variance(
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)

        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self._predict_eps_from_xstart(x, t, out[3])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        sigma = eta * ops.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * ops.sqrt(1 - alpha_bar / alpha_bar_prev)
        # Equation 12.
        noise = ops.randn_like(x, dtype=x.dtype)
        mean_pred = out[3] * ops.sqrt(alpha_bar_prev) + ops.sqrt(1 - alpha_bar_prev - sigma**2) * eps
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))  # no noise when t == 0
        sample = mean_pred + nonzero_mask * sigma * noise
        return (sample, out[3])

    def ddim_reverse_sample(
        self,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        out = self.p_mean_variance(
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if cond_fn is not None:
            out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out[3]) / _extract_into_tensor(
            self.sqrt_recipm1_alphas_cumprod, t, x.shape
        )
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = out[3] * ops.sqrt(alpha_bar_next) + ops.sqrt(1 - alpha_bar_next) * eps

        return mean_pred, out[3]

    def ddim_sample_loop(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
        progress_leave=True,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.ddim_sample_loop_progressive(
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=progress,
            progress_leave=progress_leave,
            eta=eta,
        ):
            final = sample
        return final[0]

    def ddim_sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
        progress_leave=True,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        dtype = next(self.model.get_parameters()).dtype
        if noise is not None:
            img = noise
        else:
            img = ops.randn(*shape, dtype=dtype)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, leave=progress_leave)

        for i in indices:
            t = ms.tensor([i] * shape[0])
            out = ops.stop_gradient(
                self.ddim_sample(
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                )
            )
            yield out
            img = out[0]

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = ms.tensor([self.num_timesteps - 1] * batch_size)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=ms.tensor(0.0))
        return mean_flat(kl_prior) / self.log_2

    def calc_bpd_loop(self, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = ms.tensor([t] * batch_size)
            noise = ops.randn_like(x_start, dtype=x_start.dtype)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            out = ops.stop_gradient(
                self._vb_terms_bpd(
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            )
            vb.append(out[0])
            xstart_mse.append(mean_flat((out[1] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out[1])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = ops.stack(vb, axis=1)
        xstart_mse = ops.stack(xstart_mse, axis=1)
        mse = ops.stack(mse, axis=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(axis=1) + prior_bpd
        return (
            total_bpd,
            prior_bpd,
            vb,
            xstart_mse,
            mse,
        )

    def get_eps(
        self,
        x,
        t,
        model_kwargs,
        cond_fn=None,
    ):
        model_output = self.model(x, t, **model_kwargs)[0]
        if isinstance(model_output, tuple):
            model_output, _ = model_output
        eps = model_output[:, :4]
        if cond_fn is not None:
            alpha_bar = _extract_into_tensor_lerp(self.alphas_cumprod, t, x.shape)
            eps = eps - ops.sqrt(1 - alpha_bar) * cond_fn(x, t, **model_kwargs)
        return eps

    def eps_to_pred_xstart(
        self,
        x,
        eps,
        t,
    ):
        alpha_bar = _extract_into_tensor_lerp(self.alphas_cumprod, t, x.shape)
        return (x - eps * ops.sqrt(1 - alpha_bar)) / ops.sqrt(alpha_bar)

    def pndm_transfer(
        self,
        x,
        eps,
        t_1,
        t_2,
    ):
        pred_xstart = self.eps_to_pred_xstart(x, eps, t_1)
        alpha_bar_prev = _extract_into_tensor_lerp(self.alphas_cumprod, t_2, x.shape)
        return pred_xstart * ops.sqrt(alpha_bar_prev) + ops.sqrt(1 - alpha_bar_prev) * eps

    def prk_sample_loop(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
    ):
        """
        Generate samples from the model using PRK.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.prk_sample_loop_progressive(
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=progress,
        ):
            final = sample
        return final[0]

    def prk_sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
    ):
        """
        Use PRK to sample from the model and yield intermediate samples from
        each timestep of PRK.

        Same usage as p_sample_loop_progressive().
        """
        dtype = next(self.model.get_parameters()).dtype
        if noise is not None:
            img = noise
        else:
            img = ops.randn(*shape, dtype=dtype)
        indices = list(range(self.num_timesteps))[::-1][1:-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, leave=False)

        for i in indices:
            t = ms.tensor([i] * shape[0])
            out = ops.stop_gradient(
                self.prk_sample(
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            )
            yield out
            img = out[0]

    def prk_sample(
        self,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model using fourth-order Pseudo Runge-Kutta
        (https://openreview.net/forum?id=PlKWVd2yBkY).

        Same usage as p_sample().
        """
        if model_kwargs is None:
            model_kwargs = {}

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        eps_1 = self.get_eps(x, t, model_kwargs, cond_fn)
        x_1 = self.pndm_transfer(x, eps_1, t, t - 0.5)
        eps_2 = self.get_eps(x_1, t - 0.5, model_kwargs, cond_fn)
        x_2 = self.pndm_transfer(x, eps_2, t, t - 0.5)
        eps_3 = self.get_eps(x_2, t - 0.5, model_kwargs, cond_fn)
        x_3 = self.pndm_transfer(x, eps_3, t, t - 1)
        eps_4 = self.get_eps(x_3, t - 1, model_kwargs, cond_fn)
        eps_prime = (eps_1 + 2 * eps_2 + 2 * eps_3 + eps_4) / 6

        sample = self.pndm_transfer(x, eps_prime, t, t - 1)
        pred_xstart = self.eps_to_pred_xstart(x, eps_prime, t)
        pred_xstart = process_xstart(pred_xstart)
        return (sample, pred_xstart, eps_prime)

    def plms_sample_loop(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
        progress_leave=True,
    ):
        """
        Generate samples from the model using PLMS.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.plms_sample_loop_progressive(
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            progress=progress,
            progress_leave=progress_leave,
        ):
            final = sample
        return final[0]

    def plms_sample_loop_progressive(
        self,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
        progress=False,
        progress_leave=True,
    ):
        """
        Use PLMS to sample from the model and yield intermediate samples from
        each timestep of PLMS.

        Same usage as p_sample_loop_progressive().
        """
        dtype = next(self.model.get_parameters()).dtype
        if noise is not None:
            img = noise
        else:
            img = ops.randn(*shape, dtype=dtype)
        indices = list(range(self.num_timesteps))[::-1][1:-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices, leave=progress_leave)

        old_eps = []

        for i in indices:
            t = ms.tensor([i] * shape[0])
            if len(old_eps) < 3:
                out = ops.stop_gradient(
                    self.prk_sample(
                        img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                )
            else:
                out = ops.stop_gradient(
                    self.plms_sample(
                        img,
                        old_eps,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        cond_fn=cond_fn,
                        model_kwargs=model_kwargs,
                    )
                )
                old_eps.pop(0)
            old_eps.append(out[2])
            yield out
            img = out[0]

    def plms_sample(
        self,
        x,
        old_eps,
        t,
        clip_denoised=True,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model using fourth-order Pseudo Linear Multistep
        (https://openreview.net/forum?id=PlKWVd2yBkY).
        """
        if model_kwargs is None:
            model_kwargs = {}

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        eps = self.get_eps(x, t, model_kwargs, cond_fn)
        eps_prime = (55 * eps - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        sample = self.pndm_transfer(x, eps_prime, t, t - 1)
        pred_xstart = self.eps_to_pred_xstart(x, eps, t)
        pred_xstart = process_xstart(pred_xstart)
        return (sample, pred_xstart, eps)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = (
        ops.index_select(arr, 0, timesteps)
        .reshape(timesteps.shape + (1,) * (len(broadcast_shape) - len(timesteps.shape)))
        .broadcast_to(broadcast_shape)
    )
    return res


def _extract_into_tensor_lerp(arr, timesteps, broadcast_shape):
    """
    Extract values from arr with fractional time steps
    """
    timesteps = timesteps.float()
    frac = timesteps.frac()
    frac_reshape = frac.reshape(timesteps.shape + (1,) * (len(broadcast_shape) - len(timesteps.shape)))
    res_1 = _extract_into_tensor(arr, timesteps.floor().long(), broadcast_shape)
    res_2 = _extract_into_tensor(arr, timesteps.ceil().long(), broadcast_shape)
    return ops.lerp(res_1, res_2, frac_reshape)
