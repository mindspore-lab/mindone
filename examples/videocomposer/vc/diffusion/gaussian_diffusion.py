import math

from tqdm import tqdm

import mindspore as ms
from mindspore import ops

__all__ = [
    "beta_schedule",
    "GaussianDiffusion",
]


def beta_schedule(schedule, num_timesteps=1000, init_beta=None, last_beta=None):
    """This code defines a function beta_schedule that generates a sequence of beta values based on
    the given input parameters. These beta values can be used in video diffusion processes.

    Args:
        schedule(str): Determines the type of beta schedule to be generated.
            It can be 'linear', 'linear_sd', 'quadratic', or 'cosine'.
        num_timesteps(int, optional): The number of timesteps for the generated beta schedule. Default is 1000.
        init_beta(float, optional): The initial beta value.
            If not provided, a default value is used based on the chosen schedule.
        last_beta(float, optional): The final beta value.
            If not provided, a default value is used based on the chosen schedule.

    Returns:
        The function returns a PyTorch tensor containing the generated beta values.
        The beta schedule is determined by the schedule parameter:
        1.Linear: Generates a linear sequence of beta values between init_beta and last_beta.
        2.Linear_sd: Generates a linear sequence of beta values between the square root of init_beta and
            the square root of last_beta, and then squares the result.
        3.Quadratic: Similar to the 'linear_sd' schedule, but with different default values for init_beta and last_beta.
        4.Cosine: Generates a sequence of beta values based on a cosine function,
            ensuring the values are between 0 and 0.999.

    Raises:
        If an unsupported schedule is provided, a ValueError is raised with a message indicating the issue.
    """
    if schedule == "linear":
        scale = 1000.0 / num_timesteps
        init_beta = init_beta or scale * 0.0001
        last_beta = last_beta or scale * 0.02
        return ops.linspace(init_beta, last_beta, num_timesteps).to(ms.float64)
    elif schedule == "linear_sd":
        return ops.linspace(init_beta**0.5, last_beta**0.5, num_timesteps).to(ms.float64) ** 2
    elif schedule == "quadratic":
        init_beta = init_beta or 0.0015
        last_beta = last_beta or 0.0195
        return ops.linspace(init_beta**0.5, last_beta**0.5, num_timesteps).to(ms.float64) ** 2
    elif schedule == "cosine":
        betas = []
        for step in range(num_timesteps):
            t1 = step / num_timesteps
            t2 = (step + 1) / num_timesteps
            fn = lambda u: math.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2  # noqa
            betas.append(min(1.0 - fn(t2) / fn(t1), 0.999))
        return ms.Tensor(betas, dtype=ms.float64)
    else:
        raise ValueError(f"Unsupported schedule: {schedule}")


def _i(tensor, t, x):
    r"""Index tensor using t and format the output according to x."""
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    return tensor[t].view(shape).to(x.dtype)


class GaussianDiffusion(object):
    def __init__(
        self, betas, mean_type="eps", var_type="learned_range", loss_type="mse", epsilon=1e-12, rescale_timesteps=False
    ):
        # check input
        if not isinstance(betas, ms.Tensor):
            betas = ms.Tensor(betas, dtype=ms.float32)
        if not betas.dtype == ms.float32:
            betas = betas.to(ms.float32)
        assert betas.min() > 0 and betas.max() <= 1
        assert mean_type in ["x0", "x_{t-1}", "eps"]
        assert var_type in ["learned", "learned_range", "fixed_large", "fixed_small"]
        assert loss_type in ["mse", "rescaled_mse", "kl", "rescaled_kl", "l1", "rescaled_l1", "charbonnier"]
        self.betas = betas
        self.num_timesteps = len(betas)
        self.mean_type = mean_type  # eps
        self.var_type = var_type  # 'fixed_small'
        self.loss_type = loss_type  # mse
        self.epsilon = epsilon  # 1e-12
        self.rescale_timesteps = rescale_timesteps  # False

        # alphas
        alphas = 1 - self.betas
        self.alphas_cumprod = ops.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = ops.cat([alphas.new_ones([1]), self.alphas_cumprod[:-1]])
        self.alphas_cumprod_next = ops.cat([self.alphas_cumprod[1:], alphas.new_zeros([1])])

        # q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = ops.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = ops.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = ops.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = ops.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = ops.sqrt(1.0 / self.alphas_cumprod - 1)

        # q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = ops.log(self.posterior_variance.clamp(1e-20))
        self.posterior_mean_coef1 = betas * ops.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * ops.sqrt(alphas) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        r"""Sample from q(x_t | x_0)."""
        noise = ops.randn_like(x0) if noise is None else noise
        return _i(self.sqrt_alphas_cumprod, t, x0) * x0 + _i(self.sqrt_one_minus_alphas_cumprod, t, x0) * noise

    def q_mean_variance(self, x0, t):
        r"""Distribution of q(x_t | x_0)."""
        mu = _i(self.sqrt_alphas_cumprod, t, x0) * x0
        var = _i(1.0 - self.alphas_cumprod, t, x0)
        log_var = _i(self.log_one_minus_alphas_cumprod, t, x0)
        return mu, var, log_var

    def q_posterior_mean_variance(self, x0, xt, t):
        r"""Distribution of q(x_{t-1} | x_t, x_0)."""
        mu = _i(self.posterior_mean_coef1, t, xt) * x0 + _i(self.posterior_mean_coef2, t, xt) * xt
        var = _i(self.posterior_variance, t, xt)
        log_var = _i(self.posterior_log_variance_clipped, t, xt)
        return mu, var, log_var

    # @torch.no_grad()
    def p_sample(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None):
        r"""Sample from p(x_{t-1} | x_t).
        - condition_fn: for classifier-based guidance (guided-diffusion).
        - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        # predict distribution of p(x_{t-1} | x_t)
        mu, var, log_var, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # random sample (with optional conditional function)
        noise = ops.randn_like(xt)
        mask = t.ne(0).float().view(-1, *((1,) * (xt.ndim - 1)))  # no noise when t == 0
        if condition_fn is not None:
            grad = condition_fn(xt, self._scale_timesteps(t), **model_kwargs)
            mu = mu.float() + var * grad.float()
        xt_1 = mu + mask * ops.exp(0.5 * log_var) * noise
        return xt_1, x0

    # @torch.no_grad()
    def p_sample_loop(
        self, noise, model, model_kwargs={}, clamp=None, percentile=None, condition_fn=None, guide_scale=None
    ):
        r"""Sample from p(x_{t-1} | x_t) p(x_{t-2} | x_{t-1}) ... p(x_0 | x_1)."""
        # prepare input
        b = noise.shape[0]
        xt = noise

        # diffusion process
        for step in ops.arange(self.num_timesteps).flip((0,)):
            t = ops.full((b,), step, dtype=ms.int64)
            xt, _ = self.p_sample(xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale)
        return xt

    def p_mean_variance(self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None):
        r"""Distribution of p(x_{t-1} | x_t)."""
        # predict distribution
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0])
            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1])
            dim = y_out.shape[1] if self.var_type.startswith("fixed") else y_out.shape[1] // 2
            out = ops.cat(
                [u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]), y_out[:, dim:]], axis=1
            )  # guide_scale=9.0

        # compute variance
        if self.var_type == "learned":
            out, log_var = out.chunk(2, axis=1)
            var = ops.exp(log_var)
        elif self.var_type == "learned_range":
            out, fraction = out.chunk(2, axis=1)
            min_log_var = _i(self.posterior_log_variance_clipped, t, xt)
            max_log_var = _i(ops.log(self.betas), t, xt)
            fraction = (fraction + 1) / 2.0
            log_var = fraction * max_log_var + (1 - fraction) * min_log_var
            var = ops.exp(log_var)
        elif self.var_type == "fixed_large":
            var = _i(ops.cat([self.posterior_variance[1:2], self.betas[1:]]), t, xt)
            log_var = ops.log(var)
        elif self.var_type == "fixed_small":
            var = _i(self.posterior_variance, t, xt)
            log_var = _i(self.posterior_log_variance_clipped, t, xt)

        # compute mean and x0
        if self.mean_type == "x_{t-1}":
            mu = out  # x_{t-1}
            x0 = (
                _i(1.0 / self.posterior_mean_coef1, t, xt) * mu
                - _i(self.posterior_mean_coef2 / self.posterior_mean_coef1, t, xt) * xt
            )
        elif self.mean_type == "x0":
            x0 = out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)
        elif self.mean_type == "eps":
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * out
            mu, _, _ = self.q_posterior_mean_variance(x0, xt, t)

        # restrict the range of x0
        if percentile is not None:
            assert percentile > 0 and percentile <= 1  # e.g., 0.995
            s = ops.quantile(x0.flatten(1).abs(), percentile, axis=1).clamp(1.0).view(-1, 1, 1, 1)
            x0 = ops.min(s, ops.max(-s, x0)) / s
        elif clamp is not None:
            x0 = x0.clamp(-clamp, clamp)
        return mu, var, log_var, x0

    # @torch.no_grad()
    def ddim_sample(
        self,
        xt,
        t,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        condition_fn=None,
        guide_scale=None,
        ddim_timesteps=20,
        eta=0.0,
    ):
        r"""Sample from p(x_{t-1} | x_t) using DDIM.
        - condition_fn: for classifier-based guidance (guided-diffusion).
        - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)
        if condition_fn is not None:
            # x0 -> eps
            alpha = _i(self.alphas_cumprod, t, xt)
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas = _i(self.alphas_cumprod, t, xt)
        alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
        sigmas = eta * ops.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))

        # random sample
        noise = ops.randn_like(xt)
        direction = ops.sqrt(1 - alphas_prev - sigmas**2) * eps
        mask = t.ne(0).float().view(-1, *((1,) * (xt.ndim - 1)))
        xt_1 = ops.sqrt(alphas_prev) * x0 + direction + mask * sigmas * noise
        return xt_1, x0

    # @torch.no_grad()
    def ddim_sample_loop(
        self,
        noise,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        condition_fn=None,
        guide_scale=None,
        ddim_timesteps=20,
        eta=0.0,
    ):
        # prepare input
        b = noise.shape[0]
        xt = noise

        # diffusion process (TODO: clamp is inaccurate! Consider replacing the stride by explicit prev/next steps)
        steps = (
            (1 + ops.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps))
            .clamp(0, self.num_timesteps - 1)
            .flip((0,))
        )
        steps = tqdm(steps, desc="ddim_sample_loop")
        for step in steps:
            t = ops.full((b,), step, dtype=ms.int64)
            xt, _ = self.ddim_sample(
                xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, ddim_timesteps, eta
            )
        return xt

    # @torch.no_grad()
    def ddim_reverse_sample(
        self, xt, t, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20
    ):
        r"""Sample from p(x_{t+1} | x_t) using DDIM reverse ODE (deterministic)."""
        stride = self.num_timesteps // ddim_timesteps

        # predict distribution of p(x_{t-1} | x_t)
        _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

        # derive variables
        eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
        alphas_next = _i(
            ops.cat([self.alphas_cumprod, self.alphas_cumprod.new_zeros([1])]),
            (t + stride).clamp(0, self.num_timesteps),
            xt,
        )

        # reverse sample
        mu = ops.sqrt(alphas_next) * x0 + ops.sqrt(1 - alphas_next) * eps
        return mu, x0

    # @torch.no_grad()
    def ddim_reverse_sample_loop(
        self, x0, model, model_kwargs={}, clamp=None, percentile=None, guide_scale=None, ddim_timesteps=20
    ):
        # prepare input
        b = x0.shape[0]
        xt = x0

        # reconstruction steps
        steps = ops.arange(0, self.num_timesteps, self.num_timesteps // ddim_timesteps)
        steps = tqdm(steps, desc="ddim_reverse_sample_loop")
        for step in steps:
            t = ops.full((b,), step, dtype=ms.int64)
            xt, _ = self.ddim_reverse_sample(xt, t, model, model_kwargs, clamp, percentile, guide_scale, ddim_timesteps)
        return xt

    # @torch.no_grad()
    def plms_sample(
        self,
        xt,
        t,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        condition_fn=None,
        guide_scale=None,
        plms_timesteps=20,
        eps_cache=None,
    ):
        r"""Sample from p(x_{t-1} | x_t) using PLMS.
        - condition_fn: for classifier-based guidance (guided-diffusion).
        - guide_scale: for classifier-free guidance (glide/dalle-2).
        """
        stride = self.num_timesteps // plms_timesteps

        # function for compute eps
        def compute_eps(xt, t):
            # predict distribution of p(x_{t-1} | x_t)
            _, _, _, x0 = self.p_mean_variance(xt, t, model, model_kwargs, clamp, percentile, guide_scale)

            # condition
            if condition_fn is not None:
                # x0 -> eps
                alpha = _i(self.alphas_cumprod, t, xt)
                eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(
                    self.sqrt_recipm1_alphas_cumprod, t, xt
                )
                eps = eps - (1 - alpha).sqrt() * condition_fn(xt, self._scale_timesteps(t), **model_kwargs)

                # eps -> x0
                x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps

            # derive eps
            eps = (_i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - x0) / _i(self.sqrt_recipm1_alphas_cumprod, t, xt)
            return eps

        # function for compute x_0 and x_{t-1}
        def compute_x0(eps, t):
            # eps -> x0
            x0 = _i(self.sqrt_recip_alphas_cumprod, t, xt) * xt - _i(self.sqrt_recipm1_alphas_cumprod, t, xt) * eps

            # deterministic sample
            alphas_prev = _i(self.alphas_cumprod, (t - stride).clamp(0), xt)
            direction = ops.sqrt(1 - alphas_prev) * eps
            mask = t.ne(0).float().view(-1, *((1,) * (xt.ndim - 1)))  # noqa
            xt_1 = ops.sqrt(alphas_prev) * x0 + direction
            return xt_1, x0

        # PLMS sample
        eps = compute_eps(xt, t)
        if len(eps_cache) == 0:
            # 2nd order pseudo improved Euler
            xt_1, x0 = compute_x0(eps, t)
            eps_next = compute_eps(xt_1, (t - stride).clamp(0))
            eps_prime = (eps + eps_next) / 2.0
        elif len(eps_cache) == 1:
            # 2nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (3 * eps - eps_cache[-1]) / 2.0
        elif len(eps_cache) == 2:
            # 3nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (23 * eps - 16 * eps_cache[-1] + 5 * eps_cache[-2]) / 12.0
        elif len(eps_cache) >= 3:
            # 4nd order pseudo linear multistep (Adams-Bashforth)
            eps_prime = (55 * eps - 59 * eps_cache[-1] + 37 * eps_cache[-2] - 9 * eps_cache[-3]) / 24.0
        xt_1, x0 = compute_x0(eps_prime, t)
        return xt_1, x0, eps

    # @torch.no_grad()
    def plms_sample_loop(
        self,
        noise,
        model,
        model_kwargs={},
        clamp=None,
        percentile=None,
        condition_fn=None,
        guide_scale=None,
        plms_timesteps=20,
    ):
        # prepare input
        b = noise.shape[0]
        xt = noise

        # diffusion process
        steps = (
            (1 + ops.arange(0, self.num_timesteps, self.num_timesteps // plms_timesteps))
            .clamp(0, self.num_timesteps - 1)
            .flip((0,))
        )
        eps_cache = []
        steps = tqdm(steps, desc="plms_sample_loop")
        for step in steps:
            # PLMS sampling step
            t = ops.full((b,), step, dtype=ms.int64)
            xt, _, eps = self.plms_sample(
                xt, t, model, model_kwargs, clamp, percentile, condition_fn, guide_scale, plms_timesteps, eps_cache
            )

            # update eps cache
            eps_cache.append(eps)
            if len(eps_cache) >= 4:
                eps_cache.pop(0)
        return xt

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * 1000.0 / self.num_timesteps
        return t
