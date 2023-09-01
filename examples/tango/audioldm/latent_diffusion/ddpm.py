"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import logging
from collections import namedtuple
from functools import partial

import numpy as np
from audioldm.latent_diffusion.ldm_util import extract_into_tensor
from audioldm.latent_diffusion.util import make_beta_schedule
from audioldm.utils import exists, instantiate_from_config

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

_logger = logging.getLogger(__name__)


__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


def disabled_train(self, mode=True):
    """
    Overwrite model.set_train with this function to make sure train/eval mode does not change anymore.
    """
    self.set_train(False)
    return self


class DDPMScheduler(nn.Cell):
    def __init__(self):
        super().__init__()
        self.config = namedtuple("Conf", ["prediction_type"])("v_prediction")

    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def add_noise(self, latents, noise, t):
        return self.q_sample(latents, t, noise)

    def get_velocity(self, sample, noise, t):
        # TODO: how t affects noise mean and variance here. all variance fixed?
        v = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, sample.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, sample.shape) * sample
        )
        return v


class DiffusionWrapper(nn.Cell):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm"]

    def construct(self, x, t, c_concat=None, c_crossattn=None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            x_concat = ops.concat((x, c_concat), axis=1)
            out = self.diffusion_model(x_concat, t)
        elif self.conditioning_key == "crossattn":
            context = c_crossattn
            out = self.diffusion_model(x, t, context=context)
        elif self.conditioning_key == "hybrid":
            x_concat = ops.concat((x, c_concat), axis=1)
            context = c_crossattn
            out = self.diffusion_model(x_concat, t, context=context)
        elif self.conditioning_key == "adm":
            cc = c_crossattn
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class DDPM(nn.Cell):
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor="val/loss",
        use_ema=True,
        first_stage_key="image",
        latent_t_size=256,
        latent_f_size=16,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        use_fp16=False,
    ):
        """
        Classic DDPM with Gaussian diffusion
        ===============================================================
        Args:
            v_posterior: weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta.
            parameterization:
                eps - epsilon (predicting the noise of the diffusion process)
                x0 - orginal (latent) image (directly predicting the noisy sample)
                velocity - velocity of z (see section 2.4 https://imagen.research.google/video/paper.pdf).
        """

        super().__init__()
        assert parameterization in ["eps", "x0", "velocity"], 'currently only supporting "eps", "velocity" and "x0"'
        self.parameterization = parameterization
        _logger.debug(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size

        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.use_scheduler = scheduler_config is not None
        self.use_ema = use_ema
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        assert original_elbo_weight == 0.0, "Variational lower bound loss has been removed."

        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        self.isnan = ops.IsNan()
        self.register_schedule(
            given_betas=given_betas,
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = ms.Tensor(np.full(shape=(self.num_timesteps,), fill_value=logvar_init).astype(np.float32))
        if self.learn_logvar:
            self.logvar = ms.Parameter(self.logvar, requires_grad=True)
        self.randn_like = ops.StandardNormal()
        self.mse_mean = nn.MSELoss(reduction="mean")
        self.mse_none = nn.MSELoss(reduction="none")

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(
                beta_schedule,
                timesteps,
                linear_start=linear_start,
                linear_end=linear_end,
                cosine_s=cosine_s,
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_mindspore = partial(ms.Tensor, dtype=self.dtype)
        self.betas = to_mindspore(betas)
        self.alphas_cumprod = to_mindspore(alphas_cumprod)
        self.alphas_cumprod_prev = to_mindspore(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_mindspore(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_mindspore(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_mindspore(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_mindspore(np.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_mindspore(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_mindspore(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_mindspore(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = to_mindspore((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        # if self.parameterization == "eps":
        #    lvlb_weights = self.betas ** 2 / (
        #                2 * self.posterior_variance * to_mindspore(alphas) * (1 - self.alphas_cumprod))
        # elif self.parameterization == "x0":
        #    lvlb_weights = 0.5 * msnp.sqrt(Tensor(alphas_cumprod)) / (2. * 1 - Tensor(alphas_cumprod))
        # elif self.parameterization == "velocity":
        #    # TODO: confirm
        #    lvlb_weights = self.betas ** 2 / (
        #                2 * self.posterior_variance * to_mindspore(alphas) * (1 - self.alphas_cumprod))
        # else:
        #    raise NotImplementedError("mu not supported")
        # lvlb_weights[0] = lvlb_weights[1]
        # self.lvlb_weights = to_mindspore(lvlb_weights)

    def get_velocity(self, sample, noise, t):
        # TODO: how t affects noise mean and variance here. all variance fixed?
        v = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, sample.shape) * noise
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, sample.shape) * sample
        )
        return v

    # TODO: it's a good practice. May adopt it later.
    # with ema_scopte(): save_model(), run_eval()
    """
    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            parameters = self.model.get_parameters()
            trained_parameters = [param for param in parameters if param.requires_grad is True ]
            self.model_ema.store(iter(trained_parameters))
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                parameters = self.model.get_parameters()
                trained_parameters = [param for param in parameters if param.requires_grad is True]
                self.model_ema.restore(iter(trained_parameters))
                if context is not None:
                    print(f"{context}: Restored training weights")
    """

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "l2":
            if mean:
                loss = nn.MSELoss(reduction="mean")(target, pred)
            else:
                loss = nn.MSELoss(reduction="none")(target, pred)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def q_sample(self, x_start, t, noise):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
