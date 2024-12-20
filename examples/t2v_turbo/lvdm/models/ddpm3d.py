"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import logging
from contextlib import contextmanager
from functools import partial

import numpy as np
from tqdm import tqdm

mainlogger = logging.getLogger("mainlogger")

from lvdm.basics import disabled_train
from lvdm.common import default, exists, extract_into_tensor, noise_like
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.ema import LitEma
from lvdm.models.utils_diffusion import make_beta_schedule
from lvdm.modules.encoders.ip_resampler import ImageProjModel, Resampler
from utils.utils import instantiate_from_config

import mindspore as ms
from mindspore import mint, nn, ops

__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}


class DDPM(nn.Cell):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(
        self,
        unet_config,
        timesteps=1000,
        beta_schedule="linear",
        loss_type="l2",
        ckpt_path=None,
        ignore_keys=[],
        load_only_unet=False,
        monitor=None,
        use_ema=True,
        first_stage_key="image",
        image_size=256,
        channels=3,
        log_every_t=100,
        clip_denoised=True,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        given_betas=None,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        conditioning_key=None,
        parameterization="eps",  # all assuming fixed variance schedules
        scheduler_config=None,
        use_positional_encodings=False,
        learn_logvar=False,
        logvar_init=0.0,
        use_fp16=False,
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

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
        self.logvar = ops.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = ms.Parameter(self.logvar, requires_grad=True)

    def to(self, dtype):
        self.to_float(dtype)

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

        to_ms = partial(ms.Tensor, dtype=self.dtype)

        self.betas = to_ms(betas)
        self.alphas_cumprod = to_ms(alphas_cumprod)
        self.alphas_cumprod_prev = to_ms(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_ms(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = to_ms(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = to_ms(np.log(1.0 - alphas_cumprod))
        self.sqrt_recip_alphas_cumprod = to_ms(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = to_ms(np.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        ) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = to_ms(posterior_variance)
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = to_ms(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = to_ms(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.posterior_mean_coef2 = to_ms((1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod))

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (2 * self.posterior_variance * to_ms(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(ms.Tensor(alphas_cumprod)) / (2.0 * 1 - ms.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = to_ms(lvlb_weights)
        assert not ops.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.get_parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")

        yield None

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = ms.load_checkpoint(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = (
            ms.load_param_into_net(self, sd) if not only_model else ms.load_param_into_net(self.model, sd)
        )
        mainlogger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, shape, return_intermediates=False):
        b = shape[0]
        img = ops.randn(shape)
        intermediates = [img]
        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="Sampling t",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                ops.full((b,), i, dtype=ms.int32),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: mint.randn_like(x_start))
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
            * x_start
            * extract_into_tensor(self.scale_arr, t, x_start.shape)
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_input(self, batch, k):
        x = batch[k]
        x = x.float()
        return x


class LatentDiffusion(DDPM):
    """main class"""

    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="caption",
        cond_stage_trainable=False,
        cond_stage_forward=None,
        conditioning_key=None,
        uncond_prob=0.2,
        uncond_type="empty_seq",
        scale_factor=1.0,
        scale_by_std=False,
        encoder_type="2d",
        only_model=False,
        use_scale=False,
        scale_a=1,
        scale_b=0.3,
        mid_step=400,
        fix_scale_bug=False,
        *args,
        **kwargs,
    ):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs["timesteps"]
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, "crossattn")
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        # scale factor
        self.use_scale = use_scale
        if self.use_scale:
            self.scale_a = scale_a
            self.scale_b = scale_b
            if fix_scale_bug:
                scale_step = self.num_timesteps - mid_step
            else:  # bug
                scale_step = self.num_timesteps

            scale_arr1 = np.linspace(scale_a, scale_b, mid_step)
            scale_arr2 = np.full(scale_step, scale_b)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_ms = partial(ms.Tensor, dtype=self.dtype)
            self.scale_arr = to_ms(scale_arr)

        self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1

        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.scale_factor = ms.Tensor(scale_factor)
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert encoder_type in ["2d", "3d"]
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert uncond_type in ["zero_embed", "empty_seq"]
        self.uncond_type = uncond_type

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True

    def make_cond_schedule(
        self,
    ):
        self.cond_ids = ops.full(
            size=(self.num_timesteps,),
            fill_value=self.num_timesteps - 1,
            dtype=ms.int32,
        )
        ids = ops.round(mint.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[: self.num_timesteps_cond] = ids

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: mint.randn_like(x_start))
        if self.use_scale:
            return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
                * x_start
                * extract_into_tensor(self.scale_arr, t, x_start.shape)
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )
        else:
            return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
            )

    def _freeze_model(self):
        for name, para in self.model.diffusion_model.parameters_dict():
            para.requires_grad = False

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        model.set_train(False)
        self.first_stage_model = model
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.get_parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            model.set_train(False)
            self.cond_stage_model = model
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.get_parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, "encode") and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample_with_noise(noise=noise)
        elif isinstance(encoder_posterior, ms.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, c, t, h, w = x.shape
            x = x.reshape(-1, c, h, w)
            # x = rearrange(x, "b c t h w -> (b t) c h w")
            reshape_back = True
        else:
            reshape_back = False

        encoder_posterior = self.first_stage_model.encode(x)
        results = self.get_first_stage_encoding(encoder_posterior).detach()

        if reshape_back:
            _, c, h, w = results.shape
            results = results.reshape(b, c, t, h, w)
            # results = rearrange(results, "(b t) c h w -> b c t h w", b=b, t=t)

        return results

    def encode_first_stage_2DAE(self, x):
        b, _, t, _, _ = x.shape
        results = mint.cat(
            [
                self.get_first_stage_encoding(self.first_stage_model.encode(x[:, :, i])).detach().unsqueeze(2)
                for i in range(t)
            ],
            dim=2,
        )

        return results

    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, c, t, h, w = z.shape
            z = z.reshape(-1, c, h, w)
            # z = rearrange(z, "b c t h w -> (b t) c h w")
            reshape_back = True
        else:
            reshape_back = False

        z = 1.0 / self.scale_factor * z

        results = self.first_stage_model.decode(z, **kwargs)

        if reshape_back:
            _, c, h, w = results.shape
            results = results.reshape(b, c, t, h, w)
            # results = rearrange(results, "(b t) c h w -> b c t h w", b=b, t=t)
        return results

    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def decode_first_stage_2DAE(self, z, **kwargs):
        b, _, t, _, _ = z.shape
        z = 1.0 / self.scale_factor * z
        results = mint.cat(
            [self.first_stage_model.decode(z[:, :, i], **kwargs).unsqueeze(2) for i in range(t)],
            dim=2,
        )

        return results

    def p_mean_variance(
        self,
        x,
        c,
        t,
        clip_denoised: bool,
        return_x0=False,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self,
        x,
        c,
        t,
        clip_denoised=False,
        repeat_noise=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        **kwargs,
    ):
        b = x.shape[0]
        outputs = self.p_mean_variance(
            x=x,
            c=c,
            t=t,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            score_corrector=score_corrector,
            corrector_kwargs=corrector_kwargs,
            **kwargs,
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = mint.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(
        self,
        cond,
        shape,
        return_intermediates=False,
        x_T=None,
        verbose=True,
        callback=None,
        timesteps=None,
        mask=None,
        x0=None,
        img_callback=None,
        start_T=None,
        log_every_t=None,
        **kwargs,
    ):
        if not log_every_t:
            log_every_t = self.log_every_t
        b = shape[0]
        # sample an initial noise
        if x_T is None:
            img = ops.randn(shape)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = (
            tqdm(reversed(range(0, timesteps)), desc="Sampling t", total=timesteps)
            if verbose
            else reversed(range(0, timesteps))
        )

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = ops.full((b,), i, dtype=ms.int32)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != "hybrid"
                tc = self.cond_ids[ts]
                cond = self.q_sample(x_start=cond, t=tc, noise=mint.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1.0 - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback:
                callback(i)
            if img_callback:
                img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img


class LatentVisualDiffusion(LatentDiffusion):
    def __init__(self, cond_img_config, finegrained=False, random_cond=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.random_cond = random_cond
        self.instantiate_img_embedder(cond_img_config, freeze=True)
        num_tokens = 16 if finegrained else 4
        self.image_proj_model = self.init_projector(
            use_finegrained=finegrained,
            num_tokens=num_tokens,
            input_dim=1024,
            cross_attention_dim=1024,
            dim=1280,
        )

    def instantiate_img_embedder(self, config, freeze=True):
        embedder = instantiate_from_config(config)
        if freeze:
            self.embedder = embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.get_parameters():
                param.requires_grad = False

    def init_projector(self, use_finegrained, num_tokens, input_dim, cross_attention_dim, dim):
        if not use_finegrained:
            image_proj_model = ImageProjModel(
                clip_extra_context_tokens=num_tokens,
                cross_attention_dim=cross_attention_dim,
                clip_embeddings_dim=input_dim,
            )
        else:
            image_proj_model = Resampler(
                dim=input_dim,
                depth=4,
                dim_head=64,
                heads=12,
                num_queries=num_tokens,
                embedding_dim=dim,
                output_dim=cross_attention_dim,
                ff_mult=4,
            )
        return image_proj_model

    # Never delete this func: it is used in log_images() and inference stage
    def get_image_embeds(self, batch_imgs):
        # img: b c h w
        img_token = self.embedder(batch_imgs)
        img_emb = self.image_proj_model(img_token)
        return img_emb


class DiffusionWrapper(nn.Cell):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def construct(
        self,
        x,
        t,
        c_concat: list = None,
        c_crossattn: list = None,
        c_adm=None,
        s=None,
        mask=None,
        **kwargs,
    ):
        # temporal_context = fps is foNone
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == "concat":
            xc = mint.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == "crossattn":
            cc = mint.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == "hybrid":
            # it is just right [b,c,t,h,w]: concatenate in channel dim
            xc = mint.cat([x] + c_concat, dim=1)
            cc = mint.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == "resblockcond":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == "adm":
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == "hybrid-adm":
            assert c_adm is not None
            xc = mint.cat([x] + c_concat, dim=1)
            cc = mint.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm)
        elif self.conditioning_key == "hybrid-time":
            assert s is not None
            xc = mint.cat([x] + c_concat, dim=1)
            cc = mint.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == "concat-time-mask":
            # assert s is not None
            # mainlogger.info('x & mask:',x.shape,c_concat[0].shape)
            xc = mint.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == "concat-adm-mask":
            # assert s is not None
            # mainlogger.info('x & mask:',x.shape,c_concat[0].shape)
            if c_concat is not None:
                xc = mint.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == "hybrid-adm-mask":
            cc = mint.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = mint.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif self.conditioning_key == "hybrid-time-adm":  # adm means y, e.g., class index
            # assert s is not None
            assert c_adm is not None
            xc = mint.cat([x] + c_concat, dim=1)
            cc = mint.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        else:
            raise NotImplementedError()

        return out
