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
import logging
from functools import partial

import numpy as np
from ldm.modules.diffusionmodules.util import make_beta_schedule
from ldm.util import default, exists, extract_into_tensor, instantiate_from_config

from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn
from mindspore import numpy as msnp
from mindspore import ops

_logger = logging.getLogger(__name__)


def disabled_train(self, mode=True):
    """
    Overwrite model.set_train with this function to make sure train/eval mode does not change anymore.
    """
    self.set_train(False)
    return self


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
        image_size=256,
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
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.dtype = mstype.float16 if use_fp16 else mstype.float32
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
        self.logvar = Tensor(np.full(shape=(self.num_timesteps,), fill_value=logvar_init).astype(np.float32))
        if self.learn_logvar:
            self.logvar = Parameter(self.logvar, requires_grad=True)
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
                beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s
            )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        to_mindspore = partial(Tensor, dtype=self.dtype)
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

    def predict_start_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )


class MarigoldLatentDiffusion(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        """
        main class
        """
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", Tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.uniform_int = ops.UniformInt()

        self.apply_multi_res_noise = True
        self.mr_noise_downscale_strategy = "original"
        self.mr_noise_strength = 0.9
        self.annealed_mr_noise = True

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.transpose = ops.Transpose()

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model
        for param in self.first_stage_model.get_parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            for param in self.cond_stage_model.get_parameters():
                param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def tokenize(self, c):
        tokenized_res = self.cond_stage_model.tokenize(c)
        return tokenized_res

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            c = self.cond_stage_model.encode(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_learned_conditioning_fortrain(self, c):
        c = self.cond_stage_model(c)
        return c

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, z):
        return self.scale_factor * z

    def apply_model(self, x_noisy, t, cond, return_ids=False, **kwargs):
        """
        args:
            cond: it can be a dictionary or a Tensor. When `cond` is a dictionary,
                it passes through `DiffusionWrapper` as keyword arguments. When it
                is a Tensor, it is the input argument of "c_concat" or `c_crossattn`
                depends on the predefined `conditioning_key`.
        """
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def get_input(self, x, c):
        # if len(x.shape) == 3:
        #     x = x[...,None]
        # x = self.transpose(x, (2, 0, 3, 1))
        z = ops.stop_gradient(self.get_first_stage_encoding(self.encode_first_stage(x)))
        return z, c

    def construct(
        self,
        rgb_int,
        rgb_norm,
        depth_raw_linear,
        depth_filled_linear,
        valid_mask_raw,
        valid_mask_filled,
        depth_raw_norm,
        depth_filled_norm,
        c,
    ):
        t = self.uniform_int(
            (rgb_norm.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        if len(depth_raw_norm.shape) == 4:
            depth_raw_norm = ops.Tile()(depth_raw_norm, (1, 3, 1, 1))
        elif len(depth_raw_norm.shape) == 3:
            depth_raw_norm = ops.ExpandDims()(depth_raw_norm, 1)
            depth_raw_norm = ops.Tile()(depth_raw_norm, (1, 3, 1, 1))
        depth_latent, c = self.get_input(depth_raw_norm, c)
        rgb_latent, c = self.get_input(rgb_norm, c)
        invalid_mask = ~valid_mask_raw
        valid_mask_down = ~(ops.MaxPool(8, 8)(invalid_mask.astype(mstype.float32)).bool())
        valid_mask_down = ops.Tile()(valid_mask_down, (1, 4, 1, 1))
        valid_mask_down = valid_mask_down.astype(self.dtype)
        c = self.get_learned_conditioning_fortrain(c)
        depth_latent = depth_latent.to(self.dtype)
        return self.p_losses(depth_latent, rgb_latent, c, t, valid_mask_down)

    def p_losses(self, x_start, rgb_latent, cond, t, valid_mask, noise=None, **kwargs):
        # noise = self.get_multiscale_noise(x_start, t)
        noise = msnp.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = x_noisy.to(self.dtype)
        rgb_latent = rgb_latent.to(self.dtype)
        full_input = ops.cat((x_noisy, rgb_latent), 1)
        model_output = self.apply_model(
            full_input,
            t,
            cond=cond,
            **kwargs,
        )

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            # target = sqrt_alpha_cum * noise - sqrt_one_minus_alpha_prod * x_start
            target = self.get_velocity(x_start, noise, t)  # TODO: parse train step from randint
        else:
            raise NotImplementedError()

        loss_simple = ops.sum(self.get_loss(model_output * valid_mask, target * valid_mask, mean=False)) / ops.sum(
            valid_mask
        )

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        # NOTE: original_elbo_weight is never set larger than 0. Diffuser remove it too. Let's remove it to save mem.
        # loss_vlb = self.get_loss(model_output, target, mean=False).mean((1, 2, 3))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss += (self.original_elbo_weight * loss_vlb)

        return loss


class LatentDiffusion(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        *args,
        **kwargs,
    ):
        """
        main class
        """
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", Tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.uniform_int = ops.UniformInt()

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.transpose = ops.Transpose()

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model
        for param in self.first_stage_model.get_parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            for param in self.cond_stage_model.get_parameters():
                param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def tokenize(self, c):
        tokenized_res = self.cond_stage_model.tokenize(c)
        return tokenized_res

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            c = self.cond_stage_model.encode(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_learned_conditioning_fortrain(self, c):
        c = self.cond_stage_model(c)
        return c

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, z):
        return self.scale_factor * z

    def apply_model(self, x_noisy, t, cond, return_ids=False, **kwargs):
        """
        args:
            cond: it can be a dictionary or a Tensor. When `cond` is a dictionary,
                it passes through `DiffusionWrapper` as keyword arguments. When it
                is a Tensor, it is the input argument of "c_concat" or `c_crossattn`
                depends on the predefined `conditioning_key`.
        """
        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def get_input(self, x, c):
        if len(x.shape) == 3:
            x = x[..., None]
        x = self.transpose(x, (0, 3, 1, 2))
        z = ops.stop_gradient(self.get_first_stage_encoding(self.encode_first_stage(x)))
        return z, c

    def construct(self, x, c):
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        x, c = self.get_input(x, c)
        c = self.get_learned_conditioning_fortrain(c)
        return self.p_losses(x, c, t)

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        noise = msnp.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(
            x_noisy,
            t,
            cond=cond,
            **kwargs,
        )

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            # target = sqrt_alpha_cum * noise - sqrt_one_minus_alpha_prod * x_start
            target = self.get_velocity(x_start, noise, t)  # TODO: parse train step from randint
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        # NOTE: original_elbo_weight is never set larger than 0. Diffuser remove it too. Let's remove it to save mem.
        # loss_vlb = self.get_loss(model_output, target, mean=False).mean((1, 2, 3))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss += (self.original_elbo_weight * loss_vlb)

        return loss


# latent diffusion (unet) forward based on input noised latent and encoded conditions
class DiffusionWrapper(nn.Cell):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm", "crossattn-adm"]

    def construct(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, **kwargs):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, **kwargs)
        elif self.conditioning_key == "concat":
            x_concat = ops.concat((x, c_concat), axis=1)
            out = self.diffusion_model(x_concat, t, **kwargs)
        elif self.conditioning_key == "crossattn":
            context = c_crossattn
            out = self.diffusion_model(x, t, context=context, **kwargs)
        elif self.conditioning_key == "hybrid":
            x_concat = ops.concat((x, c_concat), axis=1)
            context = c_crossattn
            out = self.diffusion_model(x_concat, t, context=context, **kwargs)
        elif self.conditioning_key == "crossattn-adm":
            context = c_crossattn
            out = self.diffusion_model(x, t, context=context, y=c_adm, **kwargs)
        elif self.conditioning_key == "adm":
            cc = c_crossattn
            out = self.diffusion_model(x, t, y=cc, **kwargs)
        else:
            raise NotImplementedError()

        return out


class LatentDiffusionDB(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        num_timesteps_cond=None,
        cond_stage_key="image",
        cond_stage_trainable=False,
        concat_mode=True,
        cond_stage_forward=None,
        conditioning_key=None,
        scale_factor=1.0,
        scale_by_std=False,
        reg_weight=1.0,
        *args,
        **kwargs,
    ):
        """
        main class
        """
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        if conditioning_key is None:
            conditioning_key = "concat" if concat_mode else "crossattn"
        if cond_stage_config == "__is_unconditional__":
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.reg_weight = reg_weight
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except Exception:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer("scale_factor", Tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None
        self.uniform_int = ops.UniformInt()

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        self.transpose = ops.Transpose()

    def register_schedule(
        self,
        given_betas=None,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model
        for param in self.first_stage_model.get_parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            c = self.cond_stage_model.encode(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_learned_conditioning_fortrain(self, c):
        c = self.cond_stage_model(c)
        return c

    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        x_noisy = ops.cast(x_noisy, self.dtype)
        cond = ops.cast(cond, self.dtype)

        if isinstance(cond, dict):
            # hybrid case, cond is expected to be a dict
            pass
        else:
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def get_input(self, x, c):
        if len(x.shape) == 3:
            x = x[..., None]
        x = self.transpose(x, (0, 3, 1, 2))
        z = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x))

        return z, c

    def shared_step(self, x, c):
        x, c = self.get_input(x, c)
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        c = self.get_learned_conditioning_fortrain(c)
        loss = self.p_losses(x, c, t)
        return loss

    def construct(self, train_x, train_c, reg_x, reg_c):
        loss_train = self.shared_step(train_x, train_c)
        loss_reg = self.shared_step(reg_x, reg_c)
        loss = loss_train + self.reg_weight * loss_reg
        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        noise = msnp.randn(x_start.shape)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            # target = sqrt_alpha_cum * noise - sqrt_one_minus_alpha_prod * x_start
            target = self.get_velocity(x_start, noise, t)  # TODO: parse train step from randint
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()

        # loss_vlb = self.get_loss(model_output, target, mean=False).mean((1, 2, 3))
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss += (self.original_elbo_weight * loss_vlb)

        return loss


class LatentDiffusionDreamBooth(LatentDiffusion):
    def __init__(self, prior_loss_weight=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prior_loss_weight = prior_loss_weight

    def shared_step(self, x, c):
        x, c = self.get_input(x, c)
        t = ops.UniformInt()(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        c = self.get_learned_conditioning_fortrain(c)
        loss = self.p_losses(x, c, t)
        return loss

    def construct(self, *args):
        if self.prior_loss_weight != 0:
            train_x, train_c, reg_x, reg_c = args
            loss_train = self.shared_step(train_x, train_c)
            loss_reg = self.shared_step(reg_x, reg_c)
            loss = loss_train + self.prior_loss_weight * loss_reg
        else:
            train_x, train_c = args
            loss_train = self.shared_step(train_x, train_c)
            loss = loss_train
        return loss


class LatentInpaintDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("mask", "masked_image"),
        masked_image_key="masked_image",
        finetune_keys=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.masked_image_key = masked_image_key
        assert self.masked_image_key in concat_keys
        self.concat_keys = concat_keys


class LatentDepthDiffusion(LatentDiffusion):
    def __init__(
        self,
        concat_keys=("depth"),
        finetune_keys=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.concat_keys = concat_keys


class ImageEmbeddingConditionedLatentDiffusion(LatentDiffusion):
    def __init__(
        self,
        embedder_config,
        embedding_dropout=0.5,
        freeze_embedder=True,
        noise_aug_config=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedding_dropout = embedding_dropout
        self._init_embedder(embedder_config, freeze_embedder)
        self._init_noise_aug(noise_aug_config)

    def _init_embedder(self, config, freeze=True):
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.set_train(False)
            for param in self.embedder.get_parameters():
                param.requires_grad = False

    def _init_noise_aug(self, config):
        if config is not None:
            self.noise_augmentor = instantiate_from_config(config)
            self.noise_augmentor.set_train(False)
            for param in self.noise_augmentor.get_parameters():
                param.requires_grad = False
        else:
            self.noise_augmentor = None

    def get_input(self, x, c):
        z, c = super().get_input(x, c)
        x = self.transpose(x, (0, 3, 1, 2))
        c_adm = self.embedder(x)
        if self.noise_augmentor is not None:
            c_adm, noise_level_emb = self.noise_augmentor(c_adm)
            # assume this gives embeddings of noise levels
            c_adm = ops.concat((c_adm, noise_level_emb), 1)

        if self.training:
            c_adm = (ops.rand((c_adm.shape[0], 1)) > self.embedding_dropout) * c_adm
        return z, c, c_adm

    def construct(self, x, c):
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        x, c, c_adm = self.get_input(x, c)
        c = self.get_learned_conditioning_fortrain(c)
        cond = {"c_crossattn": c, "c_adm": c_adm}
        return self.p_losses(x, cond, t)


class InflatedLatentDiffusion(LatentDiffusion):
    def get_input(self, x, c):
        assert len(x.shape) == 5, f"expect the input image shape is (b, f, h, w, c), but got {x.shape}"
        x = self.transpose(x, (0, 1, 4, 2, 3))  # (b, f, h, w, c)-> (b, f, c, h, w)
        b, f, ch, h, w = x.shape
        z = ops.stop_gradient(self.get_first_stage_encoding(self.encode_first_stage(x.reshape((b * f, ch, h, w)))))
        _, ch, h, w = z.shape
        z = self.transpose(z.reshape((b, f, ch, h, w)), (0, 2, 1, 3, 4))  # (b, f, c, h, w) - > (b, c, f, h, w)
        return z, c
