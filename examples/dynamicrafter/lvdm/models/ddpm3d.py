import random
import logging
from functools import partial
from typing import Union

import numpy as np
from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
from lvdm.modules.networks.util import rearrange_in_gn5d_bs, rearrange_out_gn5d

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops, mint
from mindspore import numpy as msnp

from mindone.utils.config import instantiate_from_config
from mindone.utils.misc import default, exists, extract_into_tensor

_logger = logging.getLogger(__name__)


class DDPM(nn.Cell):
    def __init__(
        self,
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
        rescale_betas_zero_snr=False,
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
        assert first_stage_key == "video"
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings

        self.dtype = mstype.float16 if use_fp16 else mstype.float32
        self.use_scheduler = scheduler_config is not None
        self.use_ema = use_ema
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
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

        assert loss_type in ["l1", "l2"], "Invalid loss type: {}".format(loss_type)
        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = Tensor(np.full(shape=(self.num_timesteps,), fill_value=logvar_init).astype(np.float32))
        if self.learn_logvar:
            self.logvar = Parameter(self.logvar, requires_grad=True)
        # self.mse_mean = nn.MSELoss(reduction="mean")
        # self.mse_none = nn.MSELoss(reduction="none")

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
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, "alphas have to be defined for each timestep"

        self.to_mindspore = partial(Tensor, dtype=self.dtype)
        self.betas = self.to_mindspore(betas)
        self.alphas_cumprod = self.to_mindspore(alphas_cumprod)
        self.alphas_cumprod_prev = self.to_mindspore(alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = self.to_mindspore(np.sqrt(alphas_cumprod))
        self.sqrt_one_minus_alphas_cumprod = self.to_mindspore(np.sqrt(1.0 - alphas_cumprod))
        self.log_one_minus_alphas_cumprod = self.to_mindspore(np.log(1.0 - alphas_cumprod))

        if self.parameterization != 'velocity':
            # self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
            # self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
            self.sqrt_recip_alphas_cumprod = self.to_mindspore(np.sqrt(1.0 / alphas_cumprod))
            self.sqrt_recipm1_alphas_cumprod = self.to_mindspore(np.sqrt(1.0 / alphas_cumprod - 1))
        else:
            # self.register_buffer('sqrt_recip_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
            # self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
            self.sqrt_recip_alphas_cumprod = ops.zeros_like(self.to_mindspore(alphas_cumprod))
            self.sqrt_recipm1_alphas_cumprod = ops.zeros_like(self.to_mindspore(alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.posterior_variance = self.to_mindspore(posterior_variance)
        # self.register_buffer('posterior_variance', to_torch(posterior_variance))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = self.to_mindspore(np.log(np.maximum(posterior_variance, 1e-20)))
        self.posterior_mean_coef1 = self.to_mindspore(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.posterior_mean_coef2 = self.to_mindspore((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * self.to_mindspore(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(Tensor(alphas_cumprod)) / (2. * 1 - Tensor(alphas_cumprod))
        elif self.parameterization == "velocity":
            lvlb_weights = ops.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * self.to_mindspore(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.lvlb_weights = lvlb_weights
        assert not ops.isnan(self.lvlb_weights).all()

    def add_noise(self, original_samples: ms.Tensor, noise: ms.Tensor, timestep: ms.Tensor) -> ms.Tensor:
        """
        return:
            noisy_samples: sample with scheduled noise, shape [bs, ...]
            snr: signla-to-noise ratio of each sample, determined by the sampled time step, the snr value is estimaed by (alpha/sigma)^2, shape [bs]
        """
        t = timestep
        x_start = original_samples
        alpha = extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape)
        sigma = extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        noisy_samples = alpha * x_start + sigma * noise

        # FIXME: if we apply beta zero rescale, we need to fix it.
        snr = (alpha / sigma) ** 2
        # [bs, 1, 1, 1] -> [bs]
        snr = snr.squeeze()

        return noisy_samples, snr

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

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = ops.randn_like(x_start)
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = ops.mse_loss(target, pred)
            else:
                loss = ops.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss


class LatentDiffusion(DDPM):
    def __init__(
        self,
        first_stage_config,
        cond_stage_config,
        unet_config,
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
        noise_strength=0,
        use_dynamic_rescale=False,
        base_scale=0.7,
        turning_step=400,
        interp_mode=False,
        fps_condition_type="fs",
        perframe_ae=False,
        logdir=None,
        rand_cond_frame=False,
        en_and_decode_n_samples_a_time=None,
        concat_mode=True,
        snr_gamma=None,
        *args,
        **kwargs,
    ):
        """
        Core latetn diffusion model
        Args:
            snr_gamma: if not None, use min-SNR weighting. If use, typical value is 5.
        Notes:
            - For SD, first_stage_model = vae, cond_stage_model = text_encoder, they are set to be not trainable by default.
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
        assert cond_stage_key == "caption"
        self.cond_stage_key = cond_stage_key
        self.noise_strength = noise_strength
        self.use_dynamic_rescale = use_dynamic_rescale
        self.interp_mode = interp_mode
        self.fps_condition_type = fps_condition_type
        self.perframe_ae = perframe_ae
        self.logdir = logdir
        self.rand_cond_frame = rand_cond_frame
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time
        self.scale_factor = scale_factor

        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            self.scale_arr = ms.Parameter(self.to_mindspore(scale_arr), requires_grad=False)

        # unet, note: to avoid change param name, don't change the var name
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.clip_denoised = False

        self.uniform_int = ops.UniformInt()
        self.cond_stage_forward = cond_stage_forward
        assert encoder_type in ["2d", "3d"]
        self.encoder_type = encoder_type
        assert uncond_type in ["zero_embed", "empty_seq"]
        self.uncond_type = uncond_type
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        if (snr_gamma is not None) and (snr_gamma > 0.0):
            self.snr_gamma = snr_gamma
        else:
            self.snr_gamma = None
        print("D--: snr gamma ", self.snr_gamma)

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

        self.shorten_cond_schedule = self.num_timesteps_cond > 1  # TODO: remove if not used

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model  # vae
        self.first_stage_model.set_train(False)
        for param in self.first_stage_model.get_parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            self.cond_stage_model.set_train(False)
            for param in self.cond_stage_model.get_parameters():
                param.requires_grad = False
        else:
            assert config != "__is_first_stage__"
            assert config != "__is_unconditional__"
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            self.cond_stage_model.set_train(True)
            for param in self.cond_stage_model.get_parameters():
                param.requires_grad = True

    def tokenize(self, c):
        tokenized_res = self.cond_stage_model.tokenize(c)
        return ms.Tensor(tokenized_res)

    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            # z = rearrange(z, 'b c t h w -> (b t) c h w')
            z = ops.transpose(z, (0, 2, 1, 3, 4))  # (b c t h w) -> (b t c h w)
            z = ops.reshape(z, (-1, z.shape[2], z.shape[3], z.shape[4]))  # (b t c h w) -> ((b t) c h w)
            reshape_back = True
        else:
            reshape_back = False

        if not self.perframe_ae:
            z = 1.0 / self.scale_factor * z
            results = self.first_stage_model.decode(z)
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1.0 / self.scale_factor * z[index : index + 1, :, :, :]
                frame_result = self.first_stage_model.decode(frame_z)
                results.append(frame_result)
            results = ops.cat(results, axis=0)

        if reshape_back:
            results = ops.reshape(results, (b, t, *results.shape[1:]))  # ((b t) c h w) -> (b t c h w)
            results = ops.transpose(results, (0, 2, 1, 3, 4))  # (b t c h w) -> (b c t h w)
        return results

    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            # x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = ops.transpose(x, (0, 2, 1, 3, 4))  # (b t c h w)
            x = ops.reshape(x, (-1, *x.shape[2:]))  # ((b t) c h w)
            reshape_back = True
        else:
            reshape_back = False
            b, t = None, None  # placeholder for graph mode

        # consume more chip memory but faster
        if not self.perframe_ae:
            # results = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x))
            results = self.scale_factor * self.first_stage_model.encode(x)
        else:  # consume less chip memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_result = self.scale_factor * self.first_stage_model.encode(x[index : index + 1, :, :, :])
                # frame_result = ops.stop_gradient(
                #     self.scale_factor * self.first_stage_model.encode(x[index : index + 1, :, :, :])
                # )
                results.append(frame_result)
            results = ops.cat(results, axis=0)

        if reshape_back:
            results = ops.reshape(results, (b, t, *results.shape[1:]))  # (b t c h w)
            results = ops.transpose(results, (0, 2, 1, 3, 4))  # (b c t h w)

        return results

    # predict previous sample, typically predict noise
    def apply_model(self, x_noisy: ms.Tensor, t: ms.Tensor, cond: Union[ms.Tensor, dict], return_ids=False, **kwargs):
        """
        args:
            cond: it can be a dictionary or a Tensor. When `cond` is a dictionary,
                it passes through `DiffusionWrapper` as keyword arguments. When it
                is a Tensor, it is the input argument of "c_concat" or `c_crossattn`
                depends on the predefined `conditioning_key`.
        """

        if isinstance(cond, ms.Tensor):
            key = "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            cond = {key: cond}
        prev_sample = self.model(x_noisy, t, **cond, **kwargs)

        if isinstance(prev_sample, tuple):
            return prev_sample[0]
        else:
            return prev_sample

    def get_latent_z(self, videos):
        b, c, t, h, w = videos.shape
        x = rearrange_out_gn5d(videos)
        z = self.encode_first_stage(x)
        z = rearrange_in_gn5d_bs(z, b=b)

        return z

    def get_latents_2d(self, x):
        B, C, H, W = x.shape
        if C != 3:
            # b h w c -> b c h w
            x = ops.transpose(x, (0, 3, 1, 2))

        z = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x))

        return z

    def get_latents(self, x):
        # "b f c h w -> (b f) c h w"
        B, F, C, H, W = x.shape
        if C != 3:
            raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
        x = ops.reshape(x, (-1, C, H, W))

        z = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x))

        # (b*f c h w) -> (b f c h w) -> (b c f h w )
        z = ops.reshape(z, (B, F, z.shape[1], z.shape[2], z.shape[3]))
        z = ops.transpose(z, (0, 2, 1, 3, 4))

        return z

    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            tokens, _ = self.cond_stage_model.tokenize(c)  # text -> tensor
            c = self.cond_stage_model(Tensor(tokens))
        else:
            raise NotImplementedError
            # assert hasattr(self.zcond_stage_model, self.cond_stage_forward)
            # c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_condition_embeddings(self, text_tokens, control=None):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        if self.cond_stage_trainable:
            text_emb = self.cond_stage_model(text_tokens)
        else:
            text_emb = ops.stop_gradient(self.cond_stage_model(text_tokens))
        cond = {"c_crossattn": text_emb}

        return cond

    def construct(self, *batch):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        return loss

    def compute_loss(self, x, c, **kwargs):
        # kwargs["fs"] = fs
        # t = ops.randint(0, self.num_timesteps, (x.shape[0],)).long()
        t = ops.randint(0, self.num_timesteps, (x.shape[0],))
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        noise = msnp.randn(x_start.shape)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        for k, v in cond.items():
            if isinstance(v, (list, tuple)):  # PYNATIVE: list, GRAPH: tuple
                assert len(v) == 1 and isinstance(v[0], ms.Tensor)
                cond[k] = v[0]
            else:
                raise ValueError

        model_output = self.apply_model(x_noisy, t, cond, **kwargs)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "velocity":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(axis=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict  


class LatentDiffusionWithEmbedding(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_latents(self, x):
        B, F, C, H, W = x.shape
        if C != 4:
            raise ValueError("Expect input shape (b f 4 h w), but get {}".format(x.shape))
        z = ops.stop_gradient(self.scale_factor * x)

        # (b f c h w) -> (b c f h w )
        z = ops.transpose(z, (0, 2, 1, 3, 4))
        return z

    def get_condition_embeddings(self, text_tokens, control=None):
        # text conditions embedding inputs for cross-attention
        text_emb = ops.stop_gradient(text_tokens)
        cond = {"c_crossattn": text_emb}

        return cond


class LatentVisualDiffusion(LatentDiffusion):
    def __init__(
        self,
        img_cond_stage_config,
        image_proj_stage_config,
        freeze_embedder=True,
        image_proj_model_trainable=True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.image_proj_model_trainable = image_proj_model_trainable
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self._init_img_ctx_projector(image_proj_stage_config, image_proj_model_trainable)

    def _init_img_ctx_projector(self, config, trainable):
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.set_train(False)
            for param in self.image_proj_model.get_parameters():
                param.requires_grad = False

    def _init_embedder(self, config, freeze=True):
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.set_train(False)
            for param in self.embedder.get_parameters():
                param.requires_grad = False

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self.compute_loss(x, c, **kwargs)
        return loss, loss_dict

    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        # video, caption, path, fps, frame_stride = batch
        video, text_emb, fps, frame_stride = batch
        ## x: b c t h w
        x = video.astype(ms.float32)

        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x)

        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = ops.rand(x.shape[0])
        else:
            random_num = ops.ones(x.shape[0])  ## by doning so, we can get text embedding and complete img emb for inference
        # prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        prompt_mask = (random_num < 2 * self.uncond_prob)[:, None, None]
        # input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")
        input_mask = 1 - ((random_num >= self.uncond_prob).astype(ms.float32) * (random_num < 3 * self.uncond_prob).astype(ms.float32))[:, None, None, None]

        null_prompt = self.get_learned_conditioning([""])
        cond_emb = text_emb.astype(null_prompt.dtype)
        prompt_imb = ops.where(prompt_mask, null_prompt, cond_emb)

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)

        img = x[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)
        img_emb = img_emb.astype(ms.float32)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = ops.zeros_like(z)
                img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[:,:,cond_frame_index,:,:]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                # img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
                img_cat_cond = mint.repeat_interleave(img_cat_cond, repeats=z.shape[2], dim=2)

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [ops.cat([prompt_imb, img_emb], axis=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = frame_stride.astype(ms.float32)
            elif self.fps_condition_type == 'fps':
                fs = fps.astype(ms.float32)
            out.append(fs)
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        return out

# latent diffusion (unet) forward based on input noised latent and encoded conditions
class DiffusionWrapper(nn.Cell):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        # unet
        self.diffusion_model = instantiate_from_config(diff_model_config)  # don't change the var name here
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, "concat", "crossattn", "hybrid", "adm", "crossattn-adm"]

    def construct(self, x, t, c_concat=None, c_crossattn=None, c_adm=None, **kwargs):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t, **kwargs)
        elif self.conditioning_key == "concat":
            x_concat = ops.concat((x, c_concat), axis=1)
            out = self.diffusion_model(x_concat, t, **kwargs)
        elif self.conditioning_key == "crossattn":  # t2v task
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
