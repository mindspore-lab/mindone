import logging
from functools import partial
from typing import Union

import numpy as np
from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
from lvdm.modules.networks.util import rearrange_in_gn5d_bs, rearrange_out_gn5d

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops

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
        self.sqrt_recip_alphas_cumprod = self.to_mindspore(np.sqrt(1.0 / alphas_cumprod))
        self.sqrt_recipm1_alphas_cumprod = self.to_mindspore(np.sqrt(1.0 / alphas_cumprod - 1))

    # def q_sample(self, x_start, t, noise):
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

        # consume more chip memory but faster
        if not self.perframe_ae:
            results = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x))
        else:  # consume less chip memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_result = ops.stop_gradient(
                    self.scale_factor * self.first_stage_model.encode(x[index : index + 1, :, :, :])
                )
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
            c = self.cond_stage_model.encode(Tensor(tokens))
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

    def construct(self, x: ms.Tensor, text_tokens: ms.Tensor, control=None, **kwargs):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape [bs, F, 3, 256, 256]
            text: text tokens padded to fixed shape [bs, 77]
            control: other conditions for future extension

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume unet3d input/output shape: (b c f h w)
                unet2d input/output shape: (b c h w)
        """

        # 1. get image/video latents z using vae
        z = self.get_latents(x)

        # 2. sample timestep and add noise to latents
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        noise = ops.randn_like(z)
        noisy_latents, snr = self.add_noise(z, noise, t)

        # 3. get condition embeddings
        cond = self.get_condition_embeddings(text_tokens, control)

        # 4.  unet forward, predict conditioned on conditions
        model_output = self.apply_model(
            noisy_latents,
            t,
            cond=cond,
            **kwargs,
        )

        # 5. compute loss
        if self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_element = self.compute_loss(model_output, target)
        loss_sample = self.reduce_loss(loss_element)

        if self.snr_gamma is not None:
            snr_gamma = ops.ones_like(snr) * self.snr_gamma
            # TODO: for v-pred, .../ (snr+1)
            # TODO: for beta zero rescale, consider snr=0
            # min{snr, gamma} / snr
            loss_weight = ops.stack((snr, snr_gamma), axis=0).min(axis=0) / snr
            loss = (loss_weight * loss_sample).mean()
        else:
            loss = loss_sample.mean()
            # loss = self.mse_mean(target, model_output)

        """
        # can be used to place more weights to high-score samples
        logvar_t = self.logvar[t]
        loss = loss_simple / ops.exp(logvar_t) + logvar_t
        loss = self.l_simple_weight * loss.mean()
        """

        return loss

    def compute_loss(self, pred, target):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
        elif self.loss_type == "l2":
            loss = self.mse_none(target, pred)
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def reduce_loss_2d(self, loss):
        # model output/loss shape: (b c h w)
        return loss.mean([1, 2, 3])

    def reduce_loss(self, loss):
        # model output/loss shape: (b c f h w)
        return loss.mean([1, 2, 3, 4])


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
