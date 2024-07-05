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
from typing import Union
import random

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops
# from mindcv.optim.adamw import AdamW as AdamW_Refined
# from mindspore.nn.optim import AdamWeightDecay, Momentum, Optimizer
from mindspore.nn import AdamWeightDecay

from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
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
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
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
        fps_condition_type='fs',
        perframe_ae=False,
        # added
        logdir=None,
        rand_cond_frame=False,
        en_and_decode_n_samples_a_time=None,
        # first_stage_key="image",  # from MS
        concat_mode=True,           # from MS
        # emb_cache=False,          # from MS
        snr_gamma=None,           # from MS
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

        # try:
        #     self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        # except Exception:
        #     self.num_downs = 0

        self.scale_factor = scale_factor
        # if not scale_by_std:
        #     self.scale_factor = scale_factor
        # else:
        #     self.register_buffer("scale_factor", Tensor(scale_factor))

        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            # to_torch = partial(torch.tensor, dtype=torch.float32)
            # self.register_buffer('scale_arr', to_torch(scale_arr))
            self.scale_arr = ms.Parameter(self.to_mindspore(scale_arr), requires_grad=False)

        # unet, note: to avoid change param name, don't change the var name
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.clip_denoised = False

        self.uniform_int = ops.UniformInt()
        self.cond_stage_forward = cond_stage_forward
        assert(encoder_type in ["2d", "3d"])
        self.encoder_type = encoder_type
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type
        self.uncond_prob = uncond_prob
        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        # self.emb_cache = emb_cache

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
        # import pdb;pdb.set_trace()
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            # z = rearrange(z, 'b c t h w -> (b t) c h w')
            z = ops.transpose(z, (0, 2, 1, 3, 4))  #  (b c t h w) -> (b t c h w)
            z = ops.reshape(z, (-1, z.shape[2], z.shape[3], z.shape[4]))  # (b t c h w) -> ((b t) c h w)
            reshape_back = True
        else:
            reshape_back = False
            
        if not self.perframe_ae:    
            z = 1. / self.scale_factor * z
            # results = self.first_stage_model.decode(z, **kwargs)
            results = self.first_stage_model.decode(z)
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1. / self.scale_factor * z[index:index+1,:,:,:]
                # frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                frame_result = self.first_stage_model.decode(frame_z)
                results.append(frame_result)
            results = ops.cat(results, axis=0)

        if reshape_back:
            # results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
            results = ops.reshape(results, (b, t, *results.shape[1:]))  # ((b t) c h w) -> (b t c h w)
            results = ops.transpose(results, (0, 2, 1, 3, 4))  # (b t c h w) -> (b c t h w)
        return results

    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)
     
    # def decode_first_stage(self, z):
    #     z = 1.0 / self.scale_factor * z
    #     return self.first_stage_model.decode(z)  # lvdm.models.autoencoder.AutoencoderKL

    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            # x = rearrange(x, 'b c t h w -> (b t) c h w')
            x = ops.transpose(x, (0, 2, 1, 3, 4))  # (b t c h w)
            x = ops.reshape(x, (-1, *x.shape[2:]))  # ((b t) c h w)
            reshape_back = True
        else:
            reshape_back = False

        ## consume more GPU memory but faster
        if not self.perframe_ae:
            results = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x))
            # encoder_posterior = self.first_stage_model.encode(x)
            # results = self.get_first_stage_encoding(encoder_posterior)
        else:  ## consume less GPU memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_result = ops.stop_gradient(self.scale_factor * self.first_stage_model.encode(x[index:index+1,:,:,:]))
                # frame_batch = self.first_stage_model.encode(x[index:index+1,:,:,:])
                # frame_result = self.get_first_stage_encoding(frame_batch)
                results.append(frame_result)
            results = ops.cat(results, axis=0)

        if reshape_back:
            # results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
            x = ops.reshape(x, (b, t, *x.shape[1:]))  # (b t c h w)
            x = ops.transpose(x, (0, 2, 1, 3, 4))  # (b c t h w)

        return results

    # def encode_first_stage(self, x):
    #     return self.first_stage_model.encode(x)

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

    def get_latents_2d(self, x):
        B, C, H, W = x.shape
        if C != 3:
            # b h w c -> b c h w
            x = ops.transpose(x, (0, 3, 1, 2))
            # raise ValueError("Expect input shape (b 3 h w), but get {}".format(x.shape))

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
            tokens, _ = self.cond_stage_model.tokenize(c)   # text -> tensor
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

    # def forward(self, x, c, **kwargs):
    #     t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
    #     if self.use_dynamic_rescale:
    #         x = x * extract_into_tensor(self.scale_arr, t, x.shape)
    #     return self.p_losses(x, c, t, **kwargs)

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
    def __init__(self, img_cond_stage_config, image_proj_stage_config, freeze_embedder=True, image_proj_model_trainable=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_proj_model_trainable = image_proj_model_trainable
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self._init_img_ctx_projector(image_proj_stage_config, image_proj_model_trainable)

    def _init_img_ctx_projector(self, config, trainable):
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.set_train(False)
            # self.image_proj_model.eval()
            # self.image_proj_model.train = disabled_train
            for param in self.image_proj_model.get_parameters():
                param.requires_grad = False

    def _init_embedder(self, config, freeze=True):
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.set_train(False)
            # self.embedder.eval()
            # self.embedder.train = disabled_train
            for param in self.embedder.get_parameters():
                param.requires_grad = False

    def shared_step(self, batch, random_uncond, **kwargs):
        x, c, fs = self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True)
        kwargs.update({"fs": fs.long()})
        loss, loss_dict = self(x, c, **kwargs)
        return loss, loss_dict
    
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False, **kwargs):
        ## x: b c t h w
        x = super().get_input(batch, self.first_stage_key)
        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x)
        
        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
                
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = ops.rand(x.size(0))
        else:
            random_num = ops.ones(x.size(0))  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")

        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = ops.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)

        img = x[:,:,cond_frame_index,...]
        img = input_mask * img
        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb)

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
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond] # b c t h w
        cond["c_crossattn"] = [ops.cat([prompt_imb, img_emb], axis=1)] ## concat in the seq_len dim

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            out.append(fs)
        if return_cond_frame:
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)

        return out

    # @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, mask=None, **kwargs):
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()

        z, c, xrec, xc, fs, cond_x = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True,
                                                return_fs=True,
                                                return_cond_frame=True)

        N = xrec.shape[0]
        log["image_condition"] = cond_x
        log["reconst"] = xrec
        xc_with_fs = []
        for idx, content in enumerate(xc):
            xc_with_fs.append(content + '_fs=' + str(fs[idx].item()))
        log["condition"] = xc_with_fs
        kwargs.update({"fs": fs.long()})

        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if 'c_concat' in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = ops.zeros_like(c_emb)
                
                img = ops.zeros_like(xrec[:,:,0]) ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img) ## b l c
                uc_img = self.image_proj_model(img_emb)

                uc = ops.cat([uc_prompt, uc_img], axis=1)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        params = list(self.model.parameters())
        _logger.info(f"@Training [{len(params)}] Full Paramters.")
        # mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            _logger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            # mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)
        
        if self.image_proj_model_trainable:
            _logger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            # mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))   

        if self.learn_logvar:
            _logger.info('Diffusion model optimizing logvar')
            # mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        # optimizer = torch.optim.AdamW(params, lr=lr)
        # if args.optim == "adamw":
        #     optim_cls = AdamWeightDecay
        # elif args.optim == "adamw_re":
        #     from mindcv.optim.adamw import AdamW as AdamW_Refined
        #     optim_cls = AdamW_Refined
        # optimizer = optim_cls(group_params, learning_rate=lr, beta1=betas[0], beta2=betas[1], eps=eps)
        optimizer = AdamWeightDecay(params, learning_rate=lr)

        ## lr scheduler
        if self.use_scheduler:
            _logger.info("Setting up scheduler...")
            # mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer


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


class LatentDiffusion2D(LatentDiffusion):
    """
    LDM for UNet2D
    """

    def get_latents(self, x):
        return self.get_latents_2d(x)

    def reduce_loss(self, loss):
        return self.reduce_loss_2d(loss)
