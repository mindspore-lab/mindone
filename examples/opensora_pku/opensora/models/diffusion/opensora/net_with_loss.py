import logging
import math

from opensora.acceleration.communications import prepare_parallel_data
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.utils.ms_utils import no_grad
from opensora.utils.utils import explicit_uniform_sampling, get_sigmas

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers.training_utils import compute_snr

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    """
    Compute the density for sampling the timesteps when doing SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "logit_normal":
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = mint.normal(mean=logit_mean, std=logit_std, size=(batch_size,))
        u = mint.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = ops.stop_gradient(mint.rand(size=(batch_size,)))
        u = 1 - u - mode_scale * (mint.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = ops.stop_gradient(mint.rand(size=(batch_size,)))
    return u


def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    """
    Computes loss weighting scheme for SD3 training.

    Courtesy: This was contributed by Rafie Walker in https://github.com/huggingface/diffusers/pull/8528.

    SD3 paper reference: https://arxiv.org/abs/2403.03206v1.
    """
    if weighting_scheme == "sigma_sqrt":
        weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap":
        bot = 1 - 2 * sigmas + 2 * sigmas**2
        weighting = 2 / (math.pi * bot)
    else:
        weighting = mint.ones_like(sigmas)
    return weighting


class DiffusionWithLoss(nn.Cell):
    """An training pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        noise_scheduler: (object): A class for noise scheduler, such as DDPM scheduler
        text_encoder / text_encoder_2 (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
        rf_scheduler (bool): whether to apply rectified flow scheduler.
    """

    def __init__(
        self,
        network: nn.Cell,
        noise_scheduler,
        vae: nn.Cell = None,
        text_encoder: nn.Cell = None,
        text_encoder_2: nn.Cell = None,  # not to use yet
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
        use_image_num: int = 0,
        dtype=ms.float32,
        noise_offset: float = 0.0,
        rf_scheduler: bool = False,
        snr_gamma=None,
        rank_id: int = 0,
        device_num: int = 1,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        weighting_scheme: str = "logit_normal",
        mode_scale: float = 1.29,
    ):
        super().__init__()
        # TODO: is set_grad() necessary?
        self.network = network.set_grad()
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.prediction_type = self.noise_scheduler.config.prediction_type
        self.num_train_timesteps = self.noise_scheduler.config.num_train_timesteps
        self.noise_offset = noise_offset
        self.rf_scheduler = rf_scheduler
        self.rank_id = rank_id
        self.device_num = device_num
        self.snr_gamma = snr_gamma
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.weighting_scheme = weighting_scheme
        self.mode_scale = mode_scale

        self.text_encoder = text_encoder
        self.dtype = dtype

        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached

        if self.text_emb_cached:
            self.text_encoder = None
            logger.info("Train with text embedding inputs")
        else:
            self.text_encoder = text_encoder

        self.use_image_num = use_image_num
        self.reduce_t = None if not get_sequence_parallel_state() else ops.AllReduce(group=hccl_info.group)
        self.sp_size = 1 if not get_sequence_parallel_state() else hccl_info.world_size
        self.all_gather = None if not get_sequence_parallel_state() else ops.AllGather(group=hccl_info.group)

    def get_condition_embeddings(self, text_tokens, encoder_attention_mask):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        # use for loop to avoid OOM?
        B, frame, L = text_tokens.shape  # B T+num_images L = b 1+4, L
        text_emb = []
        for i in range(frame):
            t = self.text_encoder(text_tokens[:, i], encoder_attention_mask[:, i])
            text_emb.append(t)
        text_emb = ops.stack(text_emb, axis=1)
        return text_emb

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        return image_latents

    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        # b, c, f, h, w = x.shape
        y = self.vae.decode(x)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        return y  # b c f h w

    def get_latents(self, x):
        if x.dim() == 5:
            B, C, F, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b 3 f h w), but get {}".format(x.shape))
            if self.use_image_num == 0:
                z = self.vae_encode(x)  # (b, c, f, h, w)
            else:
                videos, images = x[:, :, : -self.use_image_num], x[:, :, -self.use_image_num :]
                videos = self.vae_encode(videos)  # (b, c, f, h, w)
                # (b, c, f, h, w) -> (b, f, c, h, w) -> (b*f, c, h, w) -> (b*f, c, 1, h, w)
                images = images.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W).unsqueeze(2)
                images = self.vae_encode(images)  # (b*f, c, 1, h, w)
                # (b*f, c, 1, h, w) -> (b*f, c, h, w) -> (b, f, c, h, w) -> (b, c, f, h, w)
                _, c, _, h, w = images.shape
                images = images.squeeze(2).reshape(B, self.use_image_num, c, h, w).permute(0, 2, 1, 3, 4)
                z = mint.cat([videos, images], dim=2)  # b c 16+4, h, w
        else:
            raise ValueError("Incorrect Dimensions of x")
        return z

    def construct(
        self,
        x: ms.Tensor,
        attention_mask: ms.Tensor,
        text_tokens: ms.Tensor,
        encoder_attention_mask: ms.Tensor = None,
    ):  # TODO: in the future add 2nd text encoder and tokens
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape (b c f+num_img h w)
            attention_mask: the mask for latent features of shape (b t' h' w'), where t' h' w' are the shape of latent features after vae's encoding.
            text_tokens: text tokens padded to fixed shape (B F L) or text embedding of shape (B F L D) if using text embedding cache
            encoder_attention_mask: the mask for text tokens/embeddings of a fixed shape (B F L)

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f+num_img h w)
        """
        # 1. get image/video latents z using vae
        x = x.to(self.dtype)
        with no_grad():
            if not self.video_emb_cached:
                x = ops.stop_gradient(self.get_latents(x))

            # 2. get conditions
            if not self.text_emb_cached:
                text_embed = ops.stop_gradient(self.get_condition_embeddings(text_tokens, encoder_attention_mask))
            else:
                text_embed = text_tokens
        loss = self.compute_loss(x, attention_mask, text_embed, encoder_attention_mask)

        return loss

    def apply_model(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def compute_loss(self, x, attention_mask, text_embed, encoder_attention_mask):
        use_image_num = self.use_image_num
        noise = ops.stop_gradient(mint.randn_like(x))
        bsz = x.shape[0]
        if not self.rf_scheduler:
            if self.noise_offset:
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += self.noise_offset * ops.stop_gradient(mint.randn((bsz, x.shape[1], 1, 1, 1), dtype=x.dtype))
        current_step_frame = x.shape[2]
        if get_sequence_parallel_state() and current_step_frame > 1:
            x = self.all_gather(x[None])[0]
            (
                x,
                noise,
                text_embed,
                attention_mask,
                encoder_attention_mask,
                use_image_num,
            ) = prepare_parallel_data(x, noise, text_embed, attention_mask, encoder_attention_mask, use_image_num)
        if not self.rf_scheduler:
            # sample a random timestep for each image without bias
            t = explicit_uniform_sampling(
                T=self.num_train_timesteps,
                n=self.device_num,
                rank=self.rank_id,
                bsz=bsz,
            )
            # t = ops.randint(0, self.num_train_timesteps, (x.shape[0],), dtype=ms.int32)
            if get_sequence_parallel_state():
                t = self.reduce_t(t) % self.num_train_timesteps
            x_t = self.noise_scheduler.add_noise(x, noise, t)
        else:
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.weighting_scheme,
                batch_size=bsz,
                logit_mean=self.logit_mean,
                logit_std=self.logit_std,
                mode_scale=self.mode_scale,
            )
            indices = u * self.num_train_timesteps
            t = self.noise_scheduler.timesteps[indices]

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(self.noise_scheduler, t, n_dim=x.ndim, dtype=x.dtype)
            x_t = (1.0 - sigmas) * x + sigmas * noise

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        # text_embed = ops.expand_dims(text_embed, axis=1)
        model_pred = self.apply_model(
            x_t,
            t,
            encoder_hidden_states=text_embed,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            use_image_num=use_image_num,
        )
        if not self.rf_scheduler:
            if self.prediction_type == "epsilon":
                target = noise
            elif self.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(x, noise, t)
            elif self.prediction_type == "sample":
                # We set the target to latents here, but the model_pred will return the noise sample prediction.
                target = x
                # We will have to subtract the noise residual from the prediction to get the target sample.
                model_pred = model_pred - noise
            else:
                raise ValueError(f"Unknown prediction type {self.prediction_type}")
            # comment it to avoid graph syntax error
            # if attention_mask is not None and (attention_mask.bool()).all():
            #     attention_mask = None
            if get_sequence_parallel_state():
                assert (attention_mask.bool()).all()
                # assert attention_mask is None
                attention_mask = None
            # (b c t h w),
            bsz, c, _, _, _ = model_pred.shape
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).float().repeat_interleave(c, dim=1)  # b t h w -> b c t h w
                attention_mask = attention_mask.reshape(bsz, -1)

            if self.snr_gamma is None:
                # model_pred: b c t h w, attention_mask: b t h w
                loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.reshape(bsz, -1)
                if attention_mask is not None:
                    loss = (loss * attention_mask).sum() / attention_mask.sum()  # mean loss on unpad patches
                else:
                    loss = loss.mean()
            else:
                # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                # This is discussed in Section 4.2 of the same paper.
                snr = compute_snr(self.noise_scheduler, t)
                mse_loss_weights = ops.stack([snr, self.snr_gamma * ops.ones_like(t)], axis=1).min(axis=1)[0]
                if self.prediction_type == "epsilon":
                    mse_loss_weights = mse_loss_weights / snr
                elif self.prediction_type == "v_prediction":
                    mse_loss_weights = mse_loss_weights / (snr + 1)
                loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.reshape(bsz, -1)
                mse_loss_weights = mse_loss_weights.reshape(bsz, 1)
                if attention_mask is not None:
                    loss = (
                        loss * attention_mask * mse_loss_weights
                    ).sum() / attention_mask.sum()  # mean loss on unpad patches
                else:
                    loss = (loss * mse_loss_weights).mean()
        else:
            if mint.all(attention_mask.bool()):
                attention_mask = None

            b, c, _, _, _ = model_pred.shape
            if attention_mask is not None:
                attention_mask = attention_mask.unsqueeze(1).float().repeat_interleave(c, dim=1)  # b t h w -> b c t h w
                attention_mask = attention_mask.reshape(b, -1)

            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(weighting_scheme=self.weighting_scheme, sigmas=sigmas)

            # flow matching loss
            target = noise - x

            # Compute regular loss.
            loss_mse = (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1)
            if attention_mask is not None:
                loss = (loss_mse * attention_mask).sum() / attention_mask.sum()
            else:
                loss = loss_mse.mean()

        return loss


class DiffusionWithLossEval(DiffusionWithLoss):
    def construct(
        self,
        x: ms.Tensor,
        attention_mask: ms.Tensor,
        text_tokens: ms.Tensor,
        encoder_attention_mask: ms.Tensor = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape (b c f+num_img h w)
            attention_mask: the mask for latent features of shape (b t' h' w'), where t' h' w' are the shape of latent features after vae's encoding.
            text_tokens: text tokens padded to fixed shape (B F L) or text embedding of shape (B F L D) if using text embedding cache
            encoder_attention_mask: the mask for text tokens/embeddings of a fixed shape (B F L)

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f+num_img h w)
        """
        # 1. get image/video latents z using vae
        x = x.to(self.dtype)
        with no_grad():
            if not self.video_emb_cached:
                x = ops.stop_gradient(self.get_latents(x))

            # 2. get conditions
            if not self.text_emb_cached:
                text_embed = ops.stop_gradient(self.get_condition_embeddings(text_tokens, encoder_attention_mask))
            else:
                text_embed = text_tokens
        loss, model_pred, target = self.compute_loss(x, attention_mask, text_embed, encoder_attention_mask)

        return loss, model_pred, target

    def compute_loss(self, x, attention_mask, text_embed, encoder_attention_mask):
        use_image_num = self.use_image_num
        noise = ops.stop_gradient(mint.randn_like(x))
        bsz = x.shape[0]
        if self.noise_offset:
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * ops.stop_gradient(mint.randn((bsz, x.shape[1], 1, 1, 1), dtype=x.dtype))
        current_step_frame = x.shape[2]
        if get_sequence_parallel_state() and current_step_frame > 1:
            x = self.all_gather(x[None])[0]
            (
                x,
                noise,
                text_embed,
                attention_mask,
                encoder_attention_mask,
                use_image_num,
            ) = prepare_parallel_data(x, noise, text_embed, attention_mask, encoder_attention_mask, use_image_num)

        t = ops.stop_gradient(mint.randint(0, self.num_train_timesteps, (x.shape[0],), dtype=ms.int32))
        if get_sequence_parallel_state():
            t = self.reduce_t(t) % self.num_train_timesteps
        x_t = self.noise_scheduler.add_noise(x, noise, t)

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        # text_embed = ops.expand_dims(text_embed, axis=1)
        model_pred = self.apply_model(
            x_t,
            t,
            encoder_hidden_states=text_embed,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            use_image_num=use_image_num,
        )

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(x, noise, t)
        elif self.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = x
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            raise ValueError(f"Unknown prediction type {self.prediction_type}")
        # comment it to avoid graph syntax error
        # if attention_mask is not None and (attention_mask.bool()).all():
        #     attention_mask = None
        if get_sequence_parallel_state():
            assert (attention_mask.bool()).all()
            # assert attention_mask is None
            attention_mask = None
        # (b c t h w),
        bsz, c, _, _, _ = model_pred.shape
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).float().repeat_interleave(c, dim=1)  # b t h w -> b c t h w
            attention_mask = attention_mask.reshape(bsz, -1)

        if self.snr_gamma is None:
            # model_pred: b c t h w, attention_mask: b t h w
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.reshape(bsz, -1)
            if attention_mask is not None:
                loss = (loss * attention_mask).sum() / attention_mask.sum()  # mean loss on unpad patches
            else:
                loss = loss.mean()
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, t)
            mse_loss_weights = ops.stack([snr, self.snr_gamma * ops.ones_like(t)], axis=1).min(axis=1)[0]
            if self.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            loss = ops.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.reshape(bsz, -1)
            mse_loss_weights = mse_loss_weights.reshape(bsz, 1)
            if attention_mask is not None:
                loss = (
                    loss * attention_mask * mse_loss_weights
                ).sum() / attention_mask.sum()  # mean loss on unpad patches
            else:
                loss = (loss * mse_loss_weights).mean()
        return loss, model_pred, target
