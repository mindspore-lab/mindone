import logging
from typing import Optional, Tuple

import mindspore as ms
from mindspore import Tensor, nn, ops

from ..schedulers.iddpm import SpacedDiffusion
from ..schedulers.iddpm.diffusion_utils import (
    _extract_into_tensor,
    discretized_gaussian_log_likelihood,
    mean_flat,
    normal_kl,
)

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


class DiffusionWithLoss(nn.Cell):
    """An training pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        diffusion: (object): A class for Gaussian Diffusion.
        scale_factor (float): scale_factor for vae.
        condition (str): The type of conditions of model in [None, 'text', 'class'].
            If it is None, model is a un-conditional video generator.
            If it is 'text', model accepts text embeddings (B, T, N) as conditions, and generates videos.
            If it is 'class', model accepts class labels (B, ) as conditions, and generates videos.
        text_encoder (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        cond_stage_trainable (bool): whether to train the text encoder.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
    """

    def __init__(
        self,
        network: nn.Cell,
        diffusion: SpacedDiffusion,
        vae: nn.Cell = None,
        text_encoder: nn.Cell = None,
        scale_factor: float = 0.18215,
        cond_stage_trainable: bool = False,
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
        micro_batch_size: int = None,
    ):
        super().__init__()
        # TODO: is set_grad() necessary?
        self.network = network.set_grad()
        self.vae = vae
        self.diffusion = diffusion
        self.text_encoder = text_encoder

        self.scale_factor = scale_factor
        self.cond_stage_trainable = cond_stage_trainable

        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached
        self.micro_batch_size = micro_batch_size

        if self.text_emb_cached:
            self.text_encoder = None
            logger.info("Train with text embedding inputs")
        else:
            raise NotImplementedError

        if self.cond_stage_trainable and self.text_encoder:
            self.text_encoder.set_train(True)
            self.text_encoder.set_grad(True)

    def get_condition_embeddings(self, text_tokens, **kwargs):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        if self.cond_stage_trainable:
            text_emb = self.text_encoder(text_tokens, **kwargs)
        else:
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens, **kwargs))

        return text_emb

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents

    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x):
        """
        Args:
            x: (b f c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        b, f, c, h, w = x.shape
        y = []
        for x_sample in x:
            y.append(self.vae_decode(x_sample))

        y = ops.stack(y, axis=0)

        return y

    def get_latents(self, x):
        if x.dim() == 5:
            # "b f c h w -> (b f) c h w"
            B, F, C, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
            x = ops.reshape(x, (-1, C, H, W))

            if self.micro_batch_size is not None:
                # split into smaller frames to reduce memory cost
                x = ops.split(x, self.micro_batch_size, axis=0)
                z_clips = []
                for clip in x:
                    z_clips.append(ops.stop_gradient(self.vae_encode(clip)))
                z = ops.cat(z_clips, axis=0)
            else:
                z = ops.stop_gradient(self.vae_encode(x))

            # (b*f c h w) -> (b f c h w)
            z = ops.reshape(z, (B, F, z.shape[1], z.shape[2], z.shape[3]))

            # (b f c h w) -> (b c f h w)
            z = ops.transpose(z, (0, 2, 1, 3, 4))
        elif x.dim() == 4:
            B, C, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
            z = ops.stop_gradient(self.vae_encode(x))
        else:
            raise ValueError("Incorrect Dimensions of x")
        return z

    def construct(
        self,
        x: Tensor,
        text_tokens: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        ar: Optional[Tensor] = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape [bs, F, 3, 256, 256]
            text_tokens: text tokens padded to fixed shape [bs, 77]
            labels: the class labels

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f h w)
                unet2d input/output shape: (b c h w)
        """
        # 1. get image/video latents z using vae
        if not self.video_emb_cached:
            x = self.get_latents(x)
        else:
            # (b f c h w) -> (b c f h w)
            x = ops.transpose(x, (0, 2, 1, 3, 4))

        # 2. get conditions
        if not self.text_emb_cached:
            text_embed = self.get_condition_embeddings(text_tokens)
        else:
            text_embed = text_tokens  # dataset retunrs text embeddings instead of text tokens
        loss = self.compute_loss(x, text_embed, mask, frames_mask, num_frames, height, width, fps, ar)

        return loss

    def apply_model(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def _cal_vb(
        self,
        model_output: Tensor,
        model_var_values: Tensor,
        x: Tensor,
        x_t: Tensor,
        t: Tensor,
        frames_mask: Optional[Tensor] = None,
        patch_mask: Optional[Tensor] = None,
    ):
        # make sure all inputs are fp32 for accuracy
        model_output = model_output.to(ms.float32)
        model_var_values = model_var_values.to(ms.float32)

        true_mean, _, true_log_variance_clipped = self.diffusion.q_posterior_mean_variance(x_start=x, x_t=x_t, t=t)
        # p_mean_variance(model=lambda *_: frozen_out, x_t, t, clip_denoised=False) begin
        min_log = _extract_into_tensor(self.diffusion.posterior_log_variance_clipped, t, x_t.shape)
        max_log = _extract_into_tensor(self.diffusion.log_betas, t, x_t.shape)
        # The model_var_values is [-1, 1] for [min_var, max_var].
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        pred_xstart = self.diffusion.predict_xstart_from_eps(x_t=x_t, t=t, eps=model_output)
        model_mean, _, _ = self.diffusion.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        # assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x_t.shape
        # p_mean_variance end
        kl = normal_kl(true_mean, true_log_variance_clipped, model_mean, model_log_variance)
        kl = mean_flat(kl, frames_mask=frames_mask, patch_mask=patch_mask) / ms.numpy.log(2.0)  # TODO:

        # NOTE: make sure it's computed in fp32 since this func contains many exp.
        decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll, frames_mask=frames_mask, patch_mask=patch_mask) / ms.numpy.log(2.0)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        vb = ops.where(t == 0, decoder_nll.to(kl.dtype), kl)

        return vb

    def compute_loss(
        self,
        x: Tensor,
        text_embed: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        ar: Optional[Tensor] = None,
    ):
        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],))
        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x.to(ms.float32), t, noise=noise)

        if frames_mask is not None:
            t0 = ops.zeros_like(t)
            x_t0 = self.diffusion.q_sample(x, t0, noise=noise)
            x_t = ops.where(frames_mask[:, None, :, None, None], x_t, x_t0)

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        text_embed = ops.expand_dims(text_embed, axis=1)
        model_output = self.apply_model(
            x_t,
            t,
            text_embed,
            mask,
            frames_mask=frames_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            ar=ar,
            fps=fps,
        )

        # (b c t h w),
        B, C, F = x_t.shape[:3]
        assert model_output.shape == (B, C * 2, F) + x_t.shape[3:]
        model_output, model_var_values = ops.split(model_output, C, axis=1)

        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        vb = self._cal_vb(ops.stop_gradient(model_output), model_var_values, x, x_t, t, frames_mask)

        loss = mean_flat((noise - model_output) ** 2, frames_mask) + vb
        loss = loss.mean()
        return loss


class DiffusionWithLossFiTLike(DiffusionWithLoss):
    def __init__(
        self,
        *args,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        max_image_size: int = 512,
        vae_downsample_rate: float = 8.0,
        in_channels: int = 4,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.p = patch_size
        self.c = in_channels

        max_size = int(max_image_size / vae_downsample_rate)
        self.nh = max_size // self.p[1]
        self.nw = max_size // self.p[2]

        if not self.video_emb_cached:
            raise ValueError("Video embedding caching must be provided.")

    def construct(
        self,
        x: Tensor,
        text_tokens: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        ar: Optional[Tensor] = None,
        spatial_pos: Optional[Tensor] = None,
        spatial_mask: Optional[Tensor] = None,
        temporal_pos: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape [bs, F, 3, 256, 256]
            text_tokens: text tokens padded to fixed shape [bs, 77]
            labels: the class labels

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f h w)
                unet2d input/output shape: (b c h w)
        """

        # get conditions
        if not self.text_emb_cached:
            text_embed = self.get_condition_embeddings(text_tokens)
        else:
            text_embed = text_tokens  # dataset retunrs text embeddings instead of text tokens
        loss = self.compute_loss(
            x,
            text_embed,
            mask,
            frames_mask,
            num_frames,
            height,
            width,
            fps,
            ar,
            spatial_pos,
            spatial_mask,
            temporal_pos,
            temporal_mask,
        )

        return loss

    def compute_loss(
        self,
        x: Tensor,
        text_embed: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        ar: Optional[Tensor] = None,
        spatial_pos: Optional[Tensor] = None,
        spatial_mask: Optional[Tensor] = None,
        temporal_pos: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
    ):
        D = x.shape[3]
        # convert x to 5-dim first for q_sample, prevent potential bug
        x = self.unpatchify(x)  # b f t d -> b c f h w
        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],))
        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x.to(ms.float32), t, noise=noise)

        if frames_mask is not None:
            t0 = ops.zeros_like(t)
            x_t0 = self.diffusion.q_sample(x.to(ms.float32), t0, noise=noise)
            x_t = ops.where(frames_mask[:, None, :, None, None], x_t, x_t0)

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        text_embed = ops.expand_dims(text_embed, axis=1)
        model_output = self.apply_model(
            x_t,
            t,
            text_embed,
            mask,
            frames_mask=frames_mask,
            num_frames=num_frames,
            height=height,
            width=width,
            ar=ar,
            fps=fps,
            spatial_pos=spatial_pos,
            spatial_mask=spatial_mask,
            temporal_pos=temporal_pos,
            temporal_mask=temporal_mask,
        )

        # (b c t h w),
        B, C, F = x_t.shape[:3]
        assert model_output.shape == (B, C * 2, F) + x_t.shape[3:]
        model_output, model_var_values = ops.split(model_output, C, axis=1)

        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        patch_mask = temporal_mask[:, :, None, None] * spatial_mask[:, None, :, None]
        patch_mask = self.unpatchify(ops.tile(patch_mask, (1, 1, 1, D)))  # b c t h w
        vb = self._cal_vb(
            ops.stop_gradient(model_output),
            model_var_values,
            x,
            x_t,
            t,
            frames_mask=frames_mask,
            patch_mask=patch_mask,
        )

        loss = mean_flat((noise - model_output) ** 2, frames_mask=frames_mask, patch_mask=patch_mask) + vb
        loss = loss.mean()
        return loss

    def unpatchify(self, x: Tensor):
        n, f, _, _ = x.shape
        x = ops.reshape(x, (n, f, self.nh, self.nw, self.c, self.p[1], self.p[2]))
        x = ops.transpose(x, (0, 4, 1, 2, 5, 3, 6))
        x = ops.reshape(x, (n, self.c, f, self.nh * self.p[1], self.nw * self.p[2]))
        return x
