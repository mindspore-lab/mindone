import logging

from opensora.acceleration.communications import prepare_parallel_data
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.models.diffusion.diffusion import SpacedDiffusion_T as SpacedDiffusion
from opensora.models.diffusion.diffusion.diffusion_utils import (
    _extract_into_tensor,
    discretized_gaussian_log_likelihood,
    mean_flat,
    normal_kl,
)

import mindspore as ms
from mindspore import nn, ops

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


class DiffusionWithLoss(nn.Cell):
    """An training pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        diffusion: (object): A class for Gaussian Diffusion.
        text_encoder (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
    """

    def __init__(
        self,
        network: nn.Cell,
        diffusion: SpacedDiffusion,
        vae: nn.Cell = None,
        text_encoder: nn.Cell = None,
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
        use_image_num: int = 0,
        dtype=ms.float32,
    ):
        super().__init__()
        # TODO: is set_grad() necessary?
        self.network = network.set_grad()
        self.vae = vae
        self.diffusion = diffusion

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

        # FIXME: bug when sp_size=2
        # self.broadcast_t = None if not get_sequence_parallel_state() \
        #     else ops.Broadcast(root_rank=int(hccl_info.group_id * hccl_info.world_size), group=hccl_info.group)
        self.reduce_t = None if not get_sequence_parallel_state() else ops.AllReduce(group=hccl_info.group)
        self.sp_size = 1 if not get_sequence_parallel_state() else hccl_info.world_size

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
                z = ops.cat([videos, images], axis=2)  # b c 16+4, h, w
        else:
            raise ValueError("Incorrect Dimensions of x")
        return z

    def construct(
        self,
        x: ms.Tensor,
        text_tokens: ms.Tensor,
        encoder_attention_mask: ms.Tensor = None,
        attention_mask: ms.Tensor = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape (b c f+num_img h w)
            text_tokens: text tokens padded to fixed shape [bs, L]
            labels: the class labels

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f+num_img h w)
        """
        # 1. get image/video latents z using vae
        x = x.to(self.dtype)
        if not self.video_emb_cached:
            x = ops.stop_gradient(self.get_latents(x))

        # 2. get conditions
        if not self.text_emb_cached:
            text_embed = ops.stop_gradient(self.get_condition_embeddings(text_tokens, encoder_attention_mask))
        else:
            text_embed = text_tokens

        loss = self.compute_loss(x, text_embed, encoder_attention_mask, attention_mask)

        return loss

    def apply_model(self, *args, **kwargs):
        return self.network(*args, **kwargs)

    def _cal_vb(self, model_output, model_var_values, x, x_t, t):
        # make sure all inputs are fp32 for accuracy
        model_output = model_output.to(ms.float32)
        model_var_values = model_var_values.to(ms.float32)
        x = x.to(ms.float32)
        x_t = x_t.to(ms.float32)

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
        kl = mean_flat(kl) / ms.numpy.log(2.0)  # TODO:

        # print('D--: kl input type ', t.dtype, x.dtype,  model_mean.dtype, kl.dtype)

        # NOTE: make sure it's computed in fp32 since this func contains many exp.
        decoder_nll = -discretized_gaussian_log_likelihood(x, means=model_mean, log_scales=0.5 * model_log_variance)
        decoder_nll = mean_flat(decoder_nll) / ms.numpy.log(2.0)

        # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        flag = (t == 0).astype(kl.dtype)
        vb = flag * decoder_nll + (1.0 - flag) * kl

        return vb

    def compute_loss(self, x, text_embed, encoder_attention_mask, attention_mask=None):
        use_image_num = self.use_image_num

        if get_sequence_parallel_state():
            x, text_embed, attention_mask, encoder_attention_mask, use_image_num = prepare_parallel_data(
                x, text_embed, attention_mask, encoder_attention_mask, use_image_num
            )

        t = ops.randint(0, self.diffusion.num_timesteps, (x.shape[0],), dtype=ms.int32)

        if get_sequence_parallel_state():
            t = self.reduce_t(t) // self.sp_size

        noise = ops.randn_like(x)
        x_t = self.diffusion.q_sample(x, t, noise=noise)

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        # text_embed = ops.expand_dims(text_embed, axis=1)
        model_output = self.apply_model(
            x_t,
            t,
            encoder_hidden_states=text_embed,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            use_image_num=use_image_num,
        )

        # (b c t h w),
        B, C, F = x_t.shape[:3]
        assert (
            model_output.shape == (B, C * 2, F) + x_t.shape[3:]
        ), f"model_output shape {model_output.shape} and x_t shape {x_t.shape} mismatch!"
        model_output, model_var_values = ops.split(model_output, C, axis=1)

        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        vb = self._cal_vb(ops.stop_gradient(model_output), model_var_values, x, x_t, t)

        loss = mean_flat((noise - model_output) ** 2) + vb
        loss = loss.mean()
        return loss
