import logging
from typing import Optional

from hyvideo.utils.ms_utils import no_grad

import mindspore as ms
from mindspore import Tensor, nn, ops

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


class DiffusionWithLoss(nn.Cell):
    """An training pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        noise_scheduler: (object): A class for noise scheduler, such as DDPM scheduler
        text_encoder / text_encoder_2 (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: nn.Cell = None,
        text_encoder: nn.Cell = None,
        text_encoder_2: nn.Cell = None,  # not to use yet
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
        embedded_guidance_scale: float = 6.0,
    ):
        super().__init__()
        # TODO: is set_grad() necessary?
        self.network = network.set_grad()
        self.vae = vae
        self.embedded_guidance_scale = embedded_guidance_scale

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached
        self.vae_scaling_factor = self.vae.config.scaling_factor

        if self.text_emb_cached:
            self.text_encoder = None
            self.text_encoder_2 = None
            logger.info("Train with text embedding inputs")

    def get_condition_embeddings(self, text_tokens, encoder_attention_mask, index=0):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        # use for loop to avoid OOM?
        B, frame, L = text_tokens.shape  # B T+num_images L = b 1+4, L
        text_emb = []
        assert index in [0, 1], "index should be 0 or 1"
        text_encoder = self.text_encoder if index == 0 else self.text_encoder_2
        for i in range(frame):
            t = text_encoder(text_tokens[:, i], encoder_attention_mask[:, i])
            text_emb.append(t)
        text_emb = ops.stack(text_emb, axis=1)
        return text_emb

    def get_latents(self, video_tokens: Tensor) -> Tensor:
        if self.video_emb_cached:  # (B, C, T, H, W)
            return video_tokens
        with no_grad():  # (B, C, T, H, W)
            video_emb = ops.stop_gradient(self.vae.encode(video_tokens)).to(ms.float32)
            video_emb = video_emb * self.vae_scaling_factor
        return video_emb

    def construct(
        self,
        x: Tensor,
        text_tokens: Tensor,
        encoder_attention_mask: Tensor = None,
        text_tokens_2: Tensor = None,
        freqs_cos: Optional[ms.Tensor] = None,
        freqs_sin: Optional[ms.Tensor] = None,
        encoder_attention_mask_2: Tensor = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: (B, C, T, H, W).
            text_tokens: (B, L, D)
            encoder_attention_mask: (B, L)
            text_tokens_2: (B, D')
            freqs_cos: (S attn_head_dim), S - seq len of the patchified video latent (T * H //2 * W//2)
            freqs_sin: (S attn_head_dim)
            encoder_attention_mask_2: (B, L')
        Returns:
            loss: (B,)
        """
        # 1. get image/video latents z using vae
        with no_grad():
            if not self.video_emb_cached:
                x = ops.stop_gradient(self.get_latents(x))

            # 2. get conditions
            if not self.text_emb_cached:
                text_embed = ops.stop_gradient(
                    self.get_condition_embeddings(text_tokens, encoder_attention_mask, index=0)
                )
                if text_tokens_2 is not None:
                    text_embed_2 = ops.stop_gradient(
                        self.get_condition_embeddings(text_tokens_2, encoder_attention_mask_2, index=1)
                    )
                else:
                    text_embed_2 = None
            else:
                text_embed = text_tokens
                if text_tokens_2 is not None:
                    text_embed_2 = text_tokens_2
                else:
                    text_embed_2 = None
        loss = self.compute_loss(x, text_embed, encoder_attention_mask, text_embed_2, freqs_cos, freqs_sin)
        return loss

    def compute_loss(
        self,
        x,
        text_embed,
        encoder_attention_mask,
        text_embed_2,
        freqs_cos,
        freqs_sin,
    ):
        bsz = x.shape[0]
        guidance_expand = (
            Tensor(
                [self.embedded_guidance_scale] * bsz,
                dtype=ms.float32,
            ).to(x.dtype)
            * 1000.0
            if self.embedded_guidance_scale is not None
            else None
        )
        loss = self.network(
            x,
            text_states=text_embed,
            text_mask=encoder_attention_mask,
            text_states_2=text_embed_2,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
            guidance=guidance_expand,
        )
        return loss
