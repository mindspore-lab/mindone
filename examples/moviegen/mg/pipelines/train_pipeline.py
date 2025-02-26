from typing import Optional

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from ..models import TemporalAutoencoder
from ..schedulers import RFlowLossWrapper
from ..utils.model_utils import no_grad

__all__ = ["DiffusionWithLoss"]


class DiffusionWithLoss(nn.Cell):
    def __init__(
        self,
        network: RFlowLossWrapper,
        tae: Optional[TemporalAutoencoder] = None,
        text_encoder: Optional[nn.Cell] = None,
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
    ):
        super().__init__()

        if not text_emb_cached and text_encoder is None:
            raise ValueError("`text_encoder` must be provided when `text_emb_cached=False`.")
        if not video_emb_cached and tae is None:
            raise ValueError("`TAE` must be provided when `video_emb_cached=False`.")

        self.network = network
        self.tae = tae
        self.text_encoder = text_encoder
        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached

        if self.tae is not None:
            for param in self.tae.trainable_params():
                param.requires_grad = False

        if self.text_encoder is not None:
            for param in self.text_encoder.trainable_params():
                param.requires_grad = False

    def get_condition_embeddings(self, text_tokens: Tensor) -> Tensor:
        if self.text_emb_cached:
            return text_tokens
        with no_grad():
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens))
        return text_emb

    def get_latents(self, video_tokens: Tensor) -> Tensor:
        if self.video_emb_cached:  # (B, T, C, H, W)
            return video_tokens
        with no_grad():  # (B, C, T, H, W)
            video_emb = ops.stop_gradient(self.tae.encode(video_tokens)[0]).to(ms.float32)
            video_emb = (video_emb - self.tae.shift_factor) * self.tae.scale_factor
            video_emb = mint.permute(video_emb, (0, 2, 1, 3, 4))  # FIXME: move inside `Encoder`
        return video_emb

    def set_train(self, mode=True):
        # Set the diffusion model only to train or eval mode
        self.network.set_train(mode)

    def construct(self, video_tokens: Tensor, ul2_tokens: Tensor, byt5_tokens: Tensor) -> Tensor:
        latent_embedding = self.get_latents(video_tokens)
        ul2_emb = self.get_condition_embeddings(ul2_tokens)
        byt5_emb = self.get_condition_embeddings(byt5_tokens)
        # FIXME: add metaclip
        metaclip_emb = mint.ones((byt5_emb.shape[0], 300, 1280), dtype=byt5_emb.dtype)
        return self.network(latent_embedding, ul2_emb, metaclip_emb, byt5_emb)
