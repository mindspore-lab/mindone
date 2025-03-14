from typing import Literal, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, ops

from ..models import LlamaModel, TemporalAutoencoder
from ..schedulers.rectified_flow import RFLOW

__all__ = ["InferPipeline"]


class InferPipeline:
    """An Inference pipeline for Movie Gen.

    Args:
        model (LlamaModel): A noise prediction model to denoise the encoded image latents.
        tae (TemporalAutoencoder, optional): Temporal Auto-Encoder (TAE) Model to encode and decode images or videos to
                                             and from latent representations.
        scale_factor (float): scale_factor for TAE.
        guidance_scale (float): A higher guidance scale value for noise rescale.
        num_sampling_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        model: LlamaModel,
        tae: Optional[TemporalAutoencoder] = None,
        latent_size: Tuple[int, int, int] = (1, 64, 64),
        guidance_scale: float = 1.0,
        num_sampling_steps: int = 50,
        sample_method: Literal["linear", "linear-quadratic"] = "linear",
        micro_batch_size: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.tae = tae
        self.latent_size = latent_size
        self.micro_batch_size = micro_batch_size
        self.guidance_rescale = guidance_scale
        self.use_cfg = guidance_scale > 1.0
        self.rflow = RFLOW(num_sampling_steps, sample_method=sample_method)

    def tae_decode_video(self, x, num_frames=None):
        """
        Args:
            x: (b t c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        x = mint.permute(x, (0, 2, 1, 3, 4))  # FIXME: remove this redundancy
        x = x / self.tae.scale_factor + self.tae.shift_factor
        y = self.tae.decode(x, target_num_frames=num_frames)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)
        # (b 3 t h w) -> (b t h w 3)
        y = mint.permute(y, (0, 2, 3, 4, 1))
        return y

    def __call__(
        self, ul2_emb: Tensor, metaclip_emb: Tensor, byt5_emb: Tensor, num_frames: int = None
    ) -> Tuple[Union[Tensor, None], Tensor]:
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z = ms.Tensor(
            np.random.randn(
                ul2_emb.shape[0], self.latent_size[0], self.model.in_channels, self.latent_size[1], self.latent_size[2]
            ).astype(np.float32),
            dtype=self.model.dtype,
        )
        if self.use_cfg:
            raise NotImplementedError("Condition-free guidance is not supported yet.")

        latents = self.rflow(
            self.model,
            z,
            ul2_emb.to(self.model.dtype),
            metaclip_emb.to(self.model.dtype),
            byt5_emb.to(self.model.dtype),
        ).to(ms.float32)

        if self.tae is not None:
            # latents: (b t c h w)
            # out: (b T H W C)
            images = self.tae_decode_video(latents, num_frames=num_frames)
            return images, latents
        else:
            return None, latents
