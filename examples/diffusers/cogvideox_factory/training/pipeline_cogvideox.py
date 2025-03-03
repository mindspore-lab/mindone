from typing import Union
from transformers import T5Tokenizer
from mindone.diffusers import CogVideoXPipeline
from mindone.diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from mindone.diffusers.video_processor import VideoProcessor
from mindone.transformers import T5EncoderModel
from models import CogVideoXEncoder3D_SP, AutoencoderKLCogVideoX_SP


class CogVideoXSPPipeline(CogVideoXPipeline):
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX_SP,
        transformer: CogVideoXEncoder3D_SP,
        scheduler: Union[CogVideoXDDIMScheduler, CogVideoXDPMScheduler],
    ):
        super().__init__()
        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if hasattr(self, "vae") and self.vae is not None else 4
        )
        self.vae_scaling_factor_image = (
            self.vae.config.scaling_factor if hasattr(self, "vae") and self.vae is not None else 0.7
        )

        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
