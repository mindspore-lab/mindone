import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
from lvdm.models.ddpm3d import LatentDiffusion
from scheduler.t2v_turbo_scheduler import T2VTurboScheduler

import mindspore as ms
from mindspore import mint, ops

from mindone.diffusers import DiffusionPipeline
from mindone.diffusers.utils.mindspore_utils import randn_tensor

logger = logging.getLogger(__name__)


class T2VTurboVC2Pipeline(DiffusionPipeline):
    def __init__(
        self,
        pretrained_t2v: LatentDiffusion,
        scheduler: T2VTurboScheduler,
        model_config: Dict[str, Any] = None,
    ):
        super().__init__()

        self.register_modules(
            pretrained_t2v=pretrained_t2v,
            scheduler=scheduler,
        )
        self.vae = pretrained_t2v.first_stage_model
        self.unet = pretrained_t2v.model.diffusion_model
        self.text_encoder = pretrained_t2v.cond_stage_model

        self.model_config = model_config
        self.vae_scale_factor = 8

    def _encode_prompt(
        self,
        prompt,
        num_videos_per_prompt,
        prompt_embeds: None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`Mindspore.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        if prompt_embeds is None:
            prompt_embeds = self.text_encoder(prompt)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # Don't need to get uncond prompt embedding because of LCM Guided Distillation
        return prompt_embeds

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        frames,
        height,
        width,
        dtype,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            frames,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, dtype=dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def get_w_embedding(self, w, embedding_dim=512, dtype=ms.float32):
        """
        see https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298
        Args:
        timesteps: Mindspore.Tensor: generate embedding vectors at these timesteps
        embedding_dim: int: dimension of the embeddings to generate
        dtype: data type of the generated embeddings
        Returns:
        embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = mint.log(ms.Tensor(10000.0)) / (half_dim - 1)
        emb = mint.exp(mint.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = mint.cat([mint.sin(emb), mint.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = mint.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def __call__(
        self,
        prompt: Union[str, List[str], ms.Tensor] = None,
        height: Optional[int] = 320,
        width: Optional[int] = 512,
        frames: int = 16,
        fps: int = 16,
        guidance_scale: float = 7.5,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        latents: Optional[ms.Tensor] = None,
        num_inference_steps: int = 4,
        lcm_origin_steps: int = 50,
        prompt_embeds: Optional[ms.Tensor] = None,
        output_type: Optional[str] = "pil",
    ):
        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        elif prompt is not None and isinstance(prompt, ms.Tensor):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # do_classifier_free_guidance = guidance_scale > 0.0
        # # In LCM Implementation:  cfg_noise = noise_cond + cfg_scale * (noise_cond - noise_uncond)
        # (cfg_scale > 0.0 using CFG)

        # 3. Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt,
            num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
        )

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, lcm_origin_steps)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variable
        num_channels_latents = unet_config["params"]["in_channels"]
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            frames,
            height,
            width,
            prompt_embeds.dtype,
            generator,
            latents,
        )

        bs = batch_size * num_videos_per_prompt

        # 6. Get Guidance Scale Embedding
        w = ms.Tensor(guidance_scale).repeat(bs)
        w_embedding = self.get_w_embedding(w, embedding_dim=256)

        # 7. LCM MultiStep Sampling Loop:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                ts = ops.full((bs,), t, dtype=ms.int32)

                # model prediction (v-prediction, eps, x)
                context = {"context": mint.cat([prompt_embeds.float()], 1), "fps": fps}
                model_pred = self.unet(
                    latents,
                    ts,
                    **context,
                    timestep_cond=w_embedding.to(self.dtype),
                )
                # compute the previous noisy sample x_t -> x_t-1
                latents, denoised = self.scheduler.step(model_pred, i, t, latents, return_dict=False)

                # # call the callback, if provided
                # if i == len(timesteps) - 1:
                progress_bar.update()

        if not output_type == "latent":
            # denoised -> Tensor, (1, 4, 16, 40, 64)
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)
        else:
            videos = denoised

        return videos
