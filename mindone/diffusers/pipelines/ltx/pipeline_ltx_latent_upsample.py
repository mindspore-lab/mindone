# Copyright 2025 Lightricks and The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Union

import numpy as np

import mindspore as ms
from mindspore import mint

from ...image_processor import PipelineImageInput
from ...models import AutoencoderKLLTXVideo
from ...utils import get_logger
from ...utils.mindspore_utils import randn_tensor
from ...video_processor import VideoProcessor
from ..pipeline_utils import DiffusionPipeline
from .modeling_latent_upsampler import LTXLatentUpsamplerModel
from .pipeline_output import LTXPipelineOutput

logger = get_logger(__name__)  # pylint: disable=invalid-name


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    vae, encoder_output: ms.Tensor, generator: Optional[np.random.Generator] = None, sample_mode: str = "sample"
):
    if sample_mode == "sample":
        return vae.diag_gauss_dist.sample(encoder_output, generator=generator)
    elif sample_mode == "argmax":
        return vae.diag_gauss_dist.mode(encoder_output)
    # This branch is not needed because the encoder_output type is ms.Tensor as per AutoencoderKLOutput change
    # elif hasattr(encoder_output, "latents"):
    #     return encoder_output.latents
    else:
        return encoder_output


class LTXLatentUpsamplePipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKLLTXVideo,
        latent_upsampler: LTXLatentUpsamplerModel,
    ) -> None:
        super().__init__()

        self.register_modules(vae=vae, latent_upsampler=latent_upsampler)

        self.vae_spatial_compression_ratio = (
            self.vae.spatial_compression_ratio if getattr(self, "vae", None) is not None else 32
        )
        self.vae_temporal_compression_ratio = (
            self.vae.temporal_compression_ratio if getattr(self, "vae", None) is not None else 8
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_spatial_compression_ratio)

    def prepare_latents(
        self,
        video: Optional[ms.Tensor] = None,
        batch_size: int = 1,
        dtype: Optional[ms.Type] = None,
        generator: Optional[np.random.Generator] = None,
        latents: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:
        if latents is not None:
            return latents.to(dtype=dtype)

        video = video.to(dtype=self.vae.dtype)
        if isinstance(generator, list):
            if len(generator) != batch_size:
                raise ValueError(
                    f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                    f" size of {batch_size}. Make sure the batch size matches the length of the generators."
                )

            init_latents = [
                retrieve_latents(self.vae, self.vae.encode(video[i].unsqueeze(0))[0], generator[i])
                for i in range(batch_size)
            ]
        else:
            init_latents = [
                retrieve_latents(self.vae, self.vae.encode(vid.unsqueeze(0))[0], generator) for vid in video
            ]

        init_latents = mint.cat(init_latents, dim=0).to(dtype)
        init_latents = self._normalize_latents(init_latents, self.vae.latents_mean, self.vae.latents_std)
        return init_latents

    def adain_filter_latent(self, latents: ms.Tensor, reference_latents: ms.Tensor, factor: float = 1.0):
        """
        Applies Adaptive Instance Normalization (AdaIN) to a latent tensor based on statistics from a reference latent
        tensor.

        Args:
            latent (`torch.Tensor`):
                Input latents to normalize
            reference_latents (`torch.Tensor`):
                The reference latents providing style statistics.
            factor (`float`):
                Blending factor between original and transformed latent. Range: -10.0 to 10.0, Default: 1.0

        Returns:
            torch.Tensor: The transformed latent tensor
        """
        result = latents.clone()

        for i in range(latents.shape[0]):
            for c in range(latents.shape[1]):
                r_sd, r_mean = mint.std_mean(reference_latents[i, c], dim=None)  # index by original dim order
                i_sd, i_mean = mint.std_mean(result[i, c], dim=None)

                result[i, c] = ((result[i, c] - i_mean) / i_sd) * r_sd + r_mean

        result = mint.lerp(latents, result, factor)
        return result

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._normalize_latents
    def _normalize_latents(
        latents: ms.Tensor, latents_mean: ms.Tensor, latents_std: ms.Tensor, scaling_factor: float = 1.0
    ) -> ms.Tensor:
        # Normalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents = (latents - latents_mean) * scaling_factor / latents_std
        return latents

    @staticmethod
    # Copied from diffusers.pipelines.ltx.pipeline_ltx.LTXPipeline._denormalize_latents
    def _denormalize_latents(
        latents: ms.Tensor, latents_mean: ms.Tensor, latents_std: ms.Tensor, scaling_factor: float = 1.0
    ) -> ms.Tensor:
        # Denormalize latents across the channel dimension [B, C, F, H, W]
        latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents_std = latents_std.view(1, -1, 1, 1, 1).to(latents.dtype)
        latents = latents * latents_std / scaling_factor + latents_mean
        return latents

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def check_inputs(self, video, height, width, latents):
        if height % self.vae_spatial_compression_ratio != 0 or width % self.vae_spatial_compression_ratio != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 32 but are {height} and {width}.")

        if video is not None and latents is not None:
            raise ValueError("Only one of `video` or `latents` can be provided.")
        if video is None and latents is None:
            raise ValueError("One of `video` or `latents` has to be provided.")

    def __call__(
        self,
        video: Optional[List[PipelineImageInput]] = None,
        height: int = 512,
        width: int = 704,
        latents: Optional[ms.Tensor] = None,
        decode_timestep: Union[float, List[float]] = 0.0,
        decode_noise_scale: Optional[Union[float, List[float]]] = None,
        adain_factor: float = 0.0,
        generator: Optional[Union[np.random.Generator, List[np.random.Generator]]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ):
        self.check_inputs(
            video=video,
            height=height,
            width=width,
            latents=latents,
        )

        if video is not None:
            # Batched video input is not yet tested/supported. TODO: take a look later
            batch_size = 1
        else:
            batch_size = latents.shape[0]

        if video is not None:
            num_frames = len(video)
            if num_frames % self.vae_temporal_compression_ratio != 1:
                num_frames = num_frames // self.vae_temporal_compression_ratio * self.vae_temporal_compression_ratio + 1
                video = video[:num_frames]
                logger.warning(
                    f"Video length expected to be of the form `k * {self.vae_temporal_compression_ratio} + 1` but is {len(video)}. Truncating to {num_frames} frames."  # noqa: E501
                )
            video = self.video_processor.preprocess_video(video, height=height, width=width)
            video = video.to(dtype=ms.float32)

        latents = self.prepare_latents(
            video=video,
            batch_size=batch_size,
            dtype=ms.float32,
            generator=generator,
            latents=latents,
        )

        latents = self._denormalize_latents(
            latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
        )
        latents = latents.to(self.latent_upsampler.dtype)
        latents_upsampled = self.latent_upsampler(latents)

        if adain_factor > 0.0:
            latents = self.adain_filter_latent(latents_upsampled, latents, adain_factor)
        else:
            latents = latents_upsampled

        if output_type == "latent":
            latents = self._normalize_latents(
                latents, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor
            )
            video = latents
        else:
            if not self.vae.config.timestep_conditioning:
                timestep = None
            else:
                noise = randn_tensor(latents.shape, generator=generator, dtype=latents.dtype)
                if not isinstance(decode_timestep, list):
                    decode_timestep = [decode_timestep] * batch_size
                if decode_noise_scale is None:
                    decode_noise_scale = decode_timestep
                elif not isinstance(decode_noise_scale, list):
                    decode_noise_scale = [decode_noise_scale] * batch_size

                timestep = ms.tensor(decode_timestep, dtype=latents.dtype)
                decode_noise_scale = ms.tensor(decode_noise_scale, dtype=latents.dtype)[:, None, None, None, None]
                latents = (1 - decode_noise_scale) * latents + decode_noise_scale * noise

            video = self.vae.decode(latents, timestep, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)

        if not return_dict:
            return (video,)

        return LTXPipelineOutput(frames=video)
