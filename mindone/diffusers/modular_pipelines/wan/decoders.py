# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from typing import Any, List, Tuple, Union

import numpy as np
import PIL

import mindspore as ms

from ...configuration_utils import FrozenDict
from ...models import AutoencoderKLWan
from ...utils import logging
from ...video_processor import VideoProcessor
from ..modular_pipeline import ModularPipelineBlocks, PipelineState
from ..modular_pipeline_utils import ComponentSpec, InputParam, OutputParam

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class WanDecodeStep(ModularPipelineBlocks):
    model_name = "wan"

    @property
    def expected_components(self) -> List[ComponentSpec]:
        return [
            ComponentSpec("vae", AutoencoderKLWan),
            ComponentSpec(
                "video_processor",
                VideoProcessor,
                config=FrozenDict({"vae_scale_factor": 8}),
                default_creation_method="from_config",
            ),
        ]

    @property
    def description(self) -> str:
        return "Step that decodes the denoised latents into images"

    @property
    def inputs(self) -> List[Tuple[str, Any]]:
        return [
            InputParam("output_type", default="pil"),
        ]

    @property
    def intermediate_inputs(self) -> List[str]:
        return [
            InputParam(
                "latents",
                required=True,
                type_hint=ms.Tensor,
                description="The denoised latents from the denoising step",
            )
        ]

    @property
    def intermediate_outputs(self) -> List[str]:
        return [
            OutputParam(
                "videos",
                type_hint=Union[List[List[PIL.Image.Image]], List[ms.Tensor], List[np.ndarray]],
                description="The generated videos, can be a PIL.Image.Image, ms.Tensor or a numpy array",
            )
        ]

    def __call__(self, components, state: PipelineState) -> PipelineState:
        block_state = self.get_block_state(state)
        vae_dtype = components.vae.dtype

        if not block_state.output_type == "latent":
            latents = block_state.latents
            latents_mean = (
                ms.Tensor(components.vae.config.latents_mean)
                .view(1, components.vae.config.z_dim, 1, 1, 1)
                .to(latents.dtype)
            )
            latents_std = 1.0 / ms.Tensor(components.vae.config.latents_std).view(
                1, components.vae.config.z_dim, 1, 1, 1
            ).to(latents.dtype)
            latents = latents / latents_std + latents_mean
            latents = latents.to(vae_dtype)
            block_state.videos = components.vae.decode(latents, return_dict=False)[0]
        else:
            block_state.videos = block_state.latents

        block_state.videos = components.video_processor.postprocess_video(
            block_state.videos, output_type=block_state.output_type
        )

        self.set_block_state(state, block_state)

        return components, state
