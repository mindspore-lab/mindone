# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import nn, ops

from ...configuration_utils import ConfigMixin, register_to_config
from ...schedulers import ConsistencyDecoderScheduler
from ...utils import BaseOutput
from ...utils.mindspore_utils import randn_tensor
from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from ..modeling_utils import ModelMixin
from ..unets.unet_2d import UNet2DModel
from .vae import DecoderOutput, DiagonalGaussianDistribution, Encoder


@dataclass
class ConsistencyDecoderVAEOutput(BaseOutput):
    """
    Output of encoding method.

    Args:
        latent_dist (`DiagonalGaussianDistribution`):
            Encoded outputs of `Encoder` represented as the mean and logvar of `DiagonalGaussianDistribution`.
            `DiagonalGaussianDistribution` allows for sampling latents from the distribution.
    """

    latent: ms.Tensor


class ConsistencyDecoderVAE(ModelMixin, ConfigMixin):
    r"""
    The consistency decoder used with DALL-E 3.

    Examples:
        ```py
        >>> import mindspore
        >>> from mindone.diffusers import StableDiffusionPipeline, ConsistencyDecoderVAE

        >>> vae = ConsistencyDecoderVAE.from_pretrained("openai/consistency-decoder", mindspore_dtype=mindspore.float16)
        >>> pipe = StableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", vae=vae, mindspore_dtype=mindspore.float16
        ... )

        >>> pipe("horse").images
        ```
    """

    @register_to_config
    def __init__(
        self,
        scaling_factor: float = 0.18215,
        latent_channels: int = 4,
        encoder_act_fn: str = "silu",
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        encoder_double_z: bool = True,
        encoder_down_block_types: Tuple[str, ...] = (
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
            "DownEncoderBlock2D",
        ),
        encoder_in_channels: int = 3,
        encoder_layers_per_block: int = 2,
        encoder_norm_num_groups: int = 32,
        encoder_out_channels: int = 4,
        decoder_add_attention: bool = False,
        decoder_block_out_channels: Tuple[int, ...] = (320, 640, 1024, 1024),
        decoder_down_block_types: Tuple[str, ...] = (
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
            "ResnetDownsampleBlock2D",
        ),
        decoder_downsample_padding: int = 1,
        decoder_in_channels: int = 7,
        decoder_layers_per_block: int = 3,
        decoder_norm_eps: float = 1e-05,
        decoder_norm_num_groups: int = 32,
        decoder_num_train_timesteps: int = 1024,
        decoder_out_channels: int = 6,
        decoder_resnet_time_scale_shift: str = "scale_shift",
        decoder_time_embedding_type: str = "learned",
        decoder_up_block_types: Tuple[str, ...] = (
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
            "ResnetUpsampleBlock2D",
        ),
    ):
        super().__init__()
        self.encoder = Encoder(
            act_fn=encoder_act_fn,
            block_out_channels=encoder_block_out_channels,
            double_z=encoder_double_z,
            down_block_types=encoder_down_block_types,
            in_channels=encoder_in_channels,
            layers_per_block=encoder_layers_per_block,
            norm_num_groups=encoder_norm_num_groups,
            out_channels=encoder_out_channels,
        )

        self.decoder_unet = UNet2DModel(
            add_attention=decoder_add_attention,
            block_out_channels=decoder_block_out_channels,
            down_block_types=decoder_down_block_types,
            downsample_padding=decoder_downsample_padding,
            in_channels=decoder_in_channels,
            layers_per_block=decoder_layers_per_block,
            norm_eps=decoder_norm_eps,
            norm_num_groups=decoder_norm_num_groups,
            num_train_timesteps=decoder_num_train_timesteps,
            out_channels=decoder_out_channels,
            resnet_time_scale_shift=decoder_resnet_time_scale_shift,
            time_embedding_type=decoder_time_embedding_type,
            up_block_types=decoder_up_block_types,
        )
        self.diag_gauss_dist = DiagonalGaussianDistribution()
        self.decoder_scheduler = ConsistencyDecoderScheduler()
        self.register_to_config(block_out_channels=encoder_block_out_channels)
        self.register_to_config(force_upcast=False)
        self.means = ms.Tensor([0.38862467, 0.02253063, 0.07381133, -0.0171294])[None, :, None, None]
        self.stds = ms.Tensor([0.9654121, 1.0440036, 0.76147926, 0.77022034])[None, :, None, None]

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1, has_bias=True)

        self.use_slicing = False
        self.use_tiling = False

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:  # type: ignore
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors: Dict[str, AttentionProcessor]):  # type: ignore
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.name_cells().items():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.name_cells().items():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):  # type: ignore
        r"""
        Sets the attention processor to use to compute attention.
        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Cell, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.name_cells().items():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.name_cells().items():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def encode(
        self, x: ms.Tensor, return_dict: bool = False
    ) -> Union[ConsistencyDecoderVAEOutput, Tuple[DiagonalGaussianDistribution]]:
        """
        Encode a batch of images into latents.

        Args:
            x (`ms.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.consistecy_decoder_vae.ConsistencyDecoderOoutput`] instead of a plain
                tuple.

        Returns:
                The latent representations of the encoded images. If `return_dict` is True, a
                [`~models.consistency_decoder_vae.ConsistencyDecoderVAEOutput`] is returned, otherwise a plain `tuple`
                is returned.
        """
        h = self.encoder(x)

        moments = self.quant_conv(h)

        if not return_dict:
            return (moments,)

        return ConsistencyDecoderVAEOutput(latent=moments)

    def decode(
        self,
        z: ms.Tensor,
        generator: Optional[np.random.Generator] = None,
        return_dict: bool = False,
        num_inference_steps: int = 2,
    ) -> Union[DecoderOutput, Tuple[ms.Tensor]]:
        z = ((z * self.config["scaling_factor"] - self.means) / self.stds).to(z.dtype)

        scale_factor = 2 ** (len(self.config["block_out_channels"]) - 1)
        z = ops.interpolate(z, mode="nearest", size=(z.shape[-2] * scale_factor, z.shape[-1] * scale_factor))

        batch_size, _, height, width = z.shape

        # self.decoder_scheduler.set_timesteps(num_inference_steps)

        x_t = self.decoder_scheduler.init_noise_sigma * randn_tensor(
            (batch_size, 3, height, width),
            generator=generator,
            dtype=z.dtype,
        )

        for t in self.decoder_scheduler.timesteps:
            model_input = ops.concat([self.decoder_scheduler.scale_model_input(x_t, t).to(z.dtype), z], axis=1)
            model_output = self.decoder_unet(model_input, t)[0][:, :3, :, :]
            prev_sample = self.decoder_scheduler.step(model_output, t, x_t, generator)[0]
            x_t = prev_sample

        x_0 = x_t

        if not return_dict:
            return (x_0,)

        return DecoderOutput(sample=x_0)

    def construct(
        self,
        sample: ms.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = False,
        generator: Optional[np.random.Generator] = None,
    ) -> Union[DecoderOutput, Tuple[ms.Tensor]]:
        r"""
        Args:
            sample (`ms.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`np.random.Generator`, *optional*, defaults to `None`):
                Generator to use for sampling.

        Returns:
            [`DecoderOutput`] or `tuple`:
                If return_dict is True, a [`DecoderOutput`] is returned, otherwise a plain `tuple` is returned.
        """
        x = sample
        latent = self.encode(x)[0]
        if sample_posterior:
            z = self.diag_gauss_dist.sample(latent)
        else:
            z = self.diag_gauss_dist.mode(latent)
        dec = self.decode(z, generator=generator)[0]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
