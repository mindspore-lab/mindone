# Copyright 2025 Alibaba DAMO-VILAB and The HuggingFace Team. All rights reserved.
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

from typing import Any, Dict, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, nn, ops

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import UNet2DConditionLoadersMixin
from ...utils import logging
from ..activations import get_activation
from ..attention import Attention, FeedForward
from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..transformers.transformer_temporal import TransformerTemporalModel
from .unet_3d_blocks import UNetMidBlock3DCrossAttn, get_down_block, get_up_block
from .unet_3d_condition import UNet3DConditionOutput

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class I2VGenXLTransformerTemporalEncoder(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        activation_fn: str = "geglu",
        upcast_attention: bool = False,
        ff_inner_dim: Optional[int] = None,
        dropout: int = 0.0,
    ):
        super().__init__()
        self.norm1 = mint.nn.LayerNorm(dim, elementwise_affine=True, eps=1e-5)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=False,
            upcast_attention=upcast_attention,
            out_bias=True,
        )
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=False,
            inner_dim=ff_inner_dim,
            bias=True,
        )

    def construct(
        self,
        hidden_states: ms.Tensor,
    ) -> ms.Tensor:
        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        ff_output = self.ff(hidden_states)
        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class I2VGenXLUNet(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin):
    r"""
    I2VGenXL UNet. It is a conditional 3D UNet model that takes a noisy sample, conditional state, and a timestep and
    returns a sample-shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 4): The number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): The number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32): The number of groups to use for the normalization.
            If `None`, normalization and activation layers is skipped in post-processing.
        cross_attention_dim (`int`, *optional*, defaults to 1280): The dimension of the cross attention features.
        attention_head_dim (`int`, *optional*, defaults to 64): Attention head dim.
        num_attention_heads (`int`, *optional*): The number of attention heads.
    """

    _supports_gradient_checkpointing = False

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        norm_num_groups: Optional[int] = 32,
        cross_attention_dim: int = 1024,
        attention_head_dim: Union[int, Tuple[int]] = 64,
        num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
    ):
        super().__init__()

        # When we first integrated the UNet into the library, we didn't have `attention_head_dim`. As a consequence
        # of that, we used `num_attention_heads` for arguments that actually denote attention head dimension. This
        # is why we ignore `num_attention_heads` and calculate it from `attention_head_dims` below.
        # This is still an incorrect way of calculating `num_attention_heads` but we need to stick to it
        # without running proper deprecation cycles for the {down,mid,up} blocks which are a
        # part of the public API.
        num_attention_heads = attention_head_dim

        # Check inputs
        if len(down_block_types) != len(up_block_types):
            raise ValueError(
                f"Must provide the same number of `down_block_types` as `up_block_types`. "
                f"`down_block_types`: {down_block_types}. `up_block_types`: {up_block_types}."
            )

        if len(block_out_channels) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `block_out_channels` as `down_block_types`. "
                f"`block_out_channels`: {block_out_channels}. `down_block_types`: {down_block_types}."
            )

        if not isinstance(num_attention_heads, int) and len(num_attention_heads) != len(down_block_types):
            raise ValueError(
                f"Must provide the same number of `num_attention_heads` as `down_block_types`. "
                f"`num_attention_heads`: {num_attention_heads}. `down_block_types`: {down_block_types}."
            )

        # input
        self.conv_in = mint.nn.Conv2d(in_channels + in_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.transformer_in = TransformerTemporalModel(
            num_attention_heads=8,
            attention_head_dim=num_attention_heads,
            in_channels=block_out_channels[0],
            num_layers=1,
            norm_num_groups=norm_num_groups,
        )

        # image embedding
        self.image_latents_proj_in = nn.SequentialCell(
            mint.nn.Conv2d(4, in_channels * 4, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(in_channels * 4, in_channels * 4, 3, stride=1, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(in_channels * 4, in_channels, 3, stride=1, padding=1),
        )
        self.image_latents_temporal_encoder = I2VGenXLTransformerTemporalEncoder(
            dim=in_channels,
            num_attention_heads=2,
            ff_inner_dim=in_channels * 4,
            attention_head_dim=in_channels,
            activation_fn="gelu",
        )
        self.image_latents_context_embedding = nn.SequentialCell(
            mint.nn.Conv2d(4, in_channels * 8, 3, padding=1),
            mint.nn.SiLU(),
            mint.nn.AdaptiveAvgPool2d((32, 32)),
            mint.nn.Conv2d(in_channels * 8, in_channels * 16, 3, stride=2, padding=1),
            mint.nn.SiLU(),
            mint.nn.Conv2d(in_channels * 16, cross_attention_dim, 3, stride=2, padding=1),
        )

        # other embeddings -- time, context, fps, etc.
        time_embed_dim = block_out_channels[0] * 4
        self.time_proj = Timesteps(block_out_channels[0], True, 0)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn="silu")
        self.context_embedding = nn.SequentialCell(
            mint.nn.Linear(cross_attention_dim, time_embed_dim),
            mint.nn.SiLU(),
            mint.nn.Linear(time_embed_dim, cross_attention_dim * in_channels),
        )
        self.fps_embedding = nn.SequentialCell(
            mint.nn.Linear(timestep_input_dim, time_embed_dim),
            mint.nn.SiLU(),
            mint.nn.Linear(time_embed_dim, time_embed_dim),
        )

        # blocks
        self.down_blocks = []
        self.up_blocks = []

        if isinstance(num_attention_heads, int):
            num_attention_heads = (num_attention_heads,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=1e-05,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=num_attention_heads[i],
                downsample_padding=1,
                dual_cross_attention=False,
            )
            self.down_blocks.append(down_block)
        self.down_blocks = nn.CellList(self.down_blocks)

        # mid
        self.mid_block = UNetMidBlock3DCrossAttn(
            in_channels=block_out_channels[-1],
            temb_channels=time_embed_dim,
            resnet_eps=1e-05,
            resnet_act_fn="silu",
            output_scale_factor=1,
            cross_attention_dim=cross_attention_dim,
            num_attention_heads=num_attention_heads[-1],
            resnet_groups=norm_num_groups,
            dual_cross_attention=False,
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        layers_per_resnet_in_up_blocks = []
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_num_attention_heads = list(reversed(num_attention_heads))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=1e-05,
                resnet_act_fn="silu",
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                num_attention_heads=reversed_num_attention_heads[i],
                dual_cross_attention=False,
                resolution_idx=i,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel
            layers_per_resnet_in_up_blocks.append(len(up_block.resnets))
        self.up_blocks = nn.CellList(self.up_blocks)
        self.layers_per_resnet_in_up_blocks = layers_per_resnet_in_up_blocks

        # out
        self.conv_norm_out = mint.nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-05
        )
        self.conv_act = get_activation("silu")
        self.conv_out = mint.nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

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

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(self, chunk_size: Optional[int] = None, dim: int = 0) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(module: nn.Cell, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.name_cells().values():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.name_cells().values():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(module: nn.Cell, chunk_size: int, dim: int):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.name_cells().values():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.name_cells().values():
            fn_recursive_feed_forward(module, None, 0)

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

    def construct(
        self,
        sample: ms.Tensor,
        timestep: Union[ms.Tensor, float, int],
        fps: ms.Tensor,
        image_latents: ms.Tensor,
        image_embeddings: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        timestep_cond: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[UNet3DConditionOutput, Tuple[ms.Tensor]]:
        r"""
        The [`I2VGenXLUNet`] forward method.

        Args:
            sample (`ms.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width`.
            timestep (`ms.Tensor` or `float` or `int`): The number of timesteps to denoise an input.
            fps (`ms.Tensor`): Frames per second for the video being generated. Used as a "micro-condition".
            image_latents (`ms.Tensor`): Image encodings from the VAE.
            image_embeddings (`ms.Tensor`):
                Projection embeddings of the conditioning image computed with a vision encoder.
            encoder_hidden_states (`ms.Tensor`):
                The encoder hidden states with shape `(batch, sequence_length, feature_dim)`.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] instead of a plain
                tuple.

        Returns:
            [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unets.unet_3d_condition.UNet3DConditionOutput`] is returned,
                otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        batch_size, channels, num_frames, height, width = sample.shape

        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if sample.shape[-1] % default_overall_up_factor != 0 or sample.shape[-2] % default_overall_up_factor != 0:
            forward_upsample_size = True

        # 1. time
        timesteps = timestep
        if not ops.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass `timesteps` as tensors if you can
            if isinstance(timesteps, float):
                dtype = ms.float32
            else:
                dtype = ms.int32
            timesteps = ms.tensor([timesteps], dtype=dtype)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None]

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.broadcast_to((sample.shape[0],))
        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        t_emb = self.time_embedding(t_emb, timestep_cond)

        # 2. FPS
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        fps = fps.broadcast_to((fps.shape[0],))
        fps_emb = self.fps_embedding(self.time_proj(fps).to(dtype=self.dtype))

        # 3. time + FPS embeddings.
        emb = t_emb + fps_emb
        emb = emb.repeat_interleave(num_frames, dim=0, output_size=emb.shape[0] * num_frames)

        # 4. context embeddings.
        # The context embeddings consist of both text embeddings from the input prompt
        # AND the image embeddings from the input image. For images, both VAE encodings
        # and the CLIP image embeddings are incorporated.
        # So the final `context_embeddings` becomes the query for cross-attention.
        context_emb = sample.new_zeros((batch_size, 0, self.config["cross_attention_dim"]), dtype=sample.dtype)
        context_emb = context_emb.to(sample.dtype)
        context_emb = mint.cat([context_emb, encoder_hidden_states], dim=1)

        image_latents_for_context_embds = image_latents[:, :, :1, :]
        image_latents_context_embs = image_latents_for_context_embds.permute(0, 2, 1, 3, 4).reshape(
            image_latents_for_context_embds.shape[0] * image_latents_for_context_embds.shape[2],
            image_latents_for_context_embds.shape[1],
            image_latents_for_context_embds.shape[3],
            image_latents_for_context_embds.shape[4],
        )
        image_latents_context_embs = self.image_latents_context_embedding(image_latents_context_embs)

        _batch_size, _channels, _height, _width = image_latents_context_embs.shape
        image_latents_context_embs = image_latents_context_embs.permute(0, 2, 3, 1).reshape(
            _batch_size, _height * _width, _channels
        )
        context_emb = mint.cat([context_emb, image_latents_context_embs], dim=1)

        image_emb = self.context_embedding(image_embeddings)
        image_emb = image_emb.view(-1, self.config["in_channels"], self.config["cross_attention_dim"])
        context_emb = mint.cat([context_emb, image_emb], dim=1)
        context_emb = context_emb.repeat_interleave(num_frames, dim=0, output_size=context_emb.shape[0] * num_frames)

        image_latents = image_latents.permute(0, 2, 1, 3, 4).reshape(
            image_latents.shape[0] * image_latents.shape[2],
            image_latents.shape[1],
            image_latents.shape[3],
            image_latents.shape[4],
        )
        image_latents = self.image_latents_proj_in(image_latents)
        image_latents = mint.reshape(
            mint.permute(
                mint.reshape(image_latents[None, :], (batch_size, num_frames, channels, height, width)), (0, 3, 4, 1, 2)
            ),
            (batch_size * height * width, num_frames, channels),
        )
        image_latents = self.image_latents_temporal_encoder(image_latents)
        image_latents = mint.permute(
            mint.reshape(image_latents, (batch_size, height, width, num_frames, channels)), (0, 4, 3, 1, 2)
        )

        # 5. pre-process
        sample = mint.cat([sample, image_latents], dim=1)
        sample = sample.permute(0, 2, 1, 3, 4).reshape((sample.shape[0] * num_frames, -1) + sample.shape[3:])

        sample = self.conv_in(sample)
        sample = self.transformer_in(
            sample,
            num_frames=num_frames,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False,
        )[0]

        # 6. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=context_emb,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb, num_frames=num_frames)

            down_block_res_samples += res_samples

        # 7. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=context_emb,
                num_frames=num_frames,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        # 8. up
        for i, upsample_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-self.layers_per_resnet_in_up_blocks[i] :]
            down_block_res_samples = down_block_res_samples[: -self.layers_per_resnet_in_up_blocks[i]]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=context_emb,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                    num_frames=num_frames,
                )

        # 9. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)

        sample = self.conv_out(sample)

        # reshape to (batch, channel, framerate, width, height)
        sample = sample[None, :].reshape((-1, num_frames) + sample.shape[1:]).permute(0, 2, 1, 3, 4)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)
