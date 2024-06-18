import glob
import json
import logging
import os
from typing import Any, Dict, Optional

from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.models.diffusion.utils.pos_embed import PositionGetter1D, PositionGetter2D, get_1d_sincos_pos_embed

import mindspore as ms
from mindspore import nn, ops

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config

# from mindone.diffusers.utils import USE_PEFT_BACKEND
from mindone.diffusers.models.modeling_utils import ModelMixin

from .modules import (
    AdaLayerNormSingle,
    BasicTransformerBlock,
    BasicTransformerBlock_,
    CaptionProjection,
    LatteT2VBlock,
    LayerNorm,
    PatchEmbed,
)

logger = logging.getLogger(__name__)


class LatteT2V(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        patch_size_t: int = 1,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        video_length: int = 16,
        enable_flash_attention: bool = False,
        use_recompute=False,
        use_rope: bool = False,
        model_max_length: int = 300,
        rope_scaling_type: str = "linear",
        compress_kv_factor: int = 1,
        interpolation_scale_1d: float = None,
        FA_dtype=ms.bfloat16,
        num_no_recompute: int = 0,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.video_length = video_length
        self.norm_type = norm_type
        self.use_recompute = use_recompute
        self.use_rope = use_rope
        self.model_max_length = model_max_length
        self.compress_kv_factor = compress_kv_factor
        self.num_layers = num_layers
        self.config.hidden_size = model_max_length
        self.FA_dtype = FA_dtype

        assert not (self.compress_kv_factor != 1 and use_rope), "Can not both enable compressing kv and using rope"

        conv_cls = nn.Conv2d  # if USE_PEFT_BACKEND else LoRACompatibleConv
        linear_cls = nn.Dense  # if USE_PEFT_BACKEND else LoRACompatibleLinear

        # 1. Transformer2DModel can process both standard continuous images of shape
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        # self.is_input_patches = in_channels is not None and patch_size is not None
        self.is_input_patches = True

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            logger.warning("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"

        # 2. Define input layers
        assert sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

        self.height = sample_size[0]
        self.width = sample_size[1]

        self.patch_size = patch_size
        interpolation_scale_2d = self.config.sample_size[0] // 64  # => 64 (= 512 pixart) has interpolation scale 1
        interpolation_scale_2d = max(interpolation_scale_2d, 1)
        self.pos_embed = PatchEmbed(
            height=sample_size[0],
            width=sample_size[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale_2d,
        )

        # define temporal positional embedding
        if interpolation_scale_1d is None:
            if self.config.video_length % 2 == 1:
                interpolation_scale_1d = (
                    self.config.video_length - 1
                ) // 16  # => 16 (= 16 Latte) has interpolation scale 1
            else:
                interpolation_scale_1d = self.config.video_length // 16  # => 16 (= 16 Latte) has interpolation scale 1
        # interpolation_scale_1d = self.config.video_length // 5  #
        interpolation_scale_1d = max(interpolation_scale_1d, 1)

        if get_sequence_parallel_state():
            self.sp_size = hccl_info.world_size
            rank_offset = hccl_info.rank % hccl_info.world_size
            video_length = (self.video_length + self.sp_size - 1) // self.sp_size * self.sp_size
            temp_pos_embed = get_1d_sincos_pos_embed(
                inner_dim, video_length, interpolation_scale=interpolation_scale_1d
            )  # 1152 hidden size
            video_length //= self.sp_size
            self.temp_pos_st = rank_offset * video_length
            self.temp_pos_ed = (rank_offset + 1) * video_length
        else:
            temp_pos_embed = get_1d_sincos_pos_embed(
                inner_dim, video_length, interpolation_scale=interpolation_scale_1d
            )  # 1152 hidden size
        self.temp_pos_embed = ms.Parameter(ms.Tensor(temp_pos_embed).float(), requires_grad=False)

        rope_scaling = None
        if self.use_rope:
            self.position_getter_2d = PositionGetter2D()
            self.position_getter_1d = PositionGetter1D()
            rope_scaling = dict(
                type=rope_scaling_type, factor_2d=interpolation_scale_2d, factor_1d=interpolation_scale_1d
            )

        # 3. Define transformers blocks, spatial attention
        self.transformer_blocks = [
            BasicTransformerBlock(
                inner_dim,
                num_attention_heads,
                attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                activation_fn=activation_fn,
                num_embeds_ada_norm=num_embeds_ada_norm,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
                double_self_attention=double_self_attention,
                upcast_attention=upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                attention_type=attention_type,
                enable_flash_attention=enable_flash_attention,
                use_rope=use_rope,
                rope_scaling=rope_scaling,
                FA_dtype=self.FA_dtype,
                compress_kv_factor=(compress_kv_factor, compress_kv_factor)
                if d >= num_layers // 2 and compress_kv_factor != 1
                else None,  # follow pixart-sigma, apply in second-half layers
            )
            for d in range(num_layers)
        ]

        # Define temporal transformers blocks
        self.temporal_transformer_blocks = [
            BasicTransformerBlock_(  # one attention
                inner_dim,
                num_attention_heads,  # num_attention_heads
                attention_head_dim,  # attention_head_dim 72
                dropout=dropout,
                cross_attention_dim=None,
                activation_fn=activation_fn,
                num_embeds_ada_norm=num_embeds_ada_norm,
                attention_bias=attention_bias,
                only_cross_attention=only_cross_attention,
                double_self_attention=False,
                upcast_attention=upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                attention_type=attention_type,
                enable_flash_attention=enable_flash_attention,
                use_rope=use_rope,
                rope_scaling=rope_scaling,
                FA_dtype=self.FA_dtype,
                compress_kv_factor=(compress_kv_factor,)
                if d >= num_layers // 2 and compress_kv_factor != 1
                else None,  # follow pixart-sigma, apply in second-half layers
            )
            for d in range(num_layers)
        ]

        # 4. Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.is_input_continuous:
            # TODO: should use out_channels for continuous projections
            if use_linear_projection:
                self.proj_out = linear_cls(inner_dim, in_channels)
            else:
                self.proj_out = conv_cls(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
        elif self.is_input_vectorized:
            self.norm_out = LayerNorm(inner_dim)
            self.out = nn.Dense(inner_dim, self.num_vector_embeds - 1)
        elif self.is_input_patches and norm_type != "ada_norm_single":
            self.norm_out = LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Dense(inner_dim, 2 * inner_dim)
            self.proj_out_2 = nn.Dense(inner_dim, patch_size * patch_size * self.out_channels)
        elif self.is_input_patches and norm_type == "ada_norm_single":
            self.norm_out = LayerNorm(inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = ms.Parameter(ops.randn(2, inner_dim) / inner_dim**0.5)
            self.proj_out = nn.Dense(inner_dim, patch_size * patch_size * self.out_channels)

        # 5. PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if norm_type == "ada_norm_single":
            # self.use_additional_conditions = self.config.sample_size[0] == 128  # False, 128 -> 1024
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(inner_dim, use_additional_conditions=self.use_additional_conditions)

        self.caption_projection = None
        if caption_channels is not None:
            self.caption_projection = CaptionProjection(in_features=caption_channels, hidden_size=inner_dim)

        self.gradient_checkpointing = False

        self.blocks = nn.CellList(
            [
                LatteT2VBlock(d, self.transformer_blocks[d], self.temporal_transformer_blocks[d])
                for d in range(num_layers)
            ]
        )

        if self.use_recompute:
            num_no_recompute = self.config.num_no_recompute
            num_blocks = len(self.blocks)
            assert num_no_recompute >= 0, "Expect to have num_no_recompute as a positive integer."
            assert (
                num_no_recompute <= num_blocks
            ), "Expect to have num_no_recompute as an integer no greater than the number of blocks,"
            f"but got {num_no_recompute} and {num_blocks}."
            logger.info(f"Excluding {num_no_recompute} blocks from the recomputation list.")
            for bidx, block in enumerate(self.blocks):
                if bidx < num_blocks - num_no_recompute:
                    self.recompute(block)

        self.maxpool2d = nn.MaxPool2d(
            kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size)
        )
        self.compress_maxpool2d = nn.MaxPool2d(kernel_size=self.compress_kv_factor, stride=self.compress_kv_factor)

    def make_position(self, b, t, use_image_num, h, w):
        pos_hw = self.position_getter_2d(b * (t + use_image_num), h, w)  # fake_b = b*(t+use_image_num)
        pos_t = self.position_getter_1d(b * h * w, t)  # fake_b = b*h*w
        return pos_hw, pos_t

    def make_attn_mask(self, attention_mask, frame, dtype):
        # attention_mask = rearrange(attention_mask, 'b t h w -> (b t) 1 (h w)')
        b, t, h, w = attention_mask.shape
        attention_mask = attention_mask.reshape(b * t, h * w).unsqueeze(1)
        # assume that mask is expressed as:
        #   (1 = keep,      0 = discard)
        attention_mask = attention_mask.to(self.dtype)
        return attention_mask

    def vae_to_diff_mask(self, attention_mask, use_image_num):
        dtype = attention_mask.dtype
        # b, t+use_image_num, h, w, assume t as channel
        # this version do not use 3d patch embedding
        attention_mask = self.maxpool2d(attention_mask)
        attention_mask = attention_mask.bool().to(dtype)
        return attention_mask

    def construct(
        self,
        hidden_states: ms.Tensor,
        timestep: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        added_cond_kwargs: Dict[str, ms.Tensor] = None,
        class_labels: Optional[ms.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch size, num latent pixels)` if discrete, \
                `ms.Tensor` of shape `(batch size, frame, channel, height, width)` if continuous): Input `hidden_states`.
            encoder_hidden_states ( `ms.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `ms.Tensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `ms.Tensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `ms.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `ms.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
        Returns:
           a `tuple` where the first element is the sample tensor.
        """
        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num  # 20-4=16
        # b c f h w -> (b f) c h w
        hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(
            input_batch_size * (frame + use_image_num), c, h, w
        )
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is None:
            attention_mask = ops.ones((input_batch_size, frame + use_image_num, h, w), dtype=hidden_states.dtype)
        attention_mask = self.vae_to_diff_mask(attention_mask, use_image_num)
        dtype = attention_mask.dtype
        attention_mask_compress = self.compress_maxpool2d(attention_mask)
        attention_mask_compress = attention_mask_compress.to(dtype)

        attention_mask = self.make_attn_mask(attention_mask, frame, hidden_states.dtype)
        attention_mask_compress = self.make_attn_mask(attention_mask_compress, frame, hidden_states.dtype)

        # 1 + 4, 1 -> video condition, 4 -> image condition
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            # b 1 l -> (b f) 1 l
            encoder_attention_mask = encoder_attention_mask.repeat_interleave(frame, dim=0)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = encoder_attention_mask_video.repeat_interleave(frame, dim=1)
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = ops.cat([encoder_attention_mask_video, encoder_attention_mask_image], axis=1)
            # b n l -> (b n) l
            encoder_attention_mask = encoder_attention_mask.view(-1, encoder_attention_mask.shape[-1]).unsqueeze(1)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)

        # # Retrieve lora scale.
        # lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        assert self.is_input_patches, "Only support input patches now!"
        # if self.is_input_patches:  # here
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hw = (height, width)
        num_patches = height * width

        hidden_states = self.pos_embed(hidden_states)  # alrady add positional embeddings

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            # batch_size = hidden_states.shape[0]
            batch_size = input_batch_size
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
            )
        else:
            embedded_timestep = None

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # 3 120 1152

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                # b 1 t d -> b (1 f) t d
                encoder_hidden_states_video = encoder_hidden_states_video.repeat_interleave(frame, dim=1)
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = ops.cat([encoder_hidden_states_video, encoder_hidden_states_image], axis=1)
                # b f t d -> (b f) t d
                encoder_hidden_states_spatial = encoder_hidden_states.view(
                    -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
                )
            else:
                # b t d -> (b f) t d
                encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(frame, dim=0)
        else:
            encoder_hidden_states_spatial = encoder_hidden_states.repeat_interleave(frame, dim=0)  # for graph mode

        # prepare timesteps for spatial and temporal block
        # b d -> (b f) d
        timestep_spatial = timestep.repeat_interleave(frame + use_image_num, dim=0)
        # b d -> (b p) d
        timestep_temp = timestep.repeat_interleave(num_patches, dim=0)

        # BS H -> S B H
        if get_sequence_parallel_state():
            timestep_temp = timestep_temp.view(input_batch_size * num_patches, 6, -1).swapaxes(0, 1).contiguous()
            # f, h -> f, 1, h
            temp_pos_embed = self.temp_pos_embed[self.temp_pos_st : self.temp_pos_ed].unsqueeze(1)
        else:
            temp_pos_embed = self.temp_pos_embed[: self.video_length].unsqueeze(0)
            if temp_pos_embed.shape[1] != frame:
                temp_pos_embed = ops.pad(
                    temp_pos_embed, (0, 0, 0, frame - temp_pos_embed.shape[1]), mode="constant", value=0
                )

        temp_attention_mask = None

        pos_hw, pos_t = None, None
        if self.use_rope:
            pos_hw, pos_t = self.make_position(input_batch_size, frame, use_image_num, height, width)

        for i, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                class_labels,
                cross_attention_kwargs,
                attention_mask_compress if i >= self.num_layers // 2 else attention_mask,  # (b*t 1 h*w)
                encoder_hidden_states_spatial,
                timestep_spatial,
                timestep_temp,
                encoder_attention_mask,
                use_image_num,
                input_batch_size,
                frame,
                enable_temporal_attentions,
                pos_hw=pos_hw,
                pos_t=pos_t,
                hw=hw,
                num_patches=num_patches,
                temp_pos_embed=temp_pos_embed,
                temp_attention_mask=temp_attention_mask,
            )

        # if self.is_input_patches:
        if self.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(
                timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
            shift, scale = self.proj_out_1(ops.silu(conditioning)).chunk(2, axis=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            # b d -> (b f) d
            assert embedded_timestep is not None, "embedded_timestep is expected to be not None"
            embedded_timestep = embedded_timestep.repeat_interleave(frame + use_image_num, dim=0)
            shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, axis=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)

        hidden_states = hidden_states.reshape(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
        # nhwpqc->nchpwq
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
        # (b f) c h w -> b c f h w
        output = output.view(
            input_batch_size, frame + use_image_num, output.shape[-3], output.shape[-2], output.shape[-1]
        )
        output = output.permute(0, 2, 1, 3, 4)
        return output

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model = cls.from_config(config, **kwargs)
        return model

    @classmethod
    def from_pretrained(cls, pretrained_model_path, subfolder=None, checkpoint_path=None, **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, "config.json")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model = cls.from_config(config, **kwargs)
        if checkpoint_path is None or len(checkpoint_path) == 0:
            # search for ckpt under pretrained_model_path
            ckpt_paths = glob.glob(os.path.join(pretrained_model_path, "*.ckpt"))
            assert len(ckpt_paths) == 1, f"Expect to find one checkpoint file under {pretrained_model_path}"
            f", but found {len(ckpt_paths)} files that end with `.ckpt`"
            ckpt = ckpt_paths[0]
        else:
            ckpt = checkpoint_path
        logger.info(f"Restored from ckpt {ckpt}")
        model.load_from_checkpoint(ckpt)

        return model

    def construct_with_cfg(self, x, timestep, class_labels=None, cfg_scale=7.0, attention_mask=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, timestep, class_labels=class_labels, attention_mask=attention_mask)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=2)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def load_from_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found. No checkpoint loaded!!")
        else:
            sd = ms.load_checkpoint(ckpt_path)
            # filter 'network.' prefix and ignore 'temp_pos_embed'
            rm_prefix = ["network."]
            all_pnames = list(sd.keys())
            for pname in all_pnames:
                if "temp_pos_embed" in pname:
                    sd.pop(pname)
                    continue
                for pre in rm_prefix:
                    if pname.startswith(pre):
                        new_pname = pname.replace(pre, "")
                        sd[new_pname] = sd.pop(pname)

            m, u = ms.load_param_into_net(self, sd)
            print("net param not load: ", m, len(m))
            print("ckpt param not load: ", u, len(u))


def LatteT2V_XL_122(**kwargs):
    return LatteT2V(
        num_layers=28,
        attention_head_dim=72,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=1152,
        **kwargs,
    )


Latte_models = {
    "LatteT2V-XL/122": LatteT2V_XL_122,
}
