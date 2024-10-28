import glob
import json
import logging
import os
from typing import Any, Dict, Optional

from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.utils.utils import to_2tuple

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers import __version__
from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.embeddings import PixArtAlphaTextProjection
from mindone.diffusers.models.modeling_utils import ModelMixin, load_state_dict
from mindone.diffusers.models.normalization import AdaLayerNormSingle
from mindone.diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, _add_variant, _get_model_file, deprecate

from .modules import BasicTransformerBlock, LayerNorm, OverlapPatchEmbed2D, OverlapPatchEmbed3D, PatchEmbed2D

logger = logging.getLogger(__name__)


class OpenSoraT2V(ModelMixin, ConfigMixin):
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
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
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
        interpolation_scale_h: float = None,
        interpolation_scale_w: float = None,
        interpolation_scale_t: float = None,
        use_additional_conditions: Optional[bool] = None,
        attention_mode: str = "xformers",
        downsampler: str = None,
        use_recompute=False,
        use_rope: bool = False,
        FA_dtype=ms.bfloat16,
        num_no_recompute: int = 0,
        use_stable_fp32: bool = False,
    ):
        super().__init__()

        # Validate inputs.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif norm_type in ["ada_norm", "ada_norm_zero"] and num_embeds_ada_norm is None:
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )

        # Set some common variables used across the board.
        self.use_rope = use_rope
        self.use_linear_projection = use_linear_projection
        self.interpolation_scale_t = interpolation_scale_t
        self.interpolation_scale_h = interpolation_scale_h
        self.interpolation_scale_w = interpolation_scale_w
        self.downsampler = downsampler
        self.caption_channels = caption_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = use_recompute
        self.config.hidden_size = self.inner_dim
        use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions
        self.use_recompute = use_recompute
        self.FA_dtype = FA_dtype

        # 1. Transformer2DModel can process both standard continuous images of shape\
        #  `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        assert in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            deprecation_message = (
                f"The configuration file of this model: {self.__class__} is outdated. `norm_type` is either not set or"
                " incorrectly set to `'layer_norm'`.Make sure to set `norm_type` to `'ada_norm'` in the config."
                " Please make sure to update the config accordingly as leaving `norm_type` might led to incorrect"
                " results in future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it"
                " would be very nice if you could open a Pull request for the `transformer/config.json` file"
            )
            deprecate("norm_type!=num_embeds_ada_norm", "1.0.0", deprecation_message, standard_warn=False)
            norm_type = "ada_norm"
        # 2. Initialize the right blocks.
        # Initialize the output blocks and other projection blocks when necessary.
        self._init_patched_inputs(norm_type=norm_type)

        if self.use_recompute:
            num_no_recompute = self.config.num_no_recompute
            num_blocks = len(self.transformer_blocks)
            assert num_no_recompute >= 0, "Expect to have num_no_recompute as a positive integer."
            assert (
                num_no_recompute <= num_blocks
            ), "Expect to have num_no_recompute as an integer no greater than the number of blocks,"
            f"but got {num_no_recompute} and {num_blocks}."
            logger.info(f"Excluding {num_no_recompute} blocks from the recomputation list.")
            for bidx, block in enumerate(self.transformer_blocks):
                if bidx < num_blocks - num_no_recompute:
                    self.recompute(block)
        self.silu = nn.SiLU()
        self.maxpool2d = nn.MaxPool2d(
            kernel_size=(self.patch_size, self.patch_size), stride=(self.patch_size, self.patch_size)
        )
        self.max_pool3d = nn.MaxPool3d(
            kernel_size=(self.patch_size_t, self.patch_size, self.patch_size),
            stride=(self.patch_size_t, self.patch_size, self.patch_size),
        )

    # rewrite class method to allow the state dict as input
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        state_dict = kwargs.pop("state_dict", None)  # additional key argument
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        mindspore_dtype = kwargs.pop("mindspore_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )

        # load model
        model_file = None
        if from_flax:
            raise NotImplementedError("loading flax checkpoint in mindspore model is not yet supported.")
        else:
            if state_dict is None:  # edits: only search for model_file if state_dict is not provided
                if use_safetensors:
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            resume_download=resume_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            token=token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            commit_hash=commit_hash,
                        )
                    except IOError as e:
                        if not allow_pickle:
                            raise e
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )

            model = cls.from_config(config, **unused_kwargs)
            if state_dict is None:  # edits: only load model_file if state_dict is None
                state_dict = load_state_dict(model_file, variant=variant)
            model._convert_deprecated_attention_blocks(state_dict)

            model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                model,
                state_dict,
                model_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }

        if mindspore_dtype is not None and not isinstance(mindspore_dtype, ms.Type):
            raise ValueError(
                f"{mindspore_dtype} needs to be of type `ms.Type`, e.g. `ms.float16`, but is {type(mindspore_dtype)}."
            )
        elif mindspore_dtype is not None:
            model = model.to(mindspore_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)
        if output_loading_info:
            return model, loading_info

        return model

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_attention_mask(self, attention_mask):
        if attention_mask is not None:
            if self.config.attention_mode != "math":
                attention_mask = attention_mask.to(ms.bool_)
        return attention_mask

    def _init_patched_inputs(self, norm_type):
        assert self.config.sample_size_t is not None, "OpenSoraT2V over patched input must provide sample_size_t"
        assert self.config.sample_size is not None, "OpenSoraT2V over patched input must provide sample_size"
        # assert not (self.config.sample_size_t == 1 and self.config.patch_size_t == 2), "Image do not need patchfy in t-dim"

        self.num_frames = self.config.sample_size_t
        self.config.sample_size = to_2tuple(self.config.sample_size)
        self.height = self.config.sample_size[0]
        self.width = self.config.sample_size[1]
        self.patch_size_t = self.config.patch_size_t
        self.patch_size = self.config.patch_size
        interpolation_scale_t = (
            ((self.config.sample_size_t - 1) // 16 + 1)
            if self.config.sample_size_t % 2 == 1
            else self.config.sample_size_t / 16
        )
        interpolation_scale_t = (
            self.config.interpolation_scale_t
            if self.config.interpolation_scale_t is not None
            else interpolation_scale_t
        )
        interpolation_scale = (
            self.config.interpolation_scale_h
            if self.config.interpolation_scale_h is not None
            else self.config.sample_size[0] / 30,
            self.config.interpolation_scale_w
            if self.config.interpolation_scale_w is not None
            else self.config.sample_size[1] / 40,
        )

        if self.config.downsampler is not None and len(self.config.downsampler) == 9:
            self.pos_embed = OverlapPatchEmbed3D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale,
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope,
            )
        elif self.config.downsampler is not None and len(self.config.downsampler) == 7:
            self.pos_embed = OverlapPatchEmbed2D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale,
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope,
            )

        else:
            self.pos_embed = PatchEmbed2D(
                num_frames=self.config.sample_size_t,
                height=self.config.sample_size[0],
                width=self.config.sample_size[1],
                patch_size_t=self.config.patch_size_t,
                patch_size=self.config.patch_size,
                in_channels=self.in_channels,
                embed_dim=self.inner_dim,
                interpolation_scale=interpolation_scale,
                interpolation_scale_t=interpolation_scale_t,
                use_abs_pos=not self.config.use_rope,
            )
        interpolation_scale_thw = (interpolation_scale_t, *interpolation_scale)
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                    attention_mode=self.config.attention_mode,
                    FA_dtype=self.config.FA_dtype,
                    downsampler=self.config.downsampler,
                    use_rope=self.config.use_rope,
                    interpolation_scale_thw=interpolation_scale_thw,
                )
                for _ in range(self.config.num_layers)
            ]
        )

        if self.config.norm_type != "ada_norm_single":
            self.norm_out = LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Dense(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Dense(
                self.inner_dim,
                self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels,
            )
        elif self.config.norm_type == "ada_norm_single":
            self.norm_out = LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = ms.Parameter(ops.randn(2, self.inner_dim) / self.inner_dim**0.5)
            self.proj_out = nn.Dense(
                self.inner_dim,
                self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels,
            )

        # PixArt-Alpha blocks.
        self.adaln_single = None
        if self.config.norm_type == "ada_norm_single":
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )

        self.caption_projection = None
        if self.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels, hidden_size=self.inner_dim
            )

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
                a noise tensor
        """
        batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num  # 20-4=16
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
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
        attention_mask_vid, attention_mask_img = None, None
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #   (keep = +0,     discard = -10000.0)
            # b, frame+use_image_num, h, w -> a video with images
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)
            if get_sequence_parallel_state():
                attention_mask_vid = attention_mask[:, : frame * hccl_info.world_size]  # b, frame, h, w
                attention_mask_img = attention_mask[:, frame * hccl_info.world_size :]  # b, use_image_num, h, w

            else:
                attention_mask_vid = attention_mask[:, :frame]  # b, frame, h, w
                attention_mask_img = attention_mask[:, frame:]  # b, use_image_num, h, w

            if attention_mask_vid.numel() > 0:
                if self.patch_size_t - 1 > 0:
                    attention_mask_vid_first_frame = attention_mask_vid[:, :1].repeat(self.patch_size_t - 1, axis=1)
                    attention_mask_vid = ops.cat([attention_mask_vid_first_frame, attention_mask_vid], axis=1)
                attention_mask_vid = attention_mask_vid[:, None, :, :, :]  # b 1 t h w
                attention_mask_vid = self.max_pool3d(attention_mask_vid)
                # b 1 t h w -> (b 1) 1 (t h w)
                attention_mask_vid = attention_mask_vid.reshape(batch_size, 1, -1)
            if attention_mask_img.numel() > 0:
                attention_mask_img = self.maxpool2d(attention_mask_img)
                # b i h w -> (b i) 1 (h w)
                attention_mask_img = attention_mask_img.reshape(batch_size * attention_mask_img.shape[1], 1, -1)
            # do not fill in -10000.0 until MHA
            # attention_mask_vid = (1 - attention_mask_vid.bool().to(self.dtype)) * -10000.0 if attention_mask_vid.numel() > 0 else None
            # attention_mask_img = (1 - attention_mask_img.bool().to(self.dtype)) * -10000.0 if attention_mask_img.numel() > 0 else None

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
                attention_mask_img = attention_mask_vid
                attention_mask_vid = None
        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        encoder_attention_mask_vid, encoder_attention_mask_img = None, None
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)  # (b, l) -> (b, 1, l)
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)

        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)
            # b, 1+use_image_num, l -> a video with images
            # b, 1, l -> only images
            in_t = encoder_attention_mask.shape[1]
            encoder_attention_mask_vid = encoder_attention_mask[:, : in_t - use_image_num]  # b, 1, l
            # b 1 l -> (b 1) 1 l
            encoder_attention_mask_vid = encoder_attention_mask_vid if encoder_attention_mask_vid.numel() > 0 else None
            encoder_attention_mask_img = encoder_attention_mask[:, in_t - use_image_num :]  # b, use_image_num, l
            # b i l -> (b i) 1 l
            encoder_attention_mask_img = (
                encoder_attention_mask_img.reshape(-1, encoder_attention_mask.shape[-1]).unsqeeze(1)
                if encoder_attention_mask_img.numel() > 0
                else None
            )

            if frame == 1 and use_image_num == 0 and not get_sequence_parallel_state():
                encoder_attention_mask_img = encoder_attention_mask_vid
                encoder_attention_mask_vid = None

        attention_mask_vid = self.get_attention_mask(attention_mask_vid)  # use bool mask for FA
        encoder_attention_mask_vid = self.get_attention_mask(encoder_attention_mask_vid)
        attention_mask_img = self.get_attention_mask(attention_mask_img)
        encoder_attention_mask_img = self.get_attention_mask(encoder_attention_mask_img)

        # # Retrieve lora scale.
        # lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        frame = ((frame - 1) // self.patch_size_t + 1) if frame % 2 == 1 else frame // self.patch_size_t  # patchfy
        # print('frame', frame)
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        (
            hidden_states_vid,
            hidden_states_img,
            encoder_hidden_states_vid,
            encoder_hidden_states_img,
            timestep_vid,
            timestep_img,
            embedded_timestep_vid,
            embedded_timestep_img,
        ) = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
        )

        # 2. Blocks
        # BS H -> S B H
        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                # b s h -> s b h
                hidden_states_vid = hidden_states_vid.swapaxes(0, 1).contiguous()
                # b s h -> s b h
                encoder_hidden_states_vid = encoder_hidden_states_vid.swapaxes(0, 1).contiguous()
                timestep_vid = timestep_vid.view(batch_size, 6, -1).swapaxes(0, 1).contiguous()
                # print('timestep_vid', timestep_vid.shape)

        for block in self.transformer_blocks:
            if hidden_states_vid is not None:
                hidden_states_vid = block(
                    hidden_states_vid,
                    attention_mask=attention_mask_vid,
                    encoder_hidden_states=encoder_hidden_states_vid,
                    encoder_attention_mask=encoder_attention_mask_vid,
                    timestep=timestep_vid,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=frame,
                    height=height,
                    width=width,
                )
            if hidden_states_img is not None:
                hidden_states_img = block(
                    hidden_states_img,
                    attention_mask=attention_mask_img,
                    encoder_hidden_states=encoder_hidden_states_img,
                    encoder_attention_mask=encoder_attention_mask_img,
                    timestep=timestep_img,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    frame=1,
                    height=height,
                    width=width,
                )

        if get_sequence_parallel_state():
            if hidden_states_vid is not None:
                # s b h -> b s h
                hidden_states_vid = hidden_states_vid.swapaxes(0, 1).contiguous()

        # 3. Output
        output_vid, output_img = None, None
        if hidden_states_vid is not None:
            output_vid = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_vid,
                timestep=timestep_vid,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_vid,
                num_frames=frame,
                height=height,
                width=width,
            )  # b c t h w
        if hidden_states_img is not None:
            output_img = self._get_output_for_patched_inputs(
                hidden_states=hidden_states_img,
                timestep=timestep_img,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep_img,
                num_frames=1,
                height=height,
                width=width,
            )  # b c 1 h w
            if use_image_num != 0:
                # (b i) c 1 h w -> b c i h w
                _, c, _, h, w = output_img.shape
                output_img = output_img.reshape(-1, use_image_num, c, 1, h, w).swapaxes(1, 2).squeeze(3)
        output = None
        if output_vid is not None and output_img is not None:
            output = ops.cat([output_vid, output_img], axis=2)
        elif output_vid is not None:
            output = output_vid
        elif output_img is not None:
            output = output_img
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
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    @classmethod
    def load_from_checkpoint(cls, model, ckpt_path):
        if os.path.isdir(ckpt_path) or ckpt_path.endswith(".safetensors"):
            return cls.load_from_safetensors(model, ckpt_path)
        elif ckpt_path.endswith(".ckpt"):
            return cls.load_from_ms_checkpoint(ckpt_path)
        else:
            raise ValueError("Only support safetensors pretrained ckpt or MindSpore pretrained ckpt!")

    @classmethod
    def load_from_safetensors(cls, model, ckpt_path):
        if os.path.isdir(ckpt_path):
            ckpts = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
            n_ckpt = len(ckpts)
            assert (
                n_ckpt == 1
            ), f"Expect to find only one safetenesors file under {ckpt_path}, but found {n_ckpt} .safetensors files."
            model_file = ckpts[0]
            pretrained_model_name_or_path = ckpt_path
        elif ckpt_path.endswith(".safetensors"):
            model_file = ckpt_path
            pretrained_model_name_or_path = os.path.dirname(ckpt_path)
        state_dict = load_state_dict(model_file, variant=None)
        model._convert_deprecated_attention_blocks(state_dict)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            model_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=False,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }
        logger.info(loading_info)
        return model

    @classmethod
    def load_from_ms_checkpoint(self, model, ckpt_path):
        sd = ms.load_checkpoint(ckpt_path)
        # filter 'network.' prefix
        rm_prefix = ["network."]
        all_pnames = list(sd.keys())
        for pname in all_pnames:
            for pre in rm_prefix:
                if pname.startswith(pre):
                    new_pname = pname.replace(pre, "")
                    sd[new_pname] = sd.pop(pname)

        m, u = ms.load_param_into_net(model, sd)
        print("net param not load: ", m, len(m))
        print("ckpt param not load: ", u, len(u))
        return model

    def _operate_on_patched_inputs(
        self, hidden_states, encoder_hidden_states, timestep, added_cond_kwargs, batch_size, frame, use_image_num
    ):
        # batch_size = hidden_states.shape[0]
        hidden_states_vid, hidden_states_img = self.pos_embed(hidden_states.to(self.dtype), frame)
        timestep_vid, timestep_img = None, None
        embedded_timestep_vid, embedded_timestep_img = None, None
        encoder_hidden_states_vid, encoder_hidden_states_img = None, None

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
            )  # b 6d, b d
            if hidden_states_vid is None:
                timestep_img = timestep
                embedded_timestep_img = embedded_timestep
            else:
                timestep_vid = timestep
                embedded_timestep_vid = embedded_timestep
                if hidden_states_img is not None:
                    # b d -> (b i) d
                    timestep_img = timestep.repeat_interleave(use_image_num, dim=0).contiguous()
                    # b d -> (b i) d
                    embedded_timestep_img = embedded_timestep_img.repeat_interleave(use_image_num, dim=0).contiguous()
        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(
                encoder_hidden_states
            )  # b, 1+use_image_num, l, d or b, 1, l, d
            if hidden_states_vid is None:
                # b 1 l d -> (b 1) l d
                encoder_hidden_states_img = encoder_hidden_states.reshape(
                    -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
                )
            else:
                # b 1 l d -> (b 1) l d
                encoder_hidden_states_vid = encoder_hidden_states[:, :1].reshape(
                    -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
                )
                if hidden_states_img is not None:
                    encoder_hidden_states_img = encoder_hidden_states[:, 1:].reshape(
                        -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
                    )

        return (
            hidden_states_vid,
            hidden_states_img,
            encoder_hidden_states_vid,
            encoder_hidden_states_img,
            timestep_vid,
            timestep_img,
            embedded_timestep_vid,
            embedded_timestep_img,
        )

    def _get_output_for_patched_inputs(
        self, hidden_states, timestep, class_labels, embedded_timestep, num_frames, height=None, width=None
    ):
        # import ipdb;ipdb.set_trace()
        if self.config.norm_type != "ada_norm_single":
            conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels, hidden_dtype=self.dtype)
            shift, scale = mint.chunk(self.proj_out_1(self.silu(conditioning)), 2, dim=1)
            hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            hidden_states = self.proj_out_2(hidden_states)
        elif self.config.norm_type == "ada_norm_single":
            shift, scale = mint.chunk(self.scale_shift_table[None] + embedded_timestep[:, None], 2, dim=1)
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1) if hidden_states.shape[1] == 1 else hidden_states

        # unpatchify
        if self.adaln_single is None:
            height = width = int(hidden_states.shape[1] ** 0.5)
        hidden_states = hidden_states.reshape(
            (
                -1,
                num_frames,
                height,
                width,
                self.patch_size_t,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            )
        )
        # nthwopqc->nctohpwq
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(
            (
                -1,
                self.out_channels,
                num_frames * self.patch_size_t,
                height * self.patch_size,
                width * self.patch_size,
            )
        )

        return output


def OpenSoraT2V_S_122(**kwargs):
    return OpenSoraT2V(
        num_layers=28,
        attention_head_dim=96,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=1536,
        **kwargs,
    )


def OpenSoraT2V_B_122(**kwargs):
    return OpenSoraT2V(
        num_layers=32,
        attention_head_dim=96,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=1920,
        **kwargs,
    )


def OpenSoraT2V_L_122(**kwargs):
    return OpenSoraT2V(
        num_layers=40,
        attention_head_dim=128,
        num_attention_heads=16,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=2048,
        **kwargs,
    )


def OpenSoraT2V_ROPE_L_122(**kwargs):
    return OpenSoraT2V(
        num_layers=32,
        attention_head_dim=96,
        num_attention_heads=24,
        patch_size_t=1,
        patch_size=2,
        norm_type="ada_norm_single",
        caption_channels=4096,
        cross_attention_dim=2304,
        **kwargs,
    )


OpenSora_models = {
    "OpenSoraT2V-S/122": OpenSoraT2V_S_122,
    "OpenSoraT2V-B/122": OpenSoraT2V_B_122,
    "OpenSoraT2V-L/122": OpenSoraT2V_L_122,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V_ROPE_L_122,
}

OpenSora_models_class = {
    "OpenSoraT2V-S/122": OpenSoraT2V,
    "OpenSoraT2V-B/122": OpenSoraT2V,
    "OpenSoraT2V-L/122": OpenSoraT2V,
    "OpenSoraT2V-ROPE-L/122": OpenSoraT2V,
}

if __name__ == "__main__":
    from opensora.models.causalvideovae import ae_stride_config

    args = type(
        "args",
        (),
        {
            "ae": "CausalVAEModel_D4_4x8x8",
            "use_rope": True,
            "model_max_length": 512,
            "max_height": 320,
            "max_width": 240,
            "num_frames": 1,
            "use_image_num": 0,
            "interpolation_scale_t": 1,
            "interpolation_scale_h": 1,
            "interpolation_scale_w": 1,
        },
    )
    b = 16
    c = 8
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    num_frames = (args.num_frames - 1) // ae_stride_t + 1

    model = OpenSoraT2V_ROPE_L_122(
        in_channels=c,
        out_channels=c,
        sample_size=latent_size,
        sample_size_t=num_frames,
        activation_fn="gelu-approximate",
        attention_bias=True,
        attention_type="default",
        double_self_attention=False,
        norm_elementwise_affine=False,
        norm_eps=1e-06,
        norm_num_groups=32,
        num_vector_embeds=None,
        only_cross_attention=False,
        upcast_attention=False,
        use_linear_projection=False,
        use_additional_conditions=False,
        downsampler=None,
        interpolation_scale_t=args.interpolation_scale_t,
        interpolation_scale_h=args.interpolation_scale_h,
        interpolation_scale_w=args.interpolation_scale_w,
        use_rope=args.use_rope,
    )

    try:
        path = "PixArt-Alpha-XL-2-512.safetensors"
        from safetensors.torch import load_file as safe_load

        ckpt = safe_load(path, device="cpu")
        # import ipdb;ipdb.set_trace()
        if (
            ckpt["pos_embed.proj.weight"].shape != model.pos_embed.proj.weight.shape
            and ckpt["pos_embed.proj.weight"].ndim == 4
        ):
            repeat = model.pos_embed.proj.weight.shape[2]
            ckpt["pos_embed.proj.weight"] = ckpt["pos_embed.proj.weight"].unsqueeze(2).repeat(repeat, axis=2) / float(
                repeat
            )
            del ckpt["proj_out.weight"], ckpt["proj_out.bias"]
        msg = model.load_state_dict(ckpt, strict=False)
        print(msg)
    except Exception as e:
        print(e)
    print(model)
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")
    # import sys;sys.exit()
    x = ops.randn(
        b,
        c,
        1 + (args.num_frames - 1) // ae_stride_t + args.use_image_num,
        args.max_height // ae_stride_h,
        args.max_width // ae_stride_w,
    )
    cond = ops.randn(b, 1 + args.use_image_num, args.model_max_length, cond_c)
    attn_mask = ops.randint(
        0,
        2,
        (
            b,
            1 + (args.num_frames - 1) // ae_stride_t + args.use_image_num,
            args.max_height // ae_stride_h,
            args.max_width // ae_stride_w,
        ),
    )  # B L or B 1+num_images L
    cond_mask = ops.randint(0, 2, (b, 1 + args.use_image_num, args.model_max_length))  # B L or B 1+num_images L
    timestep = ops.randint(0, 1000, (b,))
    model_kwargs = dict(
        hidden_states=x,
        encoder_hidden_states=cond,
        attention_mask=attn_mask,
        encoder_attention_mask=cond_mask,
        use_image_num=args.use_image_num,
        timestep=timestep,
    )
    model.set_train(False)
    output = model(**model_kwargs)
    print(output[0].shape)
