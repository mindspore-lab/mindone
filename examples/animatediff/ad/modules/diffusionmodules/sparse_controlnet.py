import logging
from typing import Optional, Tuple

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindone.utils.amp import auto_mixed_precision

from ..attention import SpatialTransformer
from .motion_module import VanillaTemporalModule, get_motion_module
from .openaimodel import AttentionBlock, Downsample, ResBlock
from .unet3d import UNet3DModel
from .util import conv_nd, linear, timestep_embedding, zero_module

_logger = logging.getLogger(__name__)


def rearrange_in(x):
    """
    reshape x from (b c f h w) -> (b*f c h w)
    temporal 5d to spatial 4d
    """
    # b c f h w -> b f c h w -> (b f) c h w
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    x = ops.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4]))
    return x


def rearrange_out(x, f):
    """
    reshape x from (b*f c h w) -> (b c f h w)
    spatial 4d to temporal 5d
    f: num frames
    """
    # (b f) c h w -> b f c h w -> b c f h w
    x = ops.reshape(x, (x.shape[0] // f, f, x.shape[1], x.shape[2], x.shape[3]))
    x = ops.transpose(x, (0, 2, 1, 3, 4))
    return x


class SparseControlNetConditioningEmbedding(nn.Cell):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 96, 256),
        dims: int = 2,
        dtype=ms.float32,
    ):
        super().__init__()
        self.dtype = dtype

        self.conv_in = conv_nd(
            dims, conditioning_channels, block_out_channels[0], 3, padding=1, has_bias=True, pad_mode="pad"
        ).to_float(self.dtype)

        self.blocks = nn.CellList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(
                conv_nd(dims, channel_in, channel_in, kernel_size=3, padding=1, has_bias=True, pad_mode="pad").to_float(
                    self.dtype
                ),
            )
            self.blocks.append(
                conv_nd(
                    dims,
                    channel_in,
                    channel_out,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    has_bias=True,
                    pad_mode="pad",
                ).to_float(self.dtype),
            )

        self.conv_out = zero_module(
            conv_nd(
                dims,
                block_out_channels[-1],
                conditioning_embedding_channels,
                kernel_size=3,
                padding=1,
                has_bias=True,
                pad_mode="pad",
            ).to_float(self.dtype)
        )

    def construct(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = ops.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = ops.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class SparseControlNetModel(nn.Cell):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor. channels of image feature encoded by vae, i.e. 4 for SD1.5.
    :param model_channels: base channel count for the model. output channels for conv_in, i.e. 320 for SD1.5
    :param out_channels: channels in the output Tensor. i.e., 4 for SD1.5, same as input.
    :param num_res_blocks: number of residual blocks per downsample. 2 for SD1.5
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used. [ 4, 2, 1 ] for SD1.5
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.  [ 1, 2, 4, 4 ] for SD1.5
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    Additional:

    """

    def __init__(
        self,
        image_size,
        conditioning_channels,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        # conditioning
        conditioning_embedding_out_channels: Optional[Tuple[int]] = (16, 32, 96, 256),
        global_pool_conditions: bool = False,
        concate_conditioning_mask: bool = True,
        use_simplified_condition_embedding: bool = False,
        set_noisy_sample_input_to_zero: bool = False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        use_linear_in_transformer=False,
        enable_flash_attention=False,
        cross_frame_attention=False,
        unet_chunk_size=2,
        adm_in_channels=None,
        use_recompute=False,
        # Additional
        use_inflated_groupnorm=True,  # diff, default is to use in mm-v2, which is more reasonable.
        use_motion_module=True,  # TODO: why set it True by default
        motion_module_resolutions=(1, 2, 4, 8),  # used to identify which level to be injected with Motion Module
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,  # default:
        motion_module_kwargs={},  #
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
        guess_mode: bool = False,
        conditioning_scale: float = 1.0,
    ):
        super().__init__()

        # assert (
        #     use_inflated_groupnorm is True
        # ), "Only support use_inflated_groupnorm=True currently, please use configs/inference/inference_v2.yaml for --inference_config"
        if use_motion_module:
            assert unet_use_cross_frame_attention is False, "not support"
            assert unet_use_temporal_attention is False, "not support"
            assert motion_module_type == "Vanilla", "not support"
        else:
            print("D---: WARNING: not using motion module")

        self.norm_in_5d = not use_inflated_groupnorm
        print("Sparse Control Encoder Initializing...")

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if isinstance(context_dim, ListConfig):
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = 1.0 - dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.guess_mode = guess_mode
        self.conditioning_scale = conditioning_scale

        time_embed_dim = model_channels * 4
        self.time_embed = nn.SequentialCell(
            linear(model_channels, time_embed_dim, dtype=self.dtype),
            nn.SiLU().to_float(self.dtype),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim).to_float(self.dtype)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.SequentialCell(
                    nn.SequentialCell(
                        linear(adm_in_channels, time_embed_dim, dtype=self.dtype),
                        nn.SiLU().to_float(self.dtype),
                        linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
                    )
                )

            else:
                raise ValueError("`num_classes` must be an integer or string of 'continuous' or `sequential`")

        self.set_noisy_sample_input_to_zero = set_noisy_sample_input_to_zero
        self.global_pool_conditions = global_pool_conditions

        self.use_simplified_condition_embedding = use_simplified_condition_embedding
        # conditioning
        if concate_conditioning_mask:
            conditioning_channels = conditioning_channels + 1
        self.concate_conditioning_mask = concate_conditioning_mask

        # control net conditioning embedding
        if use_simplified_condition_embedding:
            self.controlnet_cond_embedding = zero_module(
                conv_nd(
                    dims, conditioning_channels, model_channels, 3, padding=1, has_bias=True, pad_mode="pad"
                ).to_float(self.dtype)
            )
        else:
            self.controlnet_cond_embedding = SparseControlNetConditioningEmbedding(
                conditioning_embedding_channels=model_channels,
                block_out_channels=conditioning_embedding_out_channels,
                conditioning_channels=conditioning_channels,
                dtype=self.dtype,
            )
        controlnet_block = zero_module(
            conv_nd(dims, model_channels, model_channels, 1, padding=0, has_bias=True).to_float(self.dtype)
        )
        self.controlnet_input_blocks = nn.CellList([controlnet_block])

        self.input_blocks = nn.CellList(
            [
                nn.CellList(
                    [
                        conv_nd(
                            dims, in_channels, model_channels, 3, padding=1, has_bias=True, pad_mode="pad"
                        ).to_float(self.dtype)
                    ]
                )
            ]
        )
        self._feature_size = model_channels  #
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        # down blocks, add 2*3+2=8 MotionModules
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                # layers: consist of ResBlock-SptailTrans-MM or ResBlock-MM
                layers = nn.CellList(
                    [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
                            norm_in_5d=self.norm_in_5d,
                        )
                    ]
                )
                ch = mult * model_channels  # feature channels are doubled in each CrossAttnDownBlock
                # For the first 3 levels, ds=1, 2, 4, create SpatialTransformer. For 4-th level, skip.
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else SpatialTransformer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth,
                            context_dim=context_dim,
                            use_checkpoint=use_checkpoint,
                            dtype=self.dtype,
                            dropout=self.dropout,
                            use_linear=use_linear_in_transformer,
                            enable_flash_attention=enable_flash_attention,
                            unet_chunk_size=unet_chunk_size,
                        )
                    )

                # add MotionModule 1) after SpatialTransformer in DownBlockWithAttn, 3*2 times, or 2) after ResBlock in DownBlockWithoutAttn, 1*2 time.
                if use_motion_module:
                    layers.append(
                        # TODO: set mm fp32/fp16 independently?
                        get_motion_module(  # return VanillaTemporalModule
                            in_channels=ch,
                            motion_module_type=motion_module_type,
                            motion_module_kwargs=motion_module_kwargs,
                            dtype=self.dtype,
                        )
                    )

                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)

            # for the first 3 levels, add a DownSampler at each level
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    nn.CellList(
                        [
                            ResBlock(
                                ch,
                                time_embed_dim,
                                self.dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                                dtype=self.dtype,
                                norm_in_5d=self.norm_in_5d,
                            )
                        ]
                    )
                    if resblock_updown
                    else nn.CellList([Downsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)])
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

            # controlnet
            for _ in range(num_res_blocks):
                controlnet_block = zero_module(conv_nd(dims, ch, ch, 1, padding=0, has_bias=True).to_float(self.dtype))
                self.controlnet_input_blocks.append(controlnet_block)
            if level != len(channel_mult) - 1:
                controlnet_block = zero_module(conv_nd(dims, ch, ch, 1, padding=0, has_bias=True).to_float(self.dtype))
                self.controlnet_input_blocks.append(controlnet_block)

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        _logger.debug(
            "Attention: output_channels={}, num_heads={}, num_head_channels:{}, dim_head={}".format(
                ch, num_heads, num_head_channels, dim_head
            )
        )
        # controlnet
        controlnet_block = conv_nd(dims, ch, ch, 1, padding=0, has_bias=True).to_float(self.dtype)
        controlnet_block = zero_module(controlnet_block)
        # self.controlnet_middle_block = nn.CellList([controlnet_block])
        self.controlnet_middle_block = controlnet_block

        self.middle_block = nn.CellList(
            [
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                    norm_in_5d=self.norm_in_5d,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else SpatialTransformer(
                    ch,
                    num_heads,
                    dim_head,
                    depth=transformer_depth,
                    context_dim=context_dim,
                    use_checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    dropout=self.dropout,
                    use_linear=use_linear_in_transformer,
                    enable_flash_attention=enable_flash_attention,
                    unet_chunk_size=unet_chunk_size,
                ),
            ]
        )
        # Add MM after SpatialTrans in MiddleBlock, 1
        if use_motion_module and motion_module_mid_block:
            self.middle_block.append(
                get_motion_module(
                    in_channels=ch,
                    motion_module_type=motion_module_type,
                    motion_module_kwargs=motion_module_kwargs,
                    dtype=self.dtype,
                )
            )
        self.middle_block.append(
            ResBlock(
                ch,
                time_embed_dim,
                self.dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                dtype=self.dtype,
                norm_in_5d=self.norm_in_5d,
            )
        )

        self._feature_size += ch

        self.cat = ops.Concat(axis=1)

        # TODO: optimize where to recompute & fix bug on cell list.
        if use_recompute:
            print("D--: recompute: ", use_recompute)
            for iblock in self.input_blocks:
                self.recompute(iblock)
                # mblock.recompute()
        self.correct_param_name()

    def correct_param_name(self):
        # for some reason unknown, the param.name is not matched with name, param in model.parameters_and_names()
        for pname, param in self.parameters_and_names():
            if pname != param.name:
                param.name = pname

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def set_mm_amp_level(self, amp_level):
        # set motion module precision
        print("D--: mm amp level: ", amp_level)
        for i, celllist in enumerate(self.input_blocks, 1):
            for cell in celllist:
                if isinstance(cell, VanillaTemporalModule):
                    cell = auto_mixed_precision(cell, amp_level)

        for module in self.middle_block:
            if isinstance(module, VanillaTemporalModule):
                module = auto_mixed_precision(module, amp_level)

    def construct(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        features_adapter: list = None,
        append_to_context=None,
        controlnet_cond: Optional[ms.Tensor] = None,
        conditioning_mask: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        """
        Apply the model to an input batch.
        :param x: (b c f h w), an [N x C x ...] Tensor of inputs.
        :param timesteps: (b,), a 1-D batch of timesteps.
        :param context: (b max_len_tokens dim_ftr), conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: (b c f h w), an [N x C x ...] Tensor of outputs.
        """

        if self.set_noisy_sample_input_to_zero:
            x = ops.zeros_like(x)
        assert len(x.shape) == 5, f"UNet3D expect x in shape (b c f h w). but got {x.shape}"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # time embedding
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        if append_to_context is not None:
            context = ops.cat([context, append_to_context], axis=1)

        if self.concate_conditioning_mask:
            controlnet_cond = self.cat([controlnet_cond, conditioning_mask])
        # controlnet_cond: (b c f h w) -> (b*f c h w)
        controlnet_cond = rearrange_in(controlnet_cond)
        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)
        # 0. rearrange inputs to (b*f, ...) for pseudo 3d until we meet temporal transformer (i.e. motion module)
        B, C, F, H, W = x.shape
        # x: (b c f h w) -> (b*f c h w)
        x = rearrange_in(x)
        # time mixed with other embedding: (b dim_emb) -> (b*f dim_emb)
        emb = emb.repeat_interleave(repeats=F, dim=0)
        # context: (b max_length dim_clip) -> (b*f dim_emb)
        context = context.repeat_interleave(repeats=F, dim=0)

        h = x

        # 1. conv_in and inputblocks
        input_block_res_samples = []
        adapter_idx = 0
        for i, celllist in enumerate(self.input_blocks, 1):
            for cell in celllist:
                if isinstance(cell, VanillaTemporalModule) or (isinstance(cell, ResBlock) and self.norm_in_5d):
                    h = cell(h, emb, context, video_length=F)
                elif isinstance(cell, conv_nd):
                    h = cell(h, emb, context)  # conv_in
                    h = h + controlnet_cond
                else:
                    h = cell(h, emb, context)

            if features_adapter and i % 3 == 0:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1

            hs.append(h)

        input_block_res_samples = hs
        if features_adapter:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        # 2. middle block
        for module in self.middle_block:
            # h = module(h, emb, context)
            if isinstance(module, VanillaTemporalModule) or (isinstance(module, ResBlock) and self.norm_in_5d):
                h = module(h, emb, context, video_length=F)
            else:
                h = module(h, emb, context)

        # 3. controlnet blocks
        controlnet_input_block_res_samples = ()

        for input_block_res_sample, controlnet_block in zip(input_block_res_samples, self.controlnet_input_blocks):
            input_block_res_sample = controlnet_block(input_block_res_sample)
            controlnet_input_block_res_samples = controlnet_input_block_res_samples + (input_block_res_sample,)

        input_block_res_samples = controlnet_input_block_res_samples

        mid_block_res_sample = self.controlnet_middle_block(h)

        # 6. scaling
        if self.guess_mode and not self.global_pool_conditions:
            scales = ops.logspace(-1, 0, len(input_block_res_samples) + 1)  # 0.1 to 1.0

            scales = scales * self.conditioning_scale
            input_block_res_samples = [sample * scale for sample, scale in zip(input_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            input_block_res_samples = [sample * self.conditioning_scale for sample in input_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * self.conditioning_scale

        if self.global_pool_conditions:
            input_block_res_samples = [
                rearrange_in(ops.mean(rearrange_out(sample, f=F), (2, 3), True)) for sample in input_block_res_samples
            ]
            mid_block_res_sample = rearrange_in(ops.mean(rearrange_out(mid_block_res_sample, f=F), (2, 3), True))

        return (input_block_res_samples, mid_block_res_sample)


class SparseCtrlUNet3D(UNet3DModel):
    def __init__(self, control_additional_config, sd_locked=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sd_locked:
            for param in self.get_parameters():
                param.requires_grad = False

        # the controlnet has all args and kwargs as UNet3DModel, in addition to controlnet related arguments.
        kwargs.update(control_additional_config)
        self.controlnet = SparseControlNetModel(*args, **kwargs)

    def construct(
        self,
        x,
        timesteps=None,
        context=None,
        y=None,
        features_adapter: list = None,
        append_to_context=None,
        controlnet_cond: Optional[ms.Tensor] = None,
        conditioning_mask: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        if self.controlnet.set_noisy_sample_input_to_zero:
            x_c = ops.zeros_like(x)
        else:
            x_c = x
        assert len(x.shape) == 5, f"UNet3D expect x in shape (b c f h w). but got {x.shape}"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # time embedding
        hs = []
        hs_c = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        emb_c = self.controlnet.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)
        if self.controlnet.num_classes is not None:
            assert y.shape[0] == x_c.shape[0]
            emb_c = emb_c + self.controlnet.label_emb(y)

        if append_to_context is not None:
            context = ops.cat([context, append_to_context], axis=1)
        assert controlnet_cond is not None, "The input control must not be None for SparseCtrlUNet3D!"
        if self.controlnet.concate_conditioning_mask:
            controlnet_cond = self.controlnet.cat([controlnet_cond, conditioning_mask])
        # 0. rearrange inputs to (b*f, ...) for pseudo 3d until we meet temporal transformer (i.e. motion module)
        B, C, F, H, W = x.shape
        if controlnet_cond.shape[0] == 1:
            # broadcast to batch size of x
            controlnet_cond = controlnet_cond.repeat_interleave(repeats=B, dim=0)
        controlnet_cond = rearrange_in(controlnet_cond)
        controlnet_cond = self.controlnet.controlnet_cond_embedding(controlnet_cond)
        # x: (b c f h w) -> (b*f c h w)
        x = rearrange_in(x)
        x_c = rearrange_in(x_c)
        # time mixed with other embedding: (b dim_emb) -> (b*f dim_emb)
        emb = emb.repeat_interleave(repeats=F, dim=0)
        emb_c = emb_c.repeat_interleave(repeats=F, dim=0)
        # context: (b max_length dim_clip) -> (b*f dim_emb)
        context = context.repeat_interleave(repeats=F, dim=0)

        h = x
        h_c = x_c

        # 1. conv_in and downblocks
        adapter_idx = 0
        # 1.1 unet input_blocks
        for i, (c_cellist, celllist, zero_convs) in enumerate(
            zip(self.controlnet.input_blocks, self.input_blocks, self.controlnet.controlnet_input_blocks), 1
        ):
            for cell in celllist:
                if isinstance(cell, VanillaTemporalModule) or (isinstance(cell, ResBlock) and self.norm_in_5d):
                    h = cell(h, emb, context, video_length=F)
                else:
                    h = cell(h, emb, context)
            for c_cell in c_cellist:
                if isinstance(c_cell, VanillaTemporalModule) or (
                    isinstance(c_cell, ResBlock) and self.controlnet.norm_in_5d
                ):
                    h_c = c_cell(h_c, emb_c, context, video_length=F)
                elif isinstance(c_cell, conv_nd):
                    h_c = c_cell(h_c, emb_c, context)  # conv_in
                    h_c = h_c + controlnet_cond
                else:
                    h_c = c_cell(h_c, emb_c, context)

            if features_adapter and i % 3 == 0:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1

            hs.append(h)
            hs_c.append(zero_convs(h_c))

        if features_adapter:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        # 2. middle block
        # 2.1 unet middle block
        for module in self.middle_block:
            # h = module(h, emb, context)
            if isinstance(module, VanillaTemporalModule) or (isinstance(module, ResBlock) and self.norm_in_5d):
                h = module(h, emb, context, video_length=F)
            else:
                h = module(h, emb, context)
        # 2.2 sparse control encoder middle block
        for module in self.controlnet.middle_block:
            if isinstance(module, VanillaTemporalModule) or (
                isinstance(module, ResBlock) and self.controlnet.norm_in_5d
            ):
                h_c = module(h_c, emb_c, context, video_length=F)
            else:
                h_c = module(h_c, emb_c, context)

        input_block_res_samples = hs_c
        mid_block_res_sample = self.controlnet.controlnet_middle_block(h_c)

        # 3.2 scaling
        if self.controlnet.guess_mode and not self.controlnet.global_pool_conditions:
            scales = ops.logspace(-1, 0, len(input_block_res_samples) + 1)  # 0.1 to 1.0

            scales = scales * self.controlnet.conditioning_scale
            input_block_res_samples = [sample * scale for sample, scale in zip(input_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]  # last one
        else:
            input_block_res_samples = [
                sample * self.controlnet.conditioning_scale for sample in input_block_res_samples
            ]
            mid_block_res_sample = mid_block_res_sample * self.controlnet.conditioning_scale

        if self.controlnet.global_pool_conditions:
            input_block_res_samples = [
                rearrange_in(ops.mean(rearrange_out(sample, f=F), (2, 3), True)) for sample in input_block_res_samples
            ]
            mid_block_res_sample = rearrange_in(ops.mean(rearrange_out(mid_block_res_sample, f=F), (2, 3), True))

        # support controlnet
        for i, in_res in enumerate(input_block_res_samples):
            hs[i] = hs[i] + in_res  # add with unet input blocks residuals

        h = h + mid_block_res_sample

        # 4. up blocks
        hs_index = -1
        for celllist in self.output_blocks:
            h = self.cat((h, hs[hs_index]))
            for cell in celllist:
                # h = cell(h, emb, context)
                if isinstance(cell, VanillaTemporalModule) or (isinstance(cell, ResBlock) and self.norm_in_5d):
                    h = cell(h, emb, context, video_length=F)
                else:
                    h = cell(h, emb, context)
            hs_index -= 1
        if self.norm_in_5d:
            h = self.conv_norm_out(h, video_length=F)
        else:
            h = self.conv_norm_out(h)

        h = self.out(h)

        # rearrange back: (b*f c h w) -> (b c f h w)
        h = rearrange_out(h, f=F)

        return h
