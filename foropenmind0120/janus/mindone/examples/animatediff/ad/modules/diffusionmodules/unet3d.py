# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import logging

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from mindone.utils.amp import auto_mixed_precision

from ..attention import SpatialTransformer
from .motion_module import VanillaTemporalModule, get_motion_module
from .openaimodel import AttentionBlock, Downsample, ResBlock, Upsample
from .util import conv_nd, linear, normalization, timestep_embedding, zero_module

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


class UNet3DModel(nn.Cell):
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
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        use_linear_in_transformer=False,
        enable_flash_attention=False,
        unet_chunk_size=2,
        adm_in_channels=None,
        use_recompute=False,
        recompute_strategy="down_up",
        # Additional
        use_inflated_groupnorm=True,  # diff, default is to use in mm-v2, which is more reasonable.
        use_motion_module=False,
        motion_module_resolutions=(1, 2, 4, 8),  # used to identify which level to be injected with Motion Module
        motion_module_mid_block=False,
        motion_module_decoder_only=False,
        motion_module_type=None,  # default:
        motion_module_kwargs={},  #
        unet_use_cross_frame_attention=None,
        unet_use_temporal_attention=None,
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
            print("WARNING: not using motion module")

        self.norm_in_5d = not use_inflated_groupnorm

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

        # middle block, add 1 MM
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

        # up blocks
        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:  # run 4 times
            for i in range(num_res_blocks + 1):  # run 3 times
                ich = input_block_chans.pop()
                layers = nn.CellList(
                    [
                        ResBlock(
                            ch + ich,
                            time_embed_dim,
                            self.dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
                            norm_in_5d=self.norm_in_5d,
                        )
                    ]
                )
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
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

                # Add MM after ResBlock in UpBlockWithoutAttn (1*3), or after SpatialTransformer in UpBlockWithAttn (3*3)
                if use_motion_module:
                    layers.append(
                        get_motion_module(
                            in_channels=ch,
                            motion_module_type=motion_module_type,
                            motion_module_kwargs=motion_module_kwargs,
                            dtype=self.dtype,
                        )
                    )

                # Upsample except for the last level
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                            norm_in_5d=self.norm_in_5d,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)
                    )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

        self.conv_norm_out = normalization(ch, norm_in_5d=self.norm_in_5d)

        self.out = nn.SequentialCell(
            # normalization(ch),
            nn.SiLU().to_float(self.dtype),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(
                    self.dtype
                )
            ),
        )

        self.cat = ops.Concat(axis=1)

        # TODO: optimize where to recompute & fix bug on cell list.
        if use_recompute:
            # print("D--: recompute strategy: ", recompute_strategy)
            if recompute_strategy in ["down_mm", "down_mm_half", "down_blocks"]:
                for iblock in self.input_blocks:
                    if recompute_strategy == "down_blocks":
                        self.recompute(iblock)
                    else:
                        # 12 input blocks
                        for idx, cell in enumerate(iblock, 1):
                            # recompute level 1 blocks (whose activations are very large), i.e. block 2-4
                            if (recompute_strategy == "down_mm_half" and idx <= 4) or (recompute_strategy == "down_mm"):
                                if isinstance(cell, VanillaTemporalModule):
                                    self.recompute(cell)
            elif recompute_strategy == "up_mm":
                for oblock in self.output_blocks:
                    for cell in oblock:
                        if isinstance(cell, VanillaTemporalModule):
                            self.recompute(cell)
            elif recompute_strategy == "down_up":
                for iblock in self.input_blocks:
                    self.recompute(iblock)
                for oblock in self.output_blocks:
                    self.recompute(oblock)
            else:
                raise NotImplementedError

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def set_mm_amp_level(self, amp_level):
        # set motion module precision
        for i, celllist in enumerate(self.input_blocks, 1):
            for cell in celllist:
                if isinstance(cell, VanillaTemporalModule):
                    cell = auto_mixed_precision(cell, amp_level)

        for module in self.middle_block:
            if isinstance(module, VanillaTemporalModule):
                module = auto_mixed_precision(module, amp_level)

        for celllist in self.output_blocks:
            for cell in celllist:
                if isinstance(cell, VanillaTemporalModule):
                    cell = auto_mixed_precision(cell, amp_level)

    def construct(
        self, x, timesteps=None, context=None, y=None, features_adapter: list = None, append_to_context=None, **kwargs
    ):
        """
        Apply the model to an input batch.
        :param x: (b c f h w), an [N x C x ...] Tensor of inputs.
        :param timesteps: (b,), a 1-D batch of timesteps.
        :param context: (b max_len_tokens dim_ftr), conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: (b c f h w), an [N x C x ...] Tensor of outputs.
        """
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

        # 0. rearrange inputs to (b*f, ...) for pseudo 3d until we meet temporal transformer (i.e. motion module)
        B, C, F, H, W = x.shape
        # x: (b c f h w) -> (b*f c h w)
        x = rearrange_in(x)
        # time mixed with other embedding: (b dim_emb) -> (b*f dim_emb)
        emb = emb.repeat_interleave(repeats=F, dim=0)
        # context: (b max_length dim_clip) -> (b*f dim_emb)
        context = context.repeat_interleave(repeats=F, dim=0)

        h = x

        # 1. conv_in and downblocks
        adapter_idx = 0
        for i, celllist in enumerate(self.input_blocks, 1):
            for cell in celllist:
                if isinstance(cell, VanillaTemporalModule) or (isinstance(cell, ResBlock) and self.norm_in_5d):
                    h = cell(h, emb, context, video_length=F)
                else:
                    h = cell(h, emb, context)

            if features_adapter and i % 3 == 0:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1

            hs.append(h)

        if features_adapter:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        # 2. middle block
        for module in self.middle_block:
            # h = module(h, emb, context)
            if isinstance(module, VanillaTemporalModule) or (isinstance(module, ResBlock) and self.norm_in_5d):
                h = module(h, emb, context, video_length=F)
            else:
                h = module(h, emb, context)

        # 3. up blocks
        hs_index = -1
        for celllist in self.output_blocks:
            h = self.cat((h, hs[hs_index]))
            for cell in celllist:
                # h = cell(h, emb, context)
                if isinstance(cell, VanillaTemporalModule) or (isinstance(cell, ResBlock) and self.norm_in_5d):
                    h = cell(h, emb, context, video_length=F)
                else:
                    if isinstance(cell, Upsample):
                        _, _, tar_h, tar_w = hs[hs_index - 1].shape
                        target_size = (tar_h, tar_w)
                        h = cell(h, emb, context, target_size)
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
