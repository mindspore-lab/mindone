import logging

from gm.modules.attention import SpatialTransformer
from gm.modules.diffusionmodules.openaimodel import (
    AttentionBlock,
    Downsample,
    ResBlock,
    Timestep,
    TimestepEmbedSequential,
    UNetModel,
)
from gm.modules.diffusionmodules.util import conv_nd, linear, timestep_embedding, zero_module
from gm.util import exists, instantiate_from_config

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

_logger = logging.getLogger(__name__)


class ControlnetUnetModel(UNetModel):
    def __init__(self, control_stage_config, guess_mode=False, strength=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.controlnet = instantiate_from_config(control_stage_config)
        self.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )

    def construct(self, x, timesteps=None, context=None, y=None, control=None, only_mid_control=False, **kwargs):
        hs = []

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        emb_c = self.controlnet.time_embed(t_emb)

        if self.num_classes is not None:
            emb = emb + self.label_emb(y)
            emb_c = emb_c + self.controlnet.label_emb(y)

        if control is not None:
            guided_hint = control
            for cell in self.controlnet.input_hint_block:
                guided_hint = cell(guided_hint)
        else:
            guided_hint = None

        control_list = []

        h_c = x
        h = x

        for c_celllist, celllist, zero_convs in zip(
            self.controlnet.input_blocks, self.input_blocks, self.controlnet.zero_convs
        ):
            if control is not None:
                h_c = c_celllist(h_c, emb_c, context)
                if guided_hint is not None:
                    h_c += guided_hint
                    guided_hint = None
                control_list.append(zero_convs(h_c, emb_c, context))

            h = celllist(h, emb, context)
            hs.append(h)

        if control is not None:
            h_c = self.controlnet.middle_block(h_c, emb_c, context)

        h = self.middle_block(h, emb, context)

        if control is not None:
            control_list.append(self.controlnet.middle_block_out(h_c, emb_c, context))
            control_list = [c * scale for c, scale in zip(control_list, self.control_scales)]

        control_index = -1
        if control_list:
            h = h + control_list[control_index]
            control_index -= 1

        hs_index = -1
        for celllist in self.output_blocks:
            if only_mid_control or len(control_list) == 0:
                h = ops.concat([h, hs[hs_index]], axis=1)
            else:
                h = ops.concat([h, hs[hs_index] + control_list[control_index]], axis=1)
            hs_index -= 1
            control_index -= 1
            h = celllist(h, emb, context)

        return self.out(h)


class ControlNet(nn.Cell):
    def __init__(
        self,
        in_channels,
        model_channels,
        hint_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        # use_checkpoint=False,
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
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
        # enable_flash_attention=False,
        # cross_frame_attention=False,
        # unet_chunk_size=2,
    ):
        super().__init__()

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.dims = dims
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        self.num_classes = num_classes

        time_embed_dim = model_channels * 4
        self.time_embed = nn.SequentialCell(
            linear(model_channels, time_embed_dim, dtype=self.dtype),
            nn.SiLU().to_float(self.dtype),
            linear(time_embed_dim, time_embed_dim, dtype=self.dtype),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Dense(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.SequentialCell(
                    [
                        Timestep(model_channels),
                        nn.SequentialCell(
                            [
                                linear(model_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        ),
                    ]
                )
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.SequentialCell(
                    [
                        nn.SequentialCell(
                            [
                                linear(adm_in_channels, time_embed_dim),
                                nn.SiLU(),
                                linear(time_embed_dim, time_embed_dim),
                            ]
                        )
                    ]
                )
            else:
                raise ValueError()

        self.input_blocks = nn.CellList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(
                        self.dtype
                    )
                )
            ]
        )

        self.zero_convs = nn.CellList([self.make_zero_conv(model_channels)])

        self.input_hint_block = nn.CellList(
            [
                conv_nd(dims, hint_channels, 16, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                nn.SiLU().to_float(self.dtype),
                conv_nd(dims, 16, 16, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                nn.SiLU().to_float(self.dtype),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
                nn.SiLU().to_float(self.dtype),
                conv_nd(dims, 32, 32, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                nn.SiLU().to_float(self.dtype),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
                nn.SiLU().to_float(self.dtype),
                conv_nd(dims, 96, 96, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype),
                nn.SiLU().to_float(self.dtype),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2, has_bias=True, pad_mode="pad").to_float(self.dtype),
                nn.SiLU().to_float(self.dtype),
                zero_module(
                    conv_nd(dims, 256, model_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(self.dtype)
                ),
            ]
        )

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = nn.CellList(
                    [
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            # use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            # dtype=self.dtype,
                        )
                    ]
                )
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

                    if not exists(num_attention_blocks) or _ < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                # use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth[level],
                                context_dim=context_dim,
                                # use_checkpoint=use_checkpoint,
                                # dtype=self.dtype,
                                dropout=self.dropout,
                                use_linear=use_linear_in_transformer,
                                # enable_flash_attention=enable_flash_attention,
                                # cross_frame_attention=cross_frame_attention,
                                # unet_chunk_size=unet_chunk_size,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.zero_convs.append(self.make_zero_conv(ch))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=dims,
                            # use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            # dtype=self.dtype,
                        )
                        if resblock_updown
                        else Downsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                self.zero_convs.append(self.make_zero_conv(ch))
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

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                self.dropout,
                dims=dims,
                # use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                # dtype=self.dtype,
            ),
            AttentionBlock(
                ch,
                # use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth[level],
                context_dim=context_dim,
                # use_checkpoint=use_checkpoint,
                # dtype=self.dtype,
                dropout=self.dropout,
                use_linear=use_linear_in_transformer,
                # enable_flash_attention=enable_flash_attention,
                # cross_frame_attention=cross_frame_attention,
                # unet_chunk_size=unet_chunk_size,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                self.dropout,
                dims=dims,
                # use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                # dtype=self.dtype,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

    def make_zero_conv(self, channels):
        return zero_module(
            TimestepEmbedSequential(
                conv_nd(self.dims, channels, channels, 1, padding=0, has_bias=True, pad_mode="pad").to_float(self.dtype)
            )
        )

    def construct(self, x, hint, timesteps, context, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        guided_hint = hint
        for cell in self.input_hint_block:
            guided_hint = cell(guided_hint)

        outs = []

        h = x
        for celllist, zero_conv in zip(self.input_blocks, self.zero_convs):
            for cell in celllist:
                h = cell(h, emb, context)
            if guided_hint is not None:
                h += guided_hint
                guided_hint = None
            outs.append(zero_conv(h, emb, context))
        for module in self.middle_block:
            h = module(h, emb, context)

        outs.append(self.middle_block_out(h, emb, context))

        return outs
