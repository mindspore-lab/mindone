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
from gm.util import default, exists, instantiate_from_config

import mindspore.nn as nn
import mindspore.ops as ops


class ControlnetUnetModel(UNetModel):
    def __init__(self, control_stage_config, guess_mode=False, strength=1.0, sd_locked=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if sd_locked:
            for param in self.get_parameters():
                param.requires_grad = False

        # add controlnet init
        self.controlnet = instantiate_from_config(control_stage_config)
        self.control_scales = (
            [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)
        )

    def construct(self, x, timesteps=None, context=None, y=None, control=None, only_mid_control=False, **kwargs):
        """
        x: latent image in shape [bs, z, H//4, W//4]
        timesteps: in shape [bs]
        context: text embedding [bs, seq_len, f]
        control: control signal [bs, 3, H, W]
        """
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
                # add encoded hint with latent image encoded projected with conv2d
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
        dropout=0.0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
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
        spatial_transformer_attn_type="vanilla",
        adm_in_channels=None,
        transformer_depth_middle=None,
        use_recompute=False,
    ):
        super().__init__()
        from omegaconf.listconfig import ListConfig

        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert num_heads != -1, "Either num_heads or num_head_channels has to be set"

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.hint_channels = hint_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        elif isinstance(transformer_depth, ListConfig):
            transformer_depth = list(transformer_depth)
        transformer_depth_middle = default(transformer_depth_middle, transformer_depth[-1])

        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        # self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )  # todo: convert to warning

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.dims = dims
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4

        self.time_embed = nn.SequentialCell(
            [
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            ]
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
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1, pad_mode="pad"))]
        )

        self.zero_convs = nn.CellList([self.make_zero_conv(model_channels)])

        self.input_hint_block = nn.CellList(
            [
                conv_nd(dims, hint_channels, 16, 3, padding=1, has_bias=True, pad_mode="pad"),
                nn.SiLU(),
                conv_nd(dims, 16, 16, 3, padding=1, has_bias=True, pad_mode="pad"),
                nn.SiLU(),
                conv_nd(dims, 16, 32, 3, padding=1, stride=2, has_bias=True, pad_mode="pad"),
                nn.SiLU(),
                conv_nd(dims, 32, 32, 3, padding=1, has_bias=True, pad_mode="pad"),
                nn.SiLU(),
                conv_nd(dims, 32, 96, 3, padding=1, stride=2, has_bias=True, pad_mode="pad"),
                nn.SiLU(),
                conv_nd(dims, 96, 96, 3, padding=1, has_bias=True, pad_mode="pad"),
                nn.SiLU(),
                conv_nd(dims, 96, 256, 3, padding=1, stride=2, has_bias=True, pad_mode="pad"),
                nn.SiLU(),
                zero_module(conv_nd(dims, 256, model_channels, 3, padding=1, has_bias=True, pad_mode="pad")),
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
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
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                                attn_type=spatial_transformer_attn_type,
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
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
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
            # num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            )
            if not use_spatial_transformer
            else SpatialTransformer(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
                attn_type=spatial_transformer_attn_type,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self.middle_block_out = self.make_zero_conv(ch)
        self._feature_size += ch

        if use_recompute:
            self.recompute_strategy_v1()

    def make_zero_conv(self, channels):
        return zero_module(
            TimestepEmbedSequential(conv_nd(self.dims, channels, channels, 1, padding=0, has_bias=True, pad_mode="pad"))
        )

    def recompute_strategy_v1(self):
        # embed
        self.time_embed.recompute()
        self.label_emb.recompute()

        # input blocks
        self.input_blocks[4][0].recompute()  # 4
        self.input_blocks[5][0].recompute()  # 5
        self.input_blocks[7][0].recompute()  # 7
        self.input_blocks[8][0].recompute()  # 8

        # middle block
        self.middle_block[0].recompute()
        self.middle_block[1].recompute()

        print("Turn on recompute with StrategyV1.")
