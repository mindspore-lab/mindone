from typing import List, Optional, Set, Tuple, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from sgm.modules.attention import CrossAttention, FeedForward, MemoryEfficientCrossAttention, SpatialTransformer
from sgm.modules.diffusionmodules.openaimodel import Downsample, ResBlock, Timestep, TimestepBlock, Upsample
from sgm.modules.diffusionmodules.util import AlphaBlender, conv_nd, normalization, timestep_embedding, zero_module

from mindspore import Tensor, nn, ops


class TimestepEmbedSequential(nn.SequentialCell, TimestepBlock):
    def construct(
        self,
        x,
        emb,
        context=None,
        image_only_indicator: Optional[Tensor] = None,
        time_context: Optional[int] = None,
        num_frames: Optional[int] = None,
    ):
        for cell in self.cell_list:
            if isinstance(cell, TimestepBlock) and not isinstance(cell, TemporalResBlock):
                x = cell(x, emb)
            elif isinstance(cell, TemporalResBlock):
                x = cell(x, emb, num_frames, image_only_indicator)
            elif isinstance(cell, TemporalTransformer):
                x = cell(
                    x,
                    context,
                    time_context,
                    num_frames,
                    image_only_indicator,
                )
            elif isinstance(cell, SpatialTransformer):
                x = cell(x, context)
            else:
                x = cell(x)
        return x


class TemporalTransformerBlock(nn.Cell):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "flash-attention": MemoryEfficientCrossAttention,
    }

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        timesteps=None,
        ff_in=False,
        inner_dim=None,
        attn_mode: Literal["vanilla", "flash-attention"] = "vanilla",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        switch_temporal_ca_to_sa=False,
    ):
        super().__init__()

        attn_cls = self.ATTENTION_MODES[attn_mode]

        self.ff_in = ff_in or inner_dim is not None
        if inner_dim is None:
            inner_dim = dim

        assert int(n_heads * d_head) == inner_dim

        self.is_res = inner_dim == dim

        if self.ff_in:
            self.norm_in = nn.LayerNorm([dim], epsilon=1e-5)
            self.ff_in = FeedForward(dim, dim_out=inner_dim, dropout=dropout, glu=gated_ff)

        self.timesteps = timesteps
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(
            query_dim=inner_dim,
            heads=n_heads,
            dim_head=d_head,
            context_dim=context_dim if self.disable_self_attn else None,
            dropout=dropout,
        )

        self.ff = FeedForward(inner_dim, dim_out=dim, dropout=dropout, glu=gated_ff)

        if disable_temporal_crossattention:
            if switch_temporal_ca_to_sa:
                raise ValueError
            else:
                self.attn2 = None
        else:
            self.norm2 = nn.LayerNorm([inner_dim], epsilon=1e-5)
            self.attn2 = attn_cls(
                query_dim=inner_dim,
                context_dim=None if switch_temporal_ca_to_sa else context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
            )

        self.norm1 = nn.LayerNorm([inner_dim], epsilon=1e-5)
        self.norm3 = nn.LayerNorm([inner_dim], epsilon=1e-5)
        self.switch_temporal_ca_to_sa = switch_temporal_ca_to_sa

    def construct(self, x: Tensor, context: Tensor = None, timesteps: Tensor = None) -> Tensor:
        assert self.timesteps or timesteps
        assert not (self.timesteps and timesteps) or self.timesteps == timesteps
        timesteps = self.timesteps or timesteps

        b, s, c = x.shape
        x = x.reshape(-1, timesteps, s, c).swapaxes(1, 2)  # (b t) s c -> b s t c
        x = x.reshape(-1, timesteps, c)  # b s t c -> (b s) t c

        if self.ff_in:
            x_skip = x
            x = self.ff_in(self.norm_in(x))
            if self.is_res:
                x += x_skip

        x = self.attn1(self.norm1(x), context=context if self.disable_self_attn else None) + x

        if self.attn2 is not None:
            x = self.attn2(self.norm2(x), context=None if self.switch_temporal_ca_to_sa else context) + x

        x_skip = x
        x = self.ff(self.norm3(x))
        if self.is_res:
            x += x_skip

        x = x.reshape(-1, s, timesteps, c).swapaxes(1, 2).reshape(-1, s, c)  # (b s) t c -> (b t) s c
        return x


class TemporalTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        use_linear=False,
        context_dim=None,
        use_spatial_context=False,
        timesteps=None,
        merge_strategy: Literal["fixed", "learned", "learned_with_images"] = "fixed",
        merge_factor: float = 0.5,
        time_context_dim=None,
        ff_in=False,
        attn_mode: Literal["vanilla", "flash-attention"] = "vanilla",
        disable_self_attn=False,
        disable_temporal_crossattention=False,
        max_time_embed_period: int = 10000,
    ):
        super().__init__(
            in_channels,
            n_heads,
            d_head,
            depth=depth,
            dropout=dropout,
            attn_type=attn_mode,
            context_dim=context_dim,
            use_linear=use_linear,
            disable_self_attn=disable_self_attn,
        )
        self.max_time_embed_period = max_time_embed_period
        # self._pe = Tensor(_positional_encoding(num_frames, in_channels), dtype=ms_dtype.float32)  # FIXME: check this

        time_mix_d_head = d_head
        n_time_mix_heads = n_heads

        time_mix_inner_dim = int(time_mix_d_head * n_time_mix_heads)

        inner_dim = n_heads * d_head
        if use_spatial_context:
            time_context_dim = context_dim

        self.time_stack = nn.CellList(
            [
                TemporalTransformerBlock(
                    inner_dim,
                    n_time_mix_heads,
                    time_mix_d_head,
                    dropout=dropout,
                    context_dim=time_context_dim,
                    timesteps=timesteps,
                    ff_in=ff_in,
                    inner_dim=time_mix_inner_dim,
                    attn_mode=attn_mode,
                    disable_self_attn=disable_self_attn,
                    disable_temporal_crossattention=disable_temporal_crossattention,
                )
                for _ in range(depth)
            ]
        )

        assert len(self.time_stack) == len(self.transformer_blocks)

        self.use_spatial_context = use_spatial_context
        self.in_channels = in_channels

        time_embed_dim = self.in_channels * 4
        self.time_pos_embed = nn.SequentialCell(
            nn.Dense(self.in_channels, time_embed_dim),
            nn.SiLU(),
            nn.Dense(time_embed_dim, self.in_channels),
        )

        self.time_mixer = AlphaBlender(alpha=merge_factor, merge_strategy=merge_strategy)

    def construct(
        self,
        x: Tensor,
        context: Optional[Tensor] = None,
        time_context: Optional[Tensor] = None,
        timesteps: Optional[int] = None,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        b, c, h, w = x.shape
        x_in = x
        spatial_context = context

        if self.use_spatial_context:
            assert context.ndim == 3, f"n dims of spatial context should be 3 but are {context.ndim}"

            time_context = context
            time_context_first_timestep = time_context[::timesteps]
            time_context = time_context_first_timestep.repeat(h * w, axis=0)  # b ... -> (b n) ...
        elif time_context is not None and not self.use_spatial_context:
            time_context = time_context.repeat(h * w, axis=0)  # b ... -> (b n) ...
            if time_context.ndim == 2:
                time_context = time_context.expand_dims(1)  # b c -> b 1 c

        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = x.transpose(0, 2, 3, 1).reshape(b, -1, c)  # b c h w -> b (h w) c FIXME: check
        if self.use_linear:
            x = self.proj_in(x)

        num_frames = ops.arange(timesteps).tile((x.shape[0] // timesteps,))
        t_emb = timestep_embedding(
            num_frames, self.in_channels, repeat_only=False, max_period=self.max_time_embed_period
        )
        emb = self.time_pos_embed(t_emb)
        # emb = self.time_pos_embed(self._pe.repeat(x.shape[0] // timesteps, axis=0))  # FIXME: check
        emb = emb[:, None, :]

        for it_, (block, mix_block) in enumerate(zip(self.transformer_blocks, self.time_stack)):
            x = block(
                x,
                context=spatial_context,
            )

            x_mix = x
            x_mix = x_mix + emb

            x_mix = mix_block(x_mix, context=time_context, timesteps=timesteps)
            x = self.time_mixer(
                x_spatial=x,
                x_temporal=x_mix,
                image_only_indicator=image_only_indicator,
            )
        if self.use_linear:
            x = self.proj_out(x)
        x = x.reshape(b, h, w, c).transpose(0, 3, 1, 2)  # b (h w) c -> b c h w
        if not self.use_linear:
            x = self.proj_out(x)
        out = x + x_in
        return out


class TemporalResBlock(ResBlock):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        dropout: float,
        video_kernel_size: Union[int, List[int]] = 3,
        merge_strategy: Literal["fixed", "learned", "learned_with_images"] = "fixed",
        merge_factor: float = 0.5,
        out_channels: Optional[int] = None,
        use_conv: bool = False,
        use_scale_shift_norm: bool = False,
        dims: int = 2,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__(
            channels,
            emb_channels,
            dropout,
            out_channels=out_channels,
            use_conv=use_conv,
            use_scale_shift_norm=use_scale_shift_norm,
            dims=dims,
            up=up,
            down=down,
        )

        self.time_stack = ResBlock(
            out_channels or channels,
            emb_channels,
            dropout=dropout,
            dims=3,
            out_channels=out_channels or channels,
            use_scale_shift_norm=False,
            use_conv=False,
            up=False,
            down=False,
            kernel_size=video_kernel_size,
            exchange_temb_dims=True,
        )
        self.time_mixer = AlphaBlender(alpha=merge_factor, merge_strategy=merge_strategy)

    def construct(
        self,
        x: Tensor,
        emb: Tensor,
        num_frames: int,
        image_only_indicator: Optional[Tensor] = None,
    ) -> Tensor:
        x_spat = super().construct(x, emb)
        # (b t) c h w -> b c t h w
        x_spat = x_spat.reshape(-1, num_frames, x_spat.shape[1], x_spat.shape[2], x_spat.shape[3]).swapaxes(1, 2)

        emb = emb.reshape(-1, num_frames, *emb.shape[1:])  # (b t) ... -> b t ...
        x_temp = self.time_stack(x_spat, emb)
        x = self.time_mixer(x_spatial=x_spat, x_temporal=x_temp, image_only_indicator=image_only_indicator)

        x = x.swapaxes(1, 2).reshape(-1, x.shape[1], x.shape[3], x.shape[4])  # b c t h w -> (b t) c h w
        return x


class VideoUNet(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: int,
        use_recompute: bool = False,
        dropout: float = 0.0,
        channel_mult: List[int] = (1, 2, 4, 8),
        conv_resample: bool = True,
        dims: int = 2,
        num_classes: Optional[int] = None,
        num_heads: int = -1,
        num_head_channels: int = -1,
        num_heads_upsample: int = -1,
        use_scale_shift_norm: bool = False,
        resblock_updown: bool = False,
        transformer_depth: Union[List[int], int] = 1,
        transformer_depth_middle: Optional[int] = None,
        context_dim: Optional[int] = None,
        time_downup: bool = False,
        time_context_dim: Optional[int] = None,
        spatial_transformer_attn_type: Literal["vanilla", "flash-attention"] = "vanilla",
        extra_ff_mix_layer: bool = False,
        use_spatial_context: bool = False,
        merge_strategy: Literal["fixed", "learned", "learned_with_images"] = "fixed",
        merge_factor: float = 0.5,
        video_kernel_size: Union[int, Tuple[int]] = 3,
        use_linear_in_transformer: bool = False,
        adm_in_channels: Optional[int] = None,
        disable_temporal_crossattention: bool = False,
        max_ddpm_temb_period: int = 10000,
    ):
        super().__init__()
        assert context_dim is not None

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1

        if num_head_channels == -1:
            assert num_heads != -1

        if not isinstance(video_kernel_size, int):  # kludge around OmegaConf
            video_kernel_size = tuple(video_kernel_size)

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(transformer_depth, int):
            transformer_depth = len(channel_mult) * [transformer_depth]
        transformer_depth_middle = transformer_depth_middle or transformer_depth[-1]

        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.SequentialCell(
            nn.Dense(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Dense(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                # print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Dense(1, time_embed_dim)
            elif self.num_classes == "timestep":
                self.label_emb = nn.SequentialCell(
                    Timestep(model_channels),
                    nn.SequentialCell(
                        nn.Dense(model_channels, time_embed_dim),
                        nn.SiLU(),
                        nn.Dense(time_embed_dim, time_embed_dim),
                    ),
                )

            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.SequentialCell(
                    nn.SequentialCell(
                        nn.Dense(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        nn.Dense(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.CellList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, has_bias=True, pad_mode="same"))]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        def get_attention_layer(
            ch,
            num_heads,
            dim_head,
            depth=1,
            context_dim=None,
            disabled_sa=False,
        ):
            return TemporalTransformer(
                ch,
                num_heads,
                dim_head,
                depth=depth,
                context_dim=context_dim,
                time_context_dim=time_context_dim,
                dropout=dropout,
                ff_in=extra_ff_mix_layer,
                use_spatial_context=use_spatial_context,
                merge_strategy=merge_strategy,
                merge_factor=merge_factor,
                use_linear=use_linear_in_transformer,
                attn_mode=spatial_transformer_attn_type,
                disable_self_attn=disabled_sa,
                disable_temporal_crossattention=disable_temporal_crossattention,
                max_time_embed_period=max_ddpm_temb_period,
            )

        def get_resblock(
            merge_factor,
            merge_strategy,
            video_kernel_size,
            ch,
            time_embed_dim,
            dropout,
            out_ch,
            dims,
            use_scale_shift_norm,
            down=False,
            up=False,
        ):
            return TemporalResBlock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                channels=ch,
                emb_channels=time_embed_dim,
                dropout=dropout,
                out_channels=out_ch,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
                down=down,
                up=up,
            )

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=mult * model_channels,
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

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            disabled_sa=False,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                if use_recompute:
                    self.input_blocks[-1].recompute()
            if level != len(channel_mult) - 1:
                ds *= 2
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_down=time_downup,
                        )
                    )
                )
                if use_recompute:
                    self.input_blocks[-1].recompute()
                ch = out_ch
                input_block_chans.append(ch)

                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels

        self.middle_block = TimestepEmbedSequential(
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                time_embed_dim=time_embed_dim,
                out_ch=None,
                dropout=dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            get_attention_layer(
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth_middle,
                context_dim=context_dim,
            ),
            get_resblock(
                merge_factor=merge_factor,
                merge_strategy=merge_strategy,
                video_kernel_size=video_kernel_size,
                ch=ch,
                out_ch=None,
                time_embed_dim=time_embed_dim,
                dropout=dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    get_resblock(
                        merge_factor=merge_factor,
                        merge_strategy=merge_strategy,
                        video_kernel_size=video_kernel_size,
                        ch=ch + ich,
                        time_embed_dim=time_embed_dim,
                        dropout=dropout,
                        out_ch=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    layers.append(
                        get_attention_layer(
                            ch,
                            num_heads,
                            dim_head,
                            depth=transformer_depth[level],
                            context_dim=context_dim,
                            disabled_sa=False,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    ds //= 2
                    layers.append(
                        get_resblock(
                            merge_factor=merge_factor,
                            merge_strategy=merge_strategy,
                            video_kernel_size=video_kernel_size,
                            ch=ch,
                            time_embed_dim=time_embed_dim,
                            dropout=dropout,
                            out_ch=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(
                            ch,
                            conv_resample,
                            dims=dims,
                            out_channels=out_ch,
                            third_up=time_downup,
                        )
                    )

                self.output_blocks.append(TimestepEmbedSequential(*layers))
                if use_recompute:
                    self.output_blocks[-1].recompute()
                self._feature_size += ch

        self.out = nn.SequentialCell(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, pad_mode="same")),
        )

        if use_recompute:
            self.label_emb.recompute()
            self.middle_block.recompute()

    def get_temporal_param_names(self, prefix: str = "") -> Set[str]:
        return {
            prefix + name
            for name, _ in self.parameters_and_names()
            if any([n in name for n in ["time_stack", "time_mixer", "time_pos_embed"]])
        }

    def construct(
        self,
        x: Tensor,
        timesteps: Tensor,
        context_pa: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        time_context: Optional[Tensor] = None,
        num_frames: Optional[int] = None,
        image_only_indicator: Optional[Tensor] = ops.zeros((1, 1)),
    ):
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional -> no, relax this TODO"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x
        for module in self.input_blocks:
            h = module(
                h,
                emb,
                context=context_pa,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_frames=num_frames,
            )
            hs.append(h)

        h = self.middle_block(
            h,
            emb,
            context=context_pa,
            image_only_indicator=image_only_indicator,
            time_context=time_context,
            num_frames=num_frames,
        )

        for i, module in enumerate(self.output_blocks, start=1):
            h = ops.cat([h, hs[-i]], axis=1)
            h = module(
                h,
                emb,
                context=context_pa,
                image_only_indicator=image_only_indicator,
                time_context=time_context,
                num_frames=num_frames,
            )

        return self.out(h)
