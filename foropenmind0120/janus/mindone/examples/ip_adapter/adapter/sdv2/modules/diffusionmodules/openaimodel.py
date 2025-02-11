import logging

from ldm.modules.diffusionmodules.openaimodel import AttentionBlock, Downsample, ResBlock, UNetModel, Upsample
from ldm.modules.diffusionmodules.util import conv_nd, linear, normalization, zero_module

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from ..attention import IPAdapterSpatialTransformer

_logger = logging.getLogger(__name__)


class IPAdapterUNetModel(UNetModel):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
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
    :param fa_max_head_dim: the maximum head dimension to apply flash attention. In case of OOM,
                            reduce this value.
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
        fa_max_head_dim=256,
        adm_in_channels=None,
        upcast_attn=False,
        use_recompute=False,
        upcast_sigmoid=False,
        ip_scale=1.0,
        num_tokens=4,
    ):
        super(UNetModel, self).__init__()

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
            nn.SiLU().to_float(ms.float32) if upcast_sigmoid else nn.SiLU(),
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
                        nn.SiLU().to_float(ms.float32) if upcast_sigmoid else nn.SiLU(),
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
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
                            upcast_sigmoid=upcast_sigmoid,
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
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=dim_head,
                            use_new_attention_order=use_new_attention_order,
                        )
                        if not use_spatial_transformer
                        else IPAdapterSpatialTransformer(
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
                            upcast_attn=upcast_attn,
                            fa_max_head_dim=fa_max_head_dim,
                            ip_scale=ip_scale,
                            num_tokens=num_tokens,
                        )
                    )
                self.input_blocks.append(layers)
                self._feature_size += ch
                input_block_chans.append(ch)
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
                                upcast_sigmoid=upcast_sigmoid,
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
                    upcast_sigmoid=upcast_sigmoid,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
                    num_heads=num_heads,
                    num_head_channels=dim_head,
                    use_new_attention_order=use_new_attention_order,
                )
                if not use_spatial_transformer
                else IPAdapterSpatialTransformer(
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
                    upcast_attn=upcast_attn,
                    fa_max_head_dim=fa_max_head_dim,
                    ip_scale=ip_scale,
                    num_tokens=num_tokens,
                ),
                ResBlock(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                    upcast_sigmoid=upcast_sigmoid,
                ),
            ]
        )
        self._feature_size += ch

        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
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
                            upcast_sigmoid=upcast_sigmoid,
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
                        else IPAdapterSpatialTransformer(
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
                            upcast_attn=upcast_attn,
                            fa_max_head_dim=fa_max_head_dim,
                            ip_scale=ip_scale,
                            num_tokens=num_tokens,
                        )
                    )
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
                            upcast_sigmoid=upcast_sigmoid,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)
                    )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

        self.out = nn.SequentialCell(
            normalization(ch),
            nn.SiLU().to_float(ms.float32) if upcast_sigmoid else nn.SiLU(),
            zero_module(
                conv_nd(dims, model_channels, out_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(
                    self.dtype
                )
            ),
        )

        if self.predict_codebook_ids:
            self.id_predictor = nn.SequentialCell(
                normalization(ch),
                conv_nd(dims, model_channels, n_embed, 1, has_bias=True, pad_mode="pad").to_float(self.dtype),
            )
        self.cat = ops.Concat(axis=1)

        # recompute to save NPU mem
        if use_recompute:
            for mblock in self.middle_block:
                mblock.recompute()
            for oblock in self.output_blocks:
                oblock.recompute()
