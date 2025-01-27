import os
import sys

import numpy as np

import mindspore as ms
from mindspore import mint, nn

sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../../../../")))  # to include sv3d
sys.path.append(os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../../")))  # to include mvdream
from sv3d.sgm.modules.diffusionmodules.openaimodel import (
    AttentionBlock,
    Downsample,
    QKVAttention,
    QKVAttentionLegacy,
    ResBlock,
    TimestepBlock,
    Upsample,
    normalization,
    zero_module,
)
from sv3d.sgm.modules.diffusionmodules.util import conv_nd, linear, timestep_embedding

from ..attention import SpatialTransformer3D, exists

# from mvdream.ldm.modules.attention import SpatialTransformer3D, exists


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += ms.Tensor([matmul_ops], dtype=ms.int64)


class QKVAttentionLegacy(QKVAttentionLegacy):
    """
    Adding flops counting for thop.
    """

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class QKVAttention(QKVAttention):
    """
    Adding flops counting for thop.
    """

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class AttentionBlock(AttentionBlock):
    """
    Override the attn worker.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_new_attention_order=False,
    ):
        super().__init__(
            channels=channels,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            use_new_attention_order=use_new_attention_order,
        )
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)


class TimestepEmbedSequential(nn.SequentialCell, TimestepBlock):
    def construct(self, x, emb, context=None, num_frames=1):
        for cell in self.cell_list:
            if isinstance(cell, TimestepBlock):
                x = cell(x, emb)
            # elif isinstance(cell, SpatialTransformer):   # SpatialTransformer3D inherits from SpatialTransformer, this line True then wrong result
            #     x = cell(x, context)
            elif isinstance(cell, SpatialTransformer3D):
                x = cell(x, context, num_frames=num_frames)
            else:
                x = cell(x)
        return x


class MultiViewUNetModel(nn.Cell):
    """
    The full multi-view UNet model with attention, timestep embedding and camera embedding.
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
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    :param camera_dim: dimensionality of camera input.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
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
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        camera_dim=None,
        use_recompute=False,
    ):
        # super().__init__(
        #     in_channels,
        #     model_channels,
        #     out_channels,
        #     num_res_blocks,
        #     attention_resolutions,
        #     num_head_channels=num_head_channels
        # )
        super().__init__()
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.num_classes = num_classes
        self.dtype = ms.float32
        self.model_channels = model_channels

        time_embed_dim = model_channels * 4
        if camera_dim is not None:
            self.camera_embed = nn.SequentialCell(
                [
                    linear(camera_dim, time_embed_dim),
                    nn.SiLU(),
                    linear(time_embed_dim, time_embed_dim),
                ]
            )
        self.time_embed = nn.SequentialCell(
            [
                linear(model_channels, time_embed_dim),
                nn.SiLU(),
                linear(time_embed_dim, time_embed_dim),
            ]
        )

        self.num_heads_upsample = num_heads_upsample

        # input blocks
        self.input_blocks = nn.CellList(
            [TimestepEmbedSequential(conv_nd(dims, in_channels, model_channels, 3, padding=1, pad_mode="pad"))]
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
                            else SpatialTransformer3D(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
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
                ds *= 2
                self._feature_size += ch

        # middle blocks
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
            else SpatialTransformer3D(  # always uses a self-attn
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn,
                use_linear=use_linear_in_transformer,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # output blocks
        self.output_blocks = nn.CellList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
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
                    if legacy:
                        # num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            )
                            if not use_spatial_transformer
                            else SpatialTransformer3D(
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=disabled_sa,
                                use_linear=use_linear_in_transformer,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.SequentialCell(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1, pad_mode="pad")),
        )

        # if use_recompute:
        #     self.recompute_strategy_v1()

    # @jit  # TODO refactor the args list to accept static graph acceleration
    def construct(self, x, timesteps=None, context=None, y=None, camera=None, num_frames=1):
        """
        Apply the model to an input batch.
        :param x: an [(N x F) x C x ...] Tensor of inputs. F is the number of frames (views).
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :param num_frames: a integer indicating number of frames for tensor reshaping.
        :return: an [(N x F) x C x ...] Tensor of outputs. F is the number of frames (views).
        """
        assert x.shape[0] % num_frames == 0, "[UNet] input batch size must be dividable by num_frames!"
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        # if self.num_classes is not None:
        #     assert y.shape[0] == x.shape[0]
        #     emb = emb + self.label_emb(y)

        # Add camera embeddings
        if camera is not None:
            assert camera.shape[0] == emb.shape[0]
            emb = emb + self.camera_embed(camera)

        h = x.to(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context, num_frames=num_frames)
            hs.append(h)
        h = self.middle_block(h, emb, context, num_frames=num_frames)
        for module in self.output_blocks:
            h = mint.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, num_frames=num_frames)
        h = h.type(x.dtype)
        return self.out(h)
