import logging

import numpy as np
from ldm.modules.attention import FeedForward, default
from ldm.modules.attention import FlashAttention as LdmFlashAttention
from ldm.modules.attention import Attention as LdmAttention
from ldm.modules.diffusionmodules.util import Identity, linear, timestep_embedding
from ldm.util import is_old_ms_version

import mindspore as ms
from mindspore import Parameter, nn, ops

from examples.stable_diffusion_v2.ldm.models.diffusion.ddpm import LatentDiffusion
from examples.stable_diffusion_v2.ldm.modules.attention import exists

_logger = logging.getLogger(__name__)
try:
    from mindspore.ops._op_impl._custom_op.flash_attention.flash_attention_impl import get_flash_attention

    FLASH_IS_AVAILABLE = True
    print("flash attention is available.")
except ImportError:
    FLASH_IS_AVAILABLE = False
    print("flash attention is unavailable.")


def refine_replace(attn_base, attn_replace, ):
    # replace_attn 1,b/2,-1,-1,77
    # mapper 1,77 哪个位置上的词需要替换？对
    # alphas 替换力度
    alphas = 1
    mapper = np.ones((1, 77))
    # 8,8,1024,1,77 -> 1,8,8,1024,77
    attn_base_replace = attn_base[:, :, :, mapper].permute(3, 0, 1, 2, 4)
    attn_replace = attn_base_replace * alphas + attn_replace * (1 - alphas)
    return attn_replace


def reweight_replace(attn_base, attn_replace):
    attn_base = refine_replace(attn_base, attn_replace)
    # equalizer 文本权重列表，默认1
    equalizer = np.ones((1, 77))
    attn_replace = attn_base[None, :, :, :] * equalizer[:, None, None, :]
    return attn_replace
    pass


def replace_replace(attn_base):
    mapper = ms.Tensor(np.eye(77).reshape(1, 77, 77))
    # shape = attn_base.shape
    # attn_base = attn_base.reshape((shape[0] // 8, 8, shape[1], shape[2]))
    # attn = ms.ops.einsum('thpw,bwn->bthpn', attn_base, mapper)
    # attn = ms.ops.einsum('hpw,bwn->bhpn', attn_base, mapper)
    # shape = attn.shape
    # attn = attn.reshape((shape[0] * shape[1], shape[2], shape[3]))
    return attn_base


class GroupNorm(nn.GroupNorm):
    def construct(self, x):
        if len(x.shape) == 5:
            # (b, c, f, h, w)
            b, c, f, h, w = x.shape
            x = x.reshape((b, c, f * h, w))
            out = super().construct(x)
            out = out.reshape((b, c, f, h, w))
        else:
            out = super().construct(x)
        return out


def normalization(channels, eps: float = 1e-5):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Cell for normalization.
    """
    return GroupNorm(32, channels, eps=eps).to_float(ms.float32)


def split_attention(attn, batch_size=2):
    index = attn.shape[0] // batch_size
    return attn[:index], attn[:index]


def reshape_attention(attn):
    b, h, s, t = attn.shape
    return ms.ops.reshape(attn, (b * h, s, t))


def replace_self_attention(attn, step, controller):
    num_self_replace = controller["num_self_replace"]
    assert step is not None, 'step is None'
    if num_self_replace[0] < step and step < num_self_replace[1]:
        if attn.shape[2] <= 256:
            attn_base, _ = split_attention(attn)
            return attn_base.tile((2, 1, 1))
    return attn


def replace_cross_attention(attn, step, controller):
    num_cross_replace = controller["num_self_replace"]
    type = controller["type"]
    if num_cross_replace[0] < step and step < num_cross_replace[1]:

        if type == 'replace':
            attn_base, _ = split_attention(attn)
            attn_new = replace_replace(attn_base)
            return ms.ops.cat([attn_base, attn_new], axis=0)
            # attn_new.tile((2, 1, 1))
    return attn


class Attention(LdmAttention):
    def construct(self, q, k, v, mask, is_cross_attention, step=None, controller=None):
        sim = ops.matmul(q, self.transpose(k, (0, 2, 1))) * self.scale

        if exists(mask):
            mask = self.reshape(mask, (mask.shape[0], -1))
            if sim.dtype == ms.float16:
                finfo_type = np.float16
            else:
                finfo_type = np.float32
            max_neg_value = -np.finfo(finfo_type).max
            mask = mask.repeat(self.heads, axis=0)
            mask = ops.expand_dims(mask, axis=1)
            sim.masked_fill(mask, max_neg_value)

        if self.upcast:
            # use fp32 for exponential inside
            attn = self.softmax(sim.astype(ms.float32)).astype(v.dtype)
        else:
            attn = self.softmax(sim)

        out = ops.matmul(attn, v)

        return out, attn

    def __init__(self, *args, **kwargs):
        self.name = kwargs["name"]
        # print(self.name)
        self.head = kwargs["head"]
        del kwargs["name"]
        del kwargs["head"]
        super().__init__(*args, **kwargs)


class FlashAttention(LdmFlashAttention):

    def construct(self, q, k, v, attention_mask=None, dropout_mask=None, alibi_mask=None, is_cross_attention=False,
                  step=None, controller=None):
        # ALiBi, reference to https://arxiv.org/abs/2108.12409
        return super().construct(q, k, v, attention_mask, dropout_mask, alibi_mask)


class CrossAttention(nn.Cell):
    def __init__(
            self,
            query_dim,
            context_dim=None,
            heads=8,
            dim_head=64,
            dropout=1.0,
            dtype=ms.float32,
            enable_flash_attention=False,
            upcast=False,
            name='CrossAttention',
    ):
        # print(name, enable_flash_attention)
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.name = name

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.reshape = ops.Reshape()
        # self.softmax = ops.Softmax(axis=-1)
        self.transpose = ops.Transpose()
        self.to_q = nn.Dense(query_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_k = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_v = nn.Dense(context_dim, inner_dim, has_bias=False).to_float(dtype)
        self.to_out = nn.SequentialCell(
            nn.Dense(inner_dim, query_dim).to_float(dtype),
            nn.Dropout(dropout) if is_old_ms_version() else nn.Dropout(p=1 - dropout),
        )
        self.use_flash_attention = (
                enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )
        if enable_flash_attention and not self.use_flash_attention:
            print("WARNING: flash attention is set to enable but not available.")
        if self.use_flash_attention:
            print("INFO: flash attention will be used.")
        # print('to_q:', query_dim, dim_head, heads)
        # print('to_k:', context_dim, inner_dim)
        # ar = AReplace()
        # ar.num_att_layers += 1
        self.attention = Attention(dim_head, upcast=upcast, name=name, head=heads)
        self.flash_attention = FlashAttention(self.heads, dim_head) if self.use_flash_attention else None

    def construct(self, x, context=None, mask=None, step=None, controller=None):
        is_cross = context is not None
        # print("test_0: x", x.shape)
        q = self.to_q(x)
        # print("test_1: q", q.shape)

        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        def rearange_in(x):
            # (b, n, h*d) -> (b*h, n, d)
            h = self.heads
            b, n, d = x.shape[0], x.shape[1], x.shape[2]
            d = d // h

            x = self.reshape(x, (b, n, h, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b * h, n, d))
            return x

        q = rearange_in(q)
        k = rearange_in(k)
        v = rearange_in(v)

        if self.use_flash_attention and q.shape[1] % 16 == 0 and k.shape[1] % 16 == 0 and q.shape[1] > 1024:
            attn = None
            out = self.flash_attention(q, k, v, is_cross_attention=is_cross, step=step, controller=controller)
        else:
            out, attn = self.attention(q, k, v, mask, is_cross_attention=is_cross, step=step, controller=controller)

        def rearange_out(x):
            # (b*h, n, d) -> (b, n, h*d)
            h = self.heads
            b, n, d = x.shape[0], x.shape[1], x.shape[2]
            b = b // h

            x = self.reshape(x, (b, h, n, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b, n, h * d))
            return x

        out = rearange_out(out)
        return self.to_out(out), attn


# Spatial Transformer and Attention Layer Modification based on SD
class SparseCausalAttention(CrossAttention):
    """
    The SparseCausalAttention, which learns the temporal attention of the current frame
    v_i between the first frame v_1 and the previous frame v_{i-1}. The spatial attention
    is learned cross all pixels.
    """

    def concat_first_previous_features(self, x, video_length, former_frame_index):
        bf, hw, c = x.shape
        # (b f) (hw) c -> b f (hw) c
        x = x.reshape((bf // video_length, video_length, hw, c))
        x = ops.cat([x[:, [0] * video_length], x[:, former_frame_index]], axis=2)
        # b f (2hw) c -> (b f) (2hw) c
        x = x.reshape((bf, 2 * hw, c))
        return x

    def construct(self, x, context=None, mask=None, video_length=None, step=None, controller=None):
        is_cross = context is not None
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        def rearange_in(x):
            # (b, n, h*d) -> (b*h, n, d)
            h = self.heads
            b, n, d = x.shape
            d = d // h

            x = self.reshape(x, (b, n, h, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b * h, n, d))
            return x

        former_frame_index = ops.arange(video_length) - 1
        former_frame_index[0] = 0

        k = self.concat_first_previous_features(k, video_length, former_frame_index)
        v = self.concat_first_previous_features(v, video_length, former_frame_index)

        q = rearange_in(q)
        k = rearange_in(k)
        v = rearange_in(v)

        if mask is not None:
            if mask.shape[-1] != q.shape[1]:
                target_length = q.shape[1]
                ndim = len(mask.shape)
                paddings = [[0, 0] for i in range(ndim - 1)] + [0, target_length]
                mask = nn.Pad(paddings)(mask)
                mask = mask.repeat_interleave(self.heads, axis=0)

        if self.use_flash_attention and q.shape[1] % 16 == 0 and k.shape[1] % 16 == 0 and q.shape[1] > 1024:
            attn = None
            out = self.flash_attention(q, k, v, is_cross_attention=is_cross, step=step, controller=controller)
        else:
            # print(self.use_flash_attention, 'sp', q.shape, k.shape)
            out, attn = self.attention(q, k, v, mask, is_cross_attention=is_cross, step=step, controller=controller)

        def rearange_out(x):
            # (b*h, n, d) -> (b, n, h*d)
            h = self.heads
            b, n, d = x.shape
            b = b // h

            x = self.reshape(x, (b, h, n, d))
            x = self.transpose(x, (0, 2, 1, 3))
            x = self.reshape(x, (b, n, h * d))
            return x

        out = rearange_out(out)
        return self.to_out(out), attn


class BasicTransformerBlock_ST(nn.Cell):
    def __init__(
            self,
            dim,
            n_heads,
            d_head,
            dropout=1.0,
            context_dim=None,
            gated_ff=True,
            checkpoint=True,
            dtype=ms.float32,
            enable_flash_attention=False,
            cross_frame_attention=False,
            unet_chunk_size=2,
    ):
        super().__init__()
        assert not cross_frame_attention, "expect to have cross_frame_attention to be False"
        self.replace_step = 8
        attn_map_shape_cross = (n_heads * 8, 4096 * 25 // (n_heads * n_heads), 77)
        attn_map_shape_self = (n_heads * 8, 4096 * 25 // (n_heads * n_heads), 4096 * 25 * 2 // (n_heads * n_heads))
        self.store_cross_attn = Parameter(ms.ops.zeros((self.replace_step,) + attn_map_shape_cross, ms.float16),
                                          requires_grad=False)
        if attn_map_shape_self[1] <= 1024:
            self.store_self_attn = Parameter(ms.ops.zeros((self.replace_step,) + attn_map_shape_self, ms.float16),
                                             requires_grad=False)
        else:
            self.store_self_attn = Parameter([], requires_grad=False)

        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
            name='SparseCausalAttention',
        )  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)

        self.attn2 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=enable_flash_attention,
            name='attn2',

        )  # is self-attn if context is none
        self.norm1 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.norm2 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.norm3 = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)
        self.checkpoint = checkpoint

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
            enable_flash_attention=True,
            # enable_flash_attention=enable_flash_attention,
            name='temp',
        )
        self.attn_temp.to_out[0].weight = Parameter(ms.Tensor(np.zeros(self.attn_temp.to_out[0].weight.shape), dtype))
        self.norm_temp = nn.LayerNorm([dim], epsilon=1e-05).to_float(dtype)

    def construct(self, x, context=None, video_length=None, step=None, controller=None, is_invert=False):
        x1, attn1 = self.attn1(self.norm1(x), video_length=video_length, step=step, controller=controller)
        x1 += x
        x2, attn2 = self.attn2(self.norm2(x1), context=context, step=step, controller=controller)
        x = x2 + x1
        x = self.ff(self.norm3(x)) + x

        if is_invert > 0 and step < 8:
            #if self.store_self_attn.shape[0] > 0:
            print(self.store_self_attn[step].shape, attn1.shape)
                #self.store_self_attn[step] = attn1
            print(self.store_cross_attn[step].shape, attn2.shape)

            #self.store_cross_attn[step] = attn2

        # temporal attention
        # (b f) (hw) c -> (b h w) f c
        bf, hw, c = x.shape
        x = x.reshape((bf // video_length, video_length, hw, c))
        x = x.transpose((0, 2, 1, 3)).reshape(((bf // video_length) * hw, video_length, c))
        x1, _ = self.attn_temp(self.norm_temp(x), step=step)
        x1 += x
        # (b h w) f c -> (b f) (hw) c
        x = x.reshape((bf // video_length, hw, video_length, c))
        x = x.transpose((0, 2, 1, 3)).reshape((bf, hw, c))
        return x


class SpatialTransformer3D(nn.Cell):
    """
    Transformer block for video data.
    """

    def __init__(
            self,
            in_channels,
            n_heads,
            d_head,
            depth=1,
            dropout=1.0,
            context_dim=None,
            use_checkpoint=True,
            use_linear=False,
            dtype=ms.float32,
            enable_flash_attention=False,
            cross_frame_attention=False,
            unet_chunk_size=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dtype = dtype
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels, eps=1e-6)

        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
            ).to_float(
                dtype
            )  # should be conv2d
        else:
            self.proj_in = nn.Dense(in_channels, inner_dim).to_float(dtype)

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock_ST(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    enable_flash_attention=enable_flash_attention,
                    cross_frame_attention=cross_frame_attention,
                    unet_chunk_size=unet_chunk_size,
                )
                for d in range(depth)
            ]
        )

        if not use_linear:
            self.proj_out = nn.Conv2d(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)
        else:
            self.proj_out = nn.Dense(in_channels, inner_dim).to_float(dtype)

        self.use_linear = use_linear
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()

    def construct(self, x, emb=None, context=None, step=None, controller=None, is_invert=False):
        # note: if no context is given, cross-attention defaults to self-attention
        assert len(x.shape) == 5, f"Expect to have five dimensions input, but got {len(x.shape)} dims"
        b, c, f, h, w = x.shape
        if context is not None:
            context = ops.repeat_interleave(context, f, 0)  # (b, n, c) -> (b*f, n, c)
        x = x.transpose((0, 2, 1, 3, 4)).reshape((b * f, c, h, w))  # (b*f, c, h, w)

        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
            ch = x.shape[1]
            x = x.transpose((0, 2, 3, 1)).reshape(
                (b * f, h * w, ch)
            )  # (b*f, ch, h, w) -> (b*f, h, w, ch) -> (b*f, h*w, ch)
        else:
            ch = x.shape[1]
            x = x.transpose((0, 2, 3, 1)).reshape(
                (b * f, h * w, ch)
            )  # (b*f, ch, h, w) -> (b*f, h, w, ch) -> (b*f, h*w, ch)
            x = self.proj_in(x)
        for block in self.transformer_blocks:
            x = block(x, context=context, video_length=f, step=step, controller=controller, is_invert=is_invert)
        if self.use_linear:
            x = self.proj_out(x)
            ch = x.shape[-1]
            x = x.reshape((b * f, h, w, ch)).transpose(
                (0, 3, 1, 2)
            )  # (b*f, h*w, ch) -> (b*f, h, w, ch) -> (b*f, ch, h, w)
        else:
            ch = x.shape[-1]
            x = x.reshape((b * f, h, w, ch)).transpose(
                (0, 3, 1, 2)
            )  # (b*f, h*w, ch) -> (b*f, h, w, ch) -> (b*f, ch, h, w)
            x = self.proj_out(x)
        out = x + x_in
        ch = x.shape[1]
        out = out.reshape((b, f, ch, h, w)).transpose(
            (0, 2, 1, 3, 4)
        )  # (b*f, ch, h, w) -> (b, f, ch, h, w) -> (b, ch, f, h, w)
        return out


# Replace 2D Blocks by 3D Blocks (Conv2d -> InflatedConv3d; Upsample -> Upsample3D, etc.)
class InflatedConv3d(nn.Conv2d):
    def construct(self, x):
        b, c, f, h, w = x.shape

        # b c f h w -> (b f) c h w
        x = x.transpose((0, 2, 1, 3, 4)).reshape((b * f, c, h, w))
        x = super().construct(x)
        _, c, h, w = x.shape
        # (b f) c h w -> b c f h w
        x = x.reshape((b, f, c, h, w)).transpose(0, 2, 1, 3, 4)

        return x


class conv_nd(nn.Cell):
    def __init__(self, dims, *args, **kwargs):
        super().__init__()
        if dims == 1:
            self.conv = nn.Conv1d(*args, **kwargs)
        elif dims == 2:
            self.conv = nn.Conv2d(*args, **kwargs)
        elif dims == 3:
            self.conv = InflatedConv3d(*args, **kwargs)  # use inflated Conv3D instead of nn.Conv3D
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

    def construct(self, x, emb=None, context=None):
        x = self.conv(x)
        return x


class InflatedAvgPool3D(nn.AvgPool2d):
    def construct(self, x):
        b, c, f, h, w = x.shape

        # b c f h w -> (b f) c h w
        x = x.transpose((0, 2, 1, 3, 4)).reshape((b * f, c, h, w))
        x = super().construct(x)
        h, w = x.shape[-2:]
        # (b f) c h w -> b c f h w
        x = x.reshape((b, f, c, h, w)).transpose(0, 2, 1, 3, 4)

        return x


class avg_pool_nd(nn.Cell):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """

    def __init__(self, dims, *args, **kwargs):
        super().__init__()
        if dims == 1:
            self.avgpool = nn.AvgPool1d(*args, **kwargs)
        elif dims == 2:
            self.avgpool = nn.AvgPool2d(*args, **kwargs)
        elif dims == 3:
            self.avgpool = InflatedAvgPool3D(*args, **kwargs)
        else:
            raise ValueError(f"unsupported dimensions: {dims}")

    def construct(self, x, emb=None, context=None):
        x = self.avgpool(x)
        return x


class Upsample3D(nn.Cell):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the last two dimensions.
    """

    def __init__(self, channels, use_conv=False, dims=3, out_channels=None, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        assert self.dims == 3, "Upsample3D must have dims=3"
        if use_conv:
            self.conv = conv_nd(
                self.dims, self.channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
            ).to_float(dtype)

    def construct(self, x, emb=None, context=None):
        if self.dims == 3:
            # b, c, f, h, w
            b, c, f, h, w = x.shape
            x = x.transpose((0, 2, 1, 3, 4)).reshape(
                (b * f, c, h, w)
            )  # (b, c, f, h, w) -> (b, f, c, h, w) -> (bf, c, h, w)
            x = ops.ResizeNearestNeighbor((h * 2, w * 2))(x)  # do not upsample the temporal axis, only the spatial axes
            h, w = x.shape[-2:]
            x = x.reshape((b, f, c, h, w)).transpose(
                (0, 2, 1, 3, 4)
            )  # (bf, c, h, w) -> (b, f, c, h, w) -> (b, c, f, h, w)
        else:
            # b, c, h, w
            x = ops.ResizeNearestNeighbor((x.shape[2] * 2, x.shape[3] * 2))(x)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample3D(nn.Cell):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the last two dimensions.
    """

    def __init__(self, channels, use_conv=False, dims=3, out_channels=None, padding=1, dtype=ms.float32):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        assert self.dims == 3, "Downsample3D must have dims=3"
        stride = 2  # InflatedConv3d is a Conv2D actually
        if use_conv:
            self.op = conv_nd(
                self.dims,
                self.channels,
                self.out_channels,
                3,
                stride=stride,
                padding=padding,
                has_bias=True,
                pad_mode="pad",
            ).to_float(dtype)
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def construct(self, x, emb=None, context=None):
        return self.op(x)


class ResnetBlock3D(nn.Cell):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
            self,
            channels,
            emb_channels,
            dropout=1.0,
            out_channels=None,
            use_conv=False,
            use_scale_shift_norm=False,
            dims=2,
            use_checkpoint=False,
            up=False,
            down=False,
            dtype=ms.float32,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.ori_channels = channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.updown = up or down
        self.dtype = dtype
        self.identity = Identity()
        self.split = ops.Split(1, 2)

        self.in_layers_norm = normalization(channels)
        self.in_layers_silu = nn.SiLU().to_float(self.dtype)
        self.in_layers_conv = conv_nd(
            dims, channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
        ).to_float(self.dtype)

        if up:
            self.h_upd = Upsample3D(channels, False, dims, dtype=self.dtype)
            self.x_upd = Upsample3D(channels, False, dims, dtype=self.dtype)
        elif down:
            self.h_upd = Downsample3D(channels, False, dims, dtype=self.dtype)
            self.x_upd = Downsample3D(channels, False, dims, dtype=self.dtype)
        else:
            self.h_upd = self.x_upd = self.identity

        self.emb_layers = nn.SequentialCell(
            nn.SiLU().to_float(self.dtype),
            linear(
                emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels, dtype=self.dtype
            ),
        )

        self.out_layers_norm = normalization(self.out_channels)
        self.out_layers_silu = nn.SiLU().to_float(self.dtype)

        if is_old_ms_version():
            self.out_layers_drop = nn.Dropout(keep_prob=self.dropout)
        else:
            self.out_layers_drop = nn.Dropout(p=1.0 - self.dropout)

        self.out_layers_conv = conv_nd(
            dims, self.out_channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
        ).to_float(self.dtype)

        if self.out_channels == channels:
            self.skip_connection = self.identity
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)
        else:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 1, has_bias=True, pad_mode="pad"
            ).to_float(self.dtype)

    def construct(self, x, emb, context=None):
        if self.updown:
            h = self.in_layers_norm(x)
            h = self.in_layers_silu(h)
            h = self.h_upd(h, emb, context)
            x = self.x_upd(x, emb, context)
            h = self.in_layers_conv(h, emb, context)
        else:
            h = self.in_layers_norm(x)
            h = self.in_layers_silu(h)
            h = self.in_layers_conv(h, emb, context)

        emb_out = self.emb_layers(emb)
        while len(emb_out.shape) < len(h.shape):
            emb_out = ops.expand_dims(emb_out, -1)

        if self.use_scale_shift_norm:
            scale, shift = self.split(emb_out)
            h = self.out_layers_norm(h) * (1 + scale) + shift
            h = self.out_layers_silu(h)
            h = self.out_layers_drop(h)
            h = self.out_layers_conv(h, emb, context)

        else:
            h = h + emb_out
            h = self.out_layers_norm(h)
            h = self.out_layers_silu(h)
            h = self.out_layers_drop(h)
            h = self.out_layers_conv(h, emb, context)

        return self.skip_connection(x) + h


class AttentionBlock(nn.Cell):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
            self,
            channels,
            num_heads=1,
            num_head_channels=-1,
            use_checkpoint=False,
            use_new_attention_order=False,
    ):
        super().__init__()


class Timestep(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def construct(self, t):
        return timestep_embedding(t, self.dim)


class UNetModel3D(nn.Cell):
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
            dims=3,
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
            adm_in_channels=None,
            use_recompute=False,
            controller=None,
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
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = nn.CellList(
                    [
                        ResnetBlock3D(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=mult * model_channels,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
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
                        else SpatialTransformer3D(
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
                            ResnetBlock3D(
                                ch,
                                time_embed_dim,
                                self.dropout,
                                out_channels=out_ch,
                                dims=dims,
                                use_checkpoint=use_checkpoint,
                                use_scale_shift_norm=use_scale_shift_norm,
                                down=True,
                                dtype=self.dtype,
                            )
                        ]
                    )
                    if resblock_updown
                    else nn.CellList(
                        [Downsample3D(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)]
                    )
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
                ResnetBlock3D(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
                ),
                AttentionBlock(
                    ch,
                    use_checkpoint=use_checkpoint,
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
                    use_checkpoint=use_checkpoint,
                    dtype=self.dtype,
                    dropout=self.dropout,
                    use_linear=use_linear_in_transformer,
                    enable_flash_attention=enable_flash_attention,
                ),
                ResnetBlock3D(
                    ch,
                    time_embed_dim,
                    self.dropout,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                    dtype=self.dtype,
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
                        ResnetBlock3D(
                            ch + ich,
                            time_embed_dim,
                            self.dropout,
                            out_channels=model_channels * mult,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            dtype=self.dtype,
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
                        else SpatialTransformer3D(
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
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResnetBlock3D(
                            ch,
                            time_embed_dim,
                            self.dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            dtype=self.dtype,
                        )
                        if resblock_updown
                        else Upsample3D(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype)
                    )
                    ds //= 2
                self.output_blocks.append(layers)
                self._feature_size += ch

        self.out = nn.SequentialCell(
            normalization(ch),
            nn.SiLU().to_float(self.dtype),
            conv_nd(dims, model_channels, out_channels, 3, padding=1, has_bias=True, pad_mode="pad").to_float(
                self.dtype
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
                mblock.recompute(parallel_optimizer_comm_recompute=True)
            for oblock in self.output_blocks:
                oblock.recompute(parallel_optimizer_comm_recompute=True)

        self.step = Parameter(ms.Tensor(0, ms.int32), requires_grad=False)
        self.is_invert = Parameter(ms.Tensor(-1, ms.int32), requires_grad=False)
        self.controller = controller

    @staticmethod
    def is_attention_layer(c):
        name = c.__class__.__name__
        if name in ['SpatialTransformer3D']:
            return True

        return False

    def construct(
            self, x, timesteps=None, context=None, y=None, features_adapter: list = None, append_to_context=None,
            **kwargs
    ):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        self.step = self.step + 1
        # print(self.controller)
        assert (y is not None) == (
                self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)

        if self.num_classes is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

        h = x

        if append_to_context is not None:
            context = ops.cat([context, append_to_context], axis=1)

        adapter_idx = 0
        for i, celllist in enumerate(self.input_blocks, 1):
            for cell in celllist:
                if self.is_attention_layer(cell):
                    h = cell(h, emb, context, self.step, self.controller, self.is_invert)
                else:
                    h = cell(h, emb, context)

            if features_adapter and i % 3 == 0:
                h = h + features_adapter[adapter_idx]
                adapter_idx += 1

            hs.append(h)

        if features_adapter:
            assert len(features_adapter) == adapter_idx, "Wrong features_adapter"

        for module in self.middle_block:
            if self.is_attention_layer(module):
                h = module(h, emb, context, self.step, self.controller, self.is_invert)
            else:
                h = module(h, emb, context)

        hs_index = -1
        for celllist in self.output_blocks:
            h = self.cat((h, hs[hs_index]))
            for cell in celllist:
                if self.is_attention_layer(cell):
                    h = cell(h, emb, context, self.step, self.controller, self.is_invert)
                else:
                    h = cell(h, emb, context)
            hs_index -= 1

        if self.predict_codebook_ids:
            return self.id_predictor(h)
        else:
            return self.out(h)


if __name__ == "__main__":
    model = UNetModel3D(
        image_size=32,
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_resolutions=[4, 2, 1],
        num_res_blocks=2,
        channel_mult=[1, 2, 4, 4],
        num_head_channels=64,  # SD_VERSION v2.0
        use_spatial_transformer=True,
        enable_flash_attention=False,
        use_linear_in_transformer=True,  # SD_VERSION v2.0
        transformer_depth=1,
        context_dim=1024,
        use_checkpoint=True,
        legacy=False,
        use_fp16=True,
        dropout=0.1,
    )


# todo 移动到ddpm
class InflatedLatentDiffusion(LatentDiffusion):
    def get_input(self, x, c):
        assert len(x.shape) == 5, f"expect the input image shape is (b, f, h, w, c), but got {x.shape}"
        x = self.transpose(x, (0, 1, 4, 2, 3))  # (b, f, h, w, c)-> (b, f, c, h, w)
        b, f, ch, h, w = x.shape
        z = ops.stop_gradient(self.get_first_stage_encoding(self.encode_first_stage(x.reshape((b * f, ch, h, w)))))
        _, ch, h, w = z.shape
        z = self.transpose(z.reshape((b, f, ch, h, w)), (0, 2, 1, 3, 4))  # (b, f, c, h, w) - > (b, c, f, h, w)
        return z, c
