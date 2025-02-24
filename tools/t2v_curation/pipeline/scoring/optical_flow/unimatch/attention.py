import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, XavierUniform

from .utils import merge_splits, merge_splits_1d, split_feature, split_feature_1d

def single_head_full_attention(q, k, v):
    # q, k, v: [B, L, C]
    assert q.ndim == k.ndim == v.ndim == 3

    scores = ops.matmul(q, k.transpose(0, 2, 1)) / (q.shape[2] ** 0.5) # [B, L, L]
    attn = ops.softmax(scores, axis=2) # [B, L, L]
    out = ops.matmul(attn, v) # [B, L, C]

    return out

def single_head_full_attention_1d(
        q,
        k,
        v,
        h = None,
        w = None,
):
    # q, k, v: [B, L, C]
    assert h is not None and w is not None
    assert q.shape[1] == h * w

    b, _, c = q.shape
    q = q.reshape[b, h, w, c]  # [B, H, W, C]
    k = k.reshape[b, h, w, c]
    v = v.reshape[b, h, w, c]

    scale_factor = c ** 0.5
    scores = ops.matmul(q, k.transpose(0, 1, 3, 2))/ scale_factor # [B, H, W, W]
    attn = ops.softmax(scores, axis = -1)
    out = ops.matmul(attn, v).reshape(b, -1, c) # [B, H*W, C]

    return out

def single_head_split_window_attention(
        q, k, v, num_splits = 1, with_shift = False, h = None, w = None, attn_mask = None,
):
    # ref: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    # q, k, v: [B, L, C]
    assert q.ndim == k.ndim == v.ndim == 3
    assert q.shape[1] == h * w
    assert h is not None and w is not None

    b, _, c = q.shape
    b_new = b * num_splits * num_splits

    window_size_h = h // num_splits
    window_size_w = w // num_splits

    q = q.reshape(b, h, w, c)  # [B, H, W, C]
    k = k.reshape(b, h, w, c)
    v = v.reshape(b, h, w, c)

    scale_factor = c ** 0.5

    if with_shift:
        assert attn_mask is not None # compute once
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2

        q = ops.roll(q, shifts = (-shift_size_h, -shift_size_w), dims = (1, 2))
        k = ops.roll(k, shifts = (-shift_size_h, -shift_size_w), dims = (1, 2))
        v = ops.roll(v, shifts = (-shift_size_h, -shift_size_w), dims = (1, 2))

    q = split_feature(q, num_splits = num_splits, channel_last = True) # [B * K * K, H / K, W / K, C]
    k = split_feature(k, num_splits = num_splits, channel_last = True)
    v = split_feature(v, num_splits = num_splits, channel_last= True)

    scores = (
        ops.matmul(q.reshape(b_new, -1, c), k.reshape(b_new, -1, c).permute(0, 2, 1)) / scale_factor
    ) # [B * K * K, H / K * W / K, H / K * W / K]

    if with_shift:
        scores += attn_mask.tile((b, 1, 1))

    attn = ops.softmax(scores, axis=-1)
    out = ops.matmul(attn, v.reshape(b_new, -1, c))  # [B * K * K, H / K * W / K, C]

    out = merge_splits(
        out.reshape(b_new, h // num_splits, w // num_splits, c), num_splits=num_splits, channel_last=True
    )  # [B, H, W, C]

    # shift back
    if with_shift:
        out = ops.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))

    out = out.reshape(b, -1, c)
    return out

def single_head_split_window_attention_1d(
    q,
    k,
    v,
    relative_position_bias=None,
    num_splits=1,
    with_shift=False,
    h=None,
    w=None,
    attn_mask=None,
):
    # q, k, v: [B, L, C]
    assert h is not None and w is not None
    assert q.shape[1] == h * w

    b, _, c = q.shape

    b_new = b * num_splits * h

    window_size_w = w // num_splits

    q = q.reshape(b * h, w, c)  # [B*H, W, C]
    k = k.reshape(b * h, w, c)
    v = v.reshape(b * h, w, c)

    scale_factor = c**0.5

    if with_shift:
        assert attn_mask is not None  # compute once
        shift_size_w = window_size_w // 2

        q = ops.roll(q, shifts=-shift_size_w, dims=1)
        k = ops.roll(k, shifts=-shift_size_w, dims=1)
        v = ops.roll(v, shifts=-shift_size_w, dims=1)

    q = split_feature_1d(q, num_splits=num_splits)  # [B*H*K, W/K, C]
    k = split_feature_1d(k, num_splits=num_splits)
    v = split_feature_1d(v, num_splits=num_splits)

    scores = (
        ops.matmul(q.reshape(b_new, -1, c), k.reshape(b_new, -1, c).permute(0, 2, 1)) / scale_factor
    )  # [B*H*K, W/K, W/K]

    if with_shift:
        # attn_mask: [K, W/K, W/K]
        scores += attn_mask.tile((b * h, 1, 1))  # [B*H*K, W/K, W/K]

    attn = ops.softmax(scores, axis=-1)

    out = ops.matmul(attn, v.reshape(b_new, -1, c))  # [B*H*K, W/K, C]

    out = merge_splits_1d(out, h, num_splits=num_splits)  # [B, H, W, C]

    # shift back
    if with_shift:
        out = ops.roll(out, shifts=shift_size_w, dims=2)

    out = out.reshape(b, -1, c)

    return out


class SelfAttnPropagation(nn.Cell):
    """
    flow propagation with self-attention on feature
    query: feature0, key: feature0, value: flow
    """

    def __init__(
        self,
        in_channels,
        **kwargs,
    ):
        super(SelfAttnPropagation, self).__init__()

        self.q_proj = nn.Dense(in_channels, in_channels)
        self.k_proj = nn.Dense(in_channels, in_channels)

        self._initialize_weights()

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            for param in cell.get_parameters():
                param_shape = param.shape
                param_dim = param.ndim
                param_dtype = param.dtype

                if param_dim > 1:
                    init = initializer(XavierUniform(), param_shape, param_dtype)
                    param.set_data(init)

    def construct(
        self,
        feature0,
        flow,
        local_window_attn=False,
        local_window_radius=1,
        **kwargs,
    ):
        # q, k: feature [B, C, H, W], v: flow [B, 2, H, W]
        if local_window_attn:
            return self.forward_local_window_attn(feature0, flow, local_window_radius=local_window_radius)

        b, c, h, w = feature0.shape

        query = feature0.reshape(b, c, h * w).permute(0, 2, 1)  # [B, H*W, C]

        # a note: the ``correct'' implementation should be:
        # ``query = self.q_proj(query), key = self.k_proj(query)''
        # this problem is observed while cleaning up the code
        # however, this doesn't affect the performance since the projection is a linear operation,
        # thus the two projection matrices for key can be merged
        # so I just leave it as is in order to not re-train all models :)
        query = self.q_proj(query)  # [B, H*W, C]
        key = self.k_proj(query)  # [B, H*W, C]

        value = flow.reshape(b, flow.shape[1], h * w).permute(0, 2, 1)  # [B, H*W, 2]

        scores = ops.matmul(query, key.permute(0, 2, 1)) / (c**0.5)  # [B, H*W, H*W]
        prob = ops.softmax(scores, axis=-1)

        out = ops.matmul(prob, value)  # [B, H*W, 2]
        out = out.reshape(b, h, w, value.shape[-1]).permute(0, 3, 1, 2)  # [B, 2, H, W]

        return out

    def forward_local_window_attn(
            self,
            feature0,
            flow,
            local_window_radius=1,
    ):
        assert flow.shape[1] == 2 or flow.shape[1] == 1  # flow or disparity or depth
        assert local_window_radius > 0

        b, c, h, w = feature0.shape
        value_channel = flow.shape[1]

        feature0_reshape = self.q_proj(feature0.reshape(b, c, -1).permute(0, 2, 1)).reshape(
            b * h * w, 1, c
        )  # [B*H*W, 1, C]

        kernel_size = 2 * local_window_radius + 1
        feature0_proj = self.k_proj(feature0.reshape(b, c, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, c, h,
                                                                                                              w)
        pad = ops.Pad(paddings=( # (4, 2)
        (0, 0), (0, 0), (local_window_radius, local_window_radius), (local_window_radius, local_window_radius)))
        feature0_proj_padded = pad(feature0_proj)

        unfold = nn.Unfold(ksizes=(1, kernel_size, kernel_size, 1), strides=(1, 1, 1, 1), rates=(1, 1, 1, 1),
                           padding='valid')
        feature0_window = unfold(feature0_proj_padded)  # [B, C*(kernel_size^2), H, W]

        feature0_window = feature0_window.reshape(b, c, kernel_size ** 2, h, w)
        feature0_window = feature0_window.permute(0, 3, 4, 1, 2).reshape(b * h * w, c,
                                                                           kernel_size ** 2)  # [B*H*W, C, (2R+1)^2]

        flow_padded = pad(flow)
        flow_window = unfold(flow_padded)  # [B, value_channel*(kernel_size^2), H, W]

        flow_window = flow_window.reshape(b, value_channel, kernel_size ** 2, h, w)
        flow_window = flow_window.permute(0, 3, 4, 2, 1).reshape(b * h * w, kernel_size ** 2,
                                                                   value_channel)  # [B*H*W, (2R+1)^2, value_channel]

        scores = ops.matmul(feature0_reshape, feature0_window) / (c ** 0.5)  # [B*H*W, 1, (2R+1)^2]
        prob = ops.softmax(scores, axis=-1)

        # Compute output
        out = ops.matmul(prob, flow_window).reshape(b, h, w, value_channel).permute(0, 3, 1,
                                                                                      2).contiguous()
        # [B, value_channel, H, W]

        return out
