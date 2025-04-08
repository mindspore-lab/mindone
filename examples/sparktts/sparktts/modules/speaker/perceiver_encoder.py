# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import namedtuple
from functools import wraps

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Parameter, mint, nn
from mindspore.common.initializer import Normal, initializer


def exists(val):
    return val is not None


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)

    return inner


print_once = once(print)

# main class


class Attend(nn.Cell):
    def __init__(self, dropout=0.0, causal=False, use_flash=False):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = mint.nn.Dropout(dropout)

        self.causal = causal
        self.mask = None

        self.use_flash = use_flash

        # determine efficient attention configs for cuda and cpu
        self.config = namedtuple(
            "EfficientAttentionConfig",
            ["enable_flash", "enable_math", "enable_mem_efficient"],
        )
        self.cpu_config = self.config(True, True, True)
        self.cuda_config = None

    def get_mask(self, n):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = mint.ones((n, n), dtype=ms.bool_).triu(1)
        self.mask = mask
        return mask

    def construct(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n = q.shape[-2]

        scale = q.shape[-1] ** -0.5

        kv_einsum_eq = "b j d" if k.ndim == 3 else "b h j d"

        # similarity

        sim = mint.einsum(f"b h i d, {kv_einsum_eq} -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            mask = mask.unsqueeze(1).unsqueeze(1)
            sim = sim.masked_fill(~mask, -mint.finfo(sim.dtype).max)

        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n)
            sim = sim.masked_fill(causal_mask, -mint.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(axis=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = mint.einsum(f"b h i j, {kv_einsum_eq} -> b h i d", attn, v)

        return out


def Sequential(*mods):
    return nn.SequentialCell(*filter(exists, mods))


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


class RMSNorm(nn.Cell):
    def __init__(self, dim, scale=True, dim_cond=None):
        super().__init__()
        self.cond = exists(dim_cond)
        self.to_gamma_beta = mint.nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim**0.5
        self.gamma = Parameter(mint.ones(dim)) if scale else None

    def construct(self, x, cond=None):
        gamma = default(self.gamma, 1)
        out = F.normalize(x, dim=-1) * self.scale * gamma

        if not self.cond:
            return out

        assert exists(cond)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)
        gamma, beta = map(lambda t: mint.unsqueeze(t, 1), (gamma, beta))
        return out * gamma + beta


class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (kernel_size,) = self.kernel_size
        (dilation,) = self.dilation
        (stride,) = self.stride

        assert stride == 1
        self.causal_padding = dilation * (kernel_size - 1)

    def construct(self, x):
        causal_padded_x = F.pad(x, (self.causal_padding, 0), value=0.0)
        return super().forward(causal_padded_x)


class GEGLU(nn.Cell):
    def construct(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x


def FeedForward(dim, mult=4, causal_conv=False):
    dim_inner = int(dim * mult * 2 / 3)

    conv = None
    if causal_conv:
        conv = nn.SequentialCell(
            lambda t: mint.permute(t, (0, 2, 1)),
            CausalConv1d(dim_inner, dim_inner, 3),
            lambda t: mint.permute(t, (0, 2, 1)),
        )

    return Sequential(mint.nn.Linear(dim, dim_inner * 2), GEGLU(), conv, mint.nn.Linear(dim_inner, dim))


class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        *,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        dropout=0.0,
        use_flash=False,
        cross_attn_include_queries=False,
    ):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        self.cross_attn_include_queries = cross_attn_include_queries

        dim_inner = dim_head * heads
        dim_context = default(dim_context, dim)

        self.attend = Attend(causal=causal, dropout=dropout, use_flash=use_flash)
        self.to_q = mint.nn.Linear(dim, dim_inner, bias=False)
        self.to_kv = mint.nn.Linear(dim_context, dim_inner * 2, bias=False)
        self.to_out = mint.nn.Linear(dim_inner, dim, bias=False)

    def construct(self, x, context=None, mask=None):
        h, has_context = self.heads, exists(context)

        context = default(context, x)

        if has_context and self.cross_attn_include_queries:
            context = mint.cat((x, context), dim=-2)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))

        b, n, _ = q.shape
        q = q.view(b, n, h, -1).permute(0, 2, 1, 3)
        b, n, _ = k.shape
        k = k.view(b, n, h, -1).permute(0, 2, 1, 3)
        b, n, _ = v.shape
        v = v.view(b, n, h, -1).permute(0, 2, 1, 3)
        # q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = self.attend(q, k, v, mask=mask)

        b_out, h_out, n_out, d_out = out.shape
        out = out.permute(0, 2, 1, 3).reshape(b_out, n_out, h_out * d_out)
        return self.to_out(out)


class PerceiverResampler(nn.Cell):
    def __init__(
        self,
        *,
        dim,
        depth=2,
        dim_context=None,
        num_latents=32,
        dim_head=64,
        heads=8,
        ff_mult=4,
        use_flash_attn=False,
    ):
        super().__init__()
        dim_context = default(dim_context, dim)

        self.proj_context = mint.nn.Linear(dim_context, dim) if dim_context != dim else nn.Identity()

        self.latents = Parameter(mint.randn(num_latents, dim))
        self.latents.set_data(
            initializer(
                Normal(sigma=0.02),
                shape=self.latents.shape,
                dtype=self.latents.dtype,
            )
        )

        self.layers = nn.CellList([])
        for _ in range(depth):
            self.layers.append(
                nn.CellList(
                    [
                        Attention(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            use_flash=use_flash_attn,
                            cross_attn_include_queries=True,
                        ),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

        self.norm = RMSNorm(dim)

    def construct(self, x, mask=None):
        batch = x.shape[0]

        x = self.proj_context(x)
        if batch > 1:
            latents = self.latents.unsqueeze(0).repeat((batch, 1, 1))
        else:
            latents = self.latents.unsqueeze(0)
        # latents = repeat(self.latents, "n d -> b n d", b=batch)

        for attn, ff in self.layers:
            latents = attn(latents, x, mask=mask) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


if __name__ == "__main__":
    model = PerceiverResampler(dim=256, dim_context=80)
    x = mint.randn(8, 200, 80)
    out = model(x)
    print(out.shape)  # [8, 32, 80]

    num_params = sum(param.numel() for param in model.parameters())
    print("{} M".format(num_params / 1e6))
