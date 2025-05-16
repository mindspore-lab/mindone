# coding=utf-8
# Copyright 2025 The Moonshot AI Team, DeepSeek-AI, and HuggingFace Inc. team. All rights reserved.
#
# The code is based on llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py), but modified for KimiVL.
#
# Licensing Information:
# - Code derived from llava (llava/modeling_llava.py) and DeepSeek-V3 (DeepSeek-V3/modeling_deepseek.py) is licensed under the Apache License, Version 2.0.
# - Other parts of the code are licensed under the MIT License.
#
# Apache License, Version 2.0:
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""MindSpore KimiVL model."""

import math
import warnings
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.distributed as dist
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeUniform, Normal, initializer

from mindone.models.utils import normal_, zeros_
from mindone.transformers.activations import ClassInstantier
from mindone.transformers.cache_utils import Cache
from mindone.transformers.generation.utils import GenerationMixin
from mindone.transformers.mindspore_adapter import dtype_to_min
from mindone.transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from mindone.transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from mindone.transformers.modeling_utils import PreTrainedModel
from mindone.transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast
from mindone.transformers.utils import logging
from mindone.utils.version_control import MS_VERSION

from .configuration_kimi_vl import DeepseekV3Config, KimiVLConfig, MoonViTConfig

logger = logging.get_logger(__name__)


ACT2CLS = {"silu": mint.nn.SiLU}
ACT2FN = ClassInstantier(ACT2CLS)


class GELUActivation(nn.Cell):
    def __init__(self, use_gelu_python: bool = False) -> None:
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = F.gelu

    def _gelu_python(self, input: ms.Tensor) -> ms.Tensor:
        return input * 0.5 * (1.0 + mint.erf(input / math.sqrt(2.0)))

    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return self.act(input)


class PytorchGELUTanh(nn.Cell):
    def construct(self, input: ms.Tensor) -> ms.Tensor:
        return F.gelu(input, approximate="tanh")


def multihead_attention(
    q: ms.Tensor, k: ms.Tensor, v: ms.Tensor, q_cu_seqlens: ms.Tensor, k_cu_seqlens: ms.Tensor
) -> ms.Tensor:
    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert k_cu_seqlens[-1] == k.shape[0] == v.shape[0], "k_cu_seqlens must sum to k.shape[0]"
    assert q.dtype in [ms.bfloat16, ms.float16], f"unsupported dtype {q.dtype} for multihead attn"

    attn_out = ops.flash_attention_score(
        q,
        k,
        v,
        q.shape[1],
        actual_seq_qlen=q_cu_seqlens,
        actual_seq_kvlen=k_cu_seqlens,
        scalar_value=1 / math.sqrt(q.shape[-1]),
        input_layout="TND",
    )
    attn_out = attn_out.flatten(start_dim=-2)

    return attn_out


def sdpa_attention(
    q: ms.Tensor, k: ms.Tensor, v: ms.Tensor, q_cu_seqlens: ms.Tensor, k_cu_seqlens: ms.Tensor
) -> ms.Tensor:
    if MS_VERSION < "2.6.0":
        logger.warning("Mindspore version is less than 2.6.0. Replace SDPA attention with Eager Attention.")
        return eager_attention(q, k, v, q_cu_seqlens, k_cu_seqlens)

    # Unified format legal check
    assert q.dim() == k.dim() == v.dim() == 3, "q, k, v must have 3 dims"
    assert q_cu_seqlens[-1] == q.shape[0], "q_cu_seqlens must sum to q.shape[0]"
    assert k_cu_seqlens[-1] == k.shape[0] == v.shape[0], "k_cu_seqlens must sum to k.shape[0]"

    # MS 2.6 above
    attn_output = ops.speed_fusion_attention(
        q, k, v, q.shape[1], "TND", actual_seq_qlen=q_cu_seqlens, actual_seq_kvlen=k_cu_seqlens
    )
    attn_output = attn_output.flatten(start_dim=-2)
    return attn_output


def eager_attention(
    q: ms.Tensor, k: ms.Tensor, v: ms.Tensor, q_cu_seqlens: ms.Tensor, k_cu_seqlens: ms.Tensor
) -> ms.Tensor:
    seq_length = q.shape[0]
    # the original repo's mask is not correct. We fix here.
    attention_mask = mint.full([1, seq_length, k.shape[0]], dtype_to_min(q.dtype).item(), dtype=q.dtype)
    for i in range(1, len(q_cu_seqlens)):
        attention_mask[..., q_cu_seqlens[i - 1] : q_cu_seqlens[i], k_cu_seqlens[i - 1] : k_cu_seqlens[i]] = 0
    q = q.transpose(0, 1)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)

    attn_weight = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    attn_weight += attention_mask
    attn_weight = mint.softmax(attn_weight, dim=-1, dtype=ms.float32).to(q.dtype)

    attn_output = attn_weight @ v
    attn_output = attn_output.transpose(0, 1)
    attn_output = attn_output.reshape(seq_length, -1)
    return attn_output


VL_VISION_ATTENTION_FUNCTIONS = {
    "flash_attention_2": multihead_attention,
    "sdpa": sdpa_attention,
    "eager": eager_attention,
}


def complex_mult(a: ms.Tensor, b: ms.Tensor) -> ms.Tensor:
    a_real, a_complex = mint.unbind(a, dim=-1)
    b_real, b_complex = mint.unbind(b, dim=-1)
    out_real = a_real * b_real - a_complex * b_complex
    out_complex = a_real * b_complex + b_real * a_complex
    return mint.stack([out_real, out_complex], dim=-1)


def apply_rope(xq: ms.Tensor, xk: ms.Tensor, freqs_cis: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
    """
    Args: (The leading dimensions of all inputs should be the same)
        xq: query, tensor of shape (..., num_heads, head_dim)
        xk: key, tensor of shape (..., num_heads, head_dim)
        freqs_cis: tensor of shape (..., head_dim/2, 2). It contains the precomputed cis(freqs) for each position in the 2D grid.
    Returns:
        xq_out, xk_out: tensors of shape (..., num_heads, head_dim)
    """
    freqs_cis = freqs_cis.unsqueeze(-3)  # ..., 1, head_dim/2, 2
    # ..., num_heads, head_dim/2
    xq_ = xq.float().view(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().view(*xk.shape[:-1], -1, 2)
    xq_out = complex_mult(xq_, freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    xk_out = complex_mult(xk_, freqs_cis).flatten(-2)  # ..., num_heads, head_dim
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Learnable2DInterpPosEmb(nn.Cell):
    def __init__(self, height: int, width: int, dim: int, interpolation_mode: str = "bicubic") -> None:
        super().__init__()
        self.height = height
        self.width = width
        self.interpolation_mode = interpolation_mode
        self.weight = ms.Parameter(initializer(Normal(sigma=1.0), shape=(height, width, dim), dtype=ms.float32))

    def construct(self, x: ms.Tensor, grid_hws: ms.Tensor) -> ms.Tensor:
        pos_embs = []
        for shape in grid_hws.tolist():
            if shape == self.weight.shape[:-1]:
                pos_embs.append(self.weight.flatten(end_dim=1))
            else:
                pos_embs.append(
                    F.interpolate(
                        self.weight.permute((2, 0, 1)).unsqueeze(0),
                        size=shape,
                        mode=self.interpolation_mode,
                    )
                    .squeeze(0)
                    .permute((1, 2, 0))
                    .flatten(end_dim=1)
                )
        out = x + mint.cat(pos_embs)
        return out


class MoonVisionPatchEmbed(nn.Cell):
    def __init__(
        self,
        out_dim: int,
        in_dim: int = 3,
        patch_size: Union[int, Tuple[int, int]] = (14, 14),
        pos_emb_height: int = 14,
        pos_emb_width: int = 14,
    ) -> None:
        super().__init__()
        assert isinstance(patch_size, (int, Sequence)), f"Invalid patch_size type: {type(patch_size)}"
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        assert len(patch_size) == 2, f"Expected patch_size to be a tuple of 2, got {patch_size}"
        self.patch_size = patch_size

        self.proj = mint.nn.Conv2d(in_dim, out_dim, kernel_size=patch_size, stride=patch_size)

        self.pos_emb = Learnable2DInterpPosEmb(height=pos_emb_height, width=pos_emb_width, dim=out_dim)

    def construct(self, x: ms.Tensor, grid_hws: ms.Tensor) -> ms.Tensor:
        """
        Args:
            x (L, Channels): input tensor
            grid_hws (N, 2): grid height and width

        Returns:
            (L, Cout) tensor
        """
        x = self.proj(x).view(x.shape[0], -1)
        # apply positional embedding
        x = self.pos_emb(x, grid_hws)
        return x


class Rope2DPosEmb(nn.Cell):
    """2D rotary position embedding with multi-resolution support.

    This class is intended to be used in the following way:
    1. Before training, create an instance of Rope2DPosEmb. This instance will hold the precomputed cis.
    2. Before each construct pass, call `get_freqs_cis_by_*` to get the `freqs_cis` tensor for this iteration.
    3. During the construct pass, pass the `freqs_cis` tensor to each attention layer, and call `apply` just before each attention operation.
        The rope is shared across all attention layers and all heads.

    Refs:
    - RoFormer: https://arxiv.org/abs/2104.09864
    - VisionLLaMA: https://arxiv.org/abs/2403.00522
    - https://github.com/Meituan-AutoML/VisionLLaMA/blob/main/dit/models.py

    Args:
        dim (int): usually the multi-head attention dimension, should be divisible by 4 (TODO: relax this constraint if needed)
        max_height (int): the maximum height of the 2D grid
        max_width (int): the maximum width of the 2D grid
        theta_base (float): the base of the theta
    """

    def __init__(self, dim: int, max_height: int, max_width: int, theta_base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        assert self.dim % 4 == 0, "dim must be divisible by 4"
        self.max_height = max_height
        self.max_width = max_width
        self.theta_base = theta_base

        self.freqs_cis = self._precompute_freqs_cis()

    def _precompute_freqs_cis(self) -> ms.Tensor:
        """Calculate the cis(freqs) for each position in the 2D grid.

        Return: complex tensor of shape (max_height, max_width, dim//2) and value:
            height axis: ret[h, w, 2*i] = cis(h * theta_base**(-4*i/dim))
            weight axis: ret[h, w, 2*i+1] = cis(w * theta_base**(-4*i/dim))   with (i in [0, dim//4))
            note: `cis` is a mathematical notation defined by cis x = cos x + i sin x,
        """
        N = self.max_height * self.max_width
        flat_pos = mint.arange(0, N).float()
        x_pos = flat_pos % self.max_width
        y_pos = flat_pos // self.max_width
        dim_range = mint.arange(0, self.dim, 4)[: (self.dim // 4)].float()  # C/4
        freqs = 1.0 / (self.theta_base ** (dim_range / self.dim))
        x_freqs = mint.outer(x_pos, freqs).float()  # N, C/4
        y_freqs = mint.outer(y_pos, freqs).float()  # N, C/4
        x_cis = mint.polar(mint.ones_like(x_freqs), x_freqs)  # N, C/4
        y_cis = mint.polar(mint.ones_like(y_freqs), y_freqs)  # N, C/4
        freqs_cis = mint.stack([x_cis, y_cis], dim=-1)  # N, C/4, 2
        freqs_cis = freqs_cis.reshape(self.max_height, self.max_width, -1)  # max_height, max_width, C/2
        freqs_cis = ops.view_as_real(freqs_cis)  # max_height, max_width, C/2, 2
        return freqs_cis

    def construct(self, grid_hws: ms.Tensor) -> ms.Tensor:
        """
        Args:
            grid_hws (ms.Tensor): grid height and width

        Returns:
            freqs_cis: tensor of shape (sum(t * height * width), dim//2, 2)
        """
        shapes = grid_hws.tolist()
        assert all(1 <= h <= self.max_height and 1 <= w <= self.max_width for h, w in shapes), (
            shapes,
            self.max_height,
            self.max_width,
        )
        freqs_cis = mint.cat([self.freqs_cis[:h, :w].reshape(-1, self.dim // 2, 2) for h, w in shapes], dim=0)
        return freqs_cis


class MLP2(nn.Cell):
    """
    Args:
        dims: [in_dim, hidden_dim, out_dim]
        bias: whether to use bias in linear layer.
    """

    def __init__(
        self, dims: list[int], activation: Union[nn.Cell, Callable[[ms.Tensor], ms.Tensor]], bias: bool = True
    ) -> None:
        super().__init__()
        assert len(dims) == 3
        self.fc0 = mint.nn.Linear(dims[0], dims[1], bias=bias)
        self.fc1 = mint.nn.Linear(dims[1], dims[2], bias=bias)
        self.activation = activation
        for m in [self.fc0, self.fc1]:
            normal_(m.weight, std=math.sqrt(2 / m.in_features))
            if m.bias is not None:
                zeros_(m.bias)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.fc0(x)
        x = self.activation(x)
        return self.fc1(x)


class MoonVitEncoderLayer(nn.Cell):
    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        *,
        attn_implementation: str = "eager",
        activation: Union[nn.Cell, Callable[[ms.Tensor], ms.Tensor]] = F.gelu,
        attn_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.hidden_size_per_attention_head = self.hidden_dim // self.num_heads
        self.attn_implementation = attn_implementation

        self.norm0 = mint.nn.LayerNorm(hidden_dim)
        self.norm1 = mint.nn.LayerNorm(hidden_dim)
        self.mlp = MLP2([hidden_dim, mlp_dim, hidden_dim], activation)
        self.wqkv = mint.nn.Linear(hidden_dim, hidden_dim * 3, bias=attn_bias)
        self.wo = mint.nn.Linear(hidden_dim, hidden_dim, bias=attn_bias)

    def attention_qkvpacked(
        self,
        x: ms.Tensor,
        cu_seqlens: ms.Tensor,
        rope_freqs_cis: ms.Tensor,
    ) -> ms.Tensor:
        """
        Args:
            x (ms.Tensor): (batch_size, seqlen, hidden_dim)
            cu_seqlens (ms.Tensor):
        """
        xqkv = self.wqkv(x)

        qkv_shape = xqkv.shape[:-1] + (
            3,
            self.num_heads,
            self.hidden_size_per_attention_head,
        )
        # xqkv: (batch_size, seqlen, 3, nheads, headdim)
        xqkv = xqkv.view(*qkv_shape)
        xq, xk, xv = mint.unbind(xqkv, dim=-3)

        xq, xk = apply_rope(xq, xk, rope_freqs_cis)

        attn_func = VL_VISION_ATTENTION_FUNCTIONS[self.attn_implementation]
        attn_out = attn_func(xq, xk, xv, q_cu_seqlens=cu_seqlens, k_cu_seqlens=cu_seqlens)

        attn_out = self.wo(attn_out)
        return attn_out

    def construct(
        self,
        hidden_states: ms.Tensor,
        cu_seqlens: ms.Tensor,
        rope_freqs_cis: ms.Tensor,
    ) -> ms.Tensor:
        """
        Args:
            hidden_states: non-packed (B, N, D) or packed (L, D). if non-packed, seqlens should be None, if packed, seqlens should be set

        Returns:
            output: same shape of input, non-packed (B, N, D) for non-packed input, (L, D) for packed input
        """
        residual = hidden_states
        hidden_states = self.norm0(hidden_states)
        attn_out = self.attention_qkvpacked(hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis)
        hidden_states = residual + attn_out

        residual = hidden_states
        hidden_states = self.mlp(self.norm1(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class MoonVitEncoder(nn.Cell):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        block_cfg: dict,
    ) -> None:
        super().__init__()

        self.rope_2d = Rope2DPosEmb(block_cfg["hidden_dim"] // block_cfg["num_heads"], 512, 512)
        self.blocks = nn.CellList([MoonVitEncoderLayer(**block_cfg) for _ in range(num_layers)])
        self.final_layernorm = mint.nn.LayerNorm(hidden_dim)

    def construct(self, hidden_states: ms.Tensor, grid_hws: ms.Tensor) -> ms.Tensor:
        rope_freqs_cis = self.rope_2d(grid_hws=grid_hws)

        lengths = mint.cat((mint.zeros(1, dtype=grid_hws.dtype), grid_hws[:, 0] * grid_hws[:, 1]))
        cu_seqlens = lengths.cumsum(dim=0, dtype=ms.int32)

        for _, block in enumerate(self.blocks):
            hidden_states = block(hidden_states, cu_seqlens, rope_freqs_cis=rope_freqs_cis)

        hidden_states = self.final_layernorm(hidden_states)

        return hidden_states


def patch_merger(
    x: ms.Tensor,
    grid_hws: ms.Tensor,
    merge_kernel_size: list[int, int] = (2, 2),
) -> List[ms.Tensor]:
    d_model = x.shape[-1]

    outputs = []
    pre_sum = 0
    for x_shape in grid_hws.tolist():
        height, width = x_shape[0], x_shape[1]
        # Get the current sequence
        seq = x[pre_sum : pre_sum + height * width]
        # Reshape along self.merge_kernel_size and concat to the last dimension
        kernel_height, kernel_width = merge_kernel_size
        new_height, new_width = height // kernel_height, width // kernel_width
        reshaped_seq = seq.view(new_height, kernel_height, new_width, kernel_width, d_model)
        reshaped_seq = reshaped_seq.permute(0, 2, 1, 3, 4).contiguous()
        padded_seq = reshaped_seq.view(new_height * new_width, kernel_height * kernel_width, -1)
        outputs.append(padded_seq)
        pre_sum += height * width

    return outputs


class DeepseekV3RMSNorm(nn.Cell):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        """
        DeepseekV3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = ms.Parameter(np.ones(hidden_size, dtype=np.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        input_dtype = hidden_states.dtype
        result, _ = ops.rms_norm(hidden_states.to(ms.float32), self.weight.to(ms.float32), self.variance_epsilon)
        return result.to(input_dtype)


class DeepseekV3RotaryEmbedding(nn.Cell):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0) -> None:
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (np.arange(0, self.dim, 2, dtype=np.float32) / self.dim))
        self.inv_freq = ms.Tensor(inv_freq, dtype=ms.float32)
        self.max_seq_len_cached = ms.Tensor(max_position_embeddings)
        self.cos_cached = ms.Tensor(np.zeros((max_position_embeddings, dim), dtype=np.float32))
        self.sin_cached = ms.Tensor(np.zeros((max_position_embeddings, dim), dtype=np.float32))
        self._set_cos_sin_cache(max_position_embeddings)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        ops.assign(self.max_seq_len_cached, seq_len)
        t = mint.arange(self.max_seq_len_cached.item(), dtype=ms.float32)

        freqs = mint.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mint.cat((freqs, freqs), dim=-1)
        ops.assign(self.cos_cached, mint.cos(emb))
        ops.assign(self.sin_cached, mint.sin(emb))

    def construct(self, x: ms.Tensor, seq_len: int) -> Tuple[ms.Tensor, ms.Tensor]:
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, dtype=x.dtype)

        return (self.cos_cached[:seq_len].to(x.dtype), self.sin_cached[:seq_len].to(x.dtype))


class DeepseekV3LinearScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
    """DeepseekV3RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0, scaling_factor: float = 1.0
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        ops.assign(self.max_seq_len_cached, seq_len)
        t = mint.arange(self.max_seq_len_cached, dtype=ms.float32)
        t = t / self.scaling_factor

        freqs = mint.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mint.cat((freqs, freqs), dim=-1)
        ops.assign(self.cos_cached, mint.cos(emb))
        ops.assign(self.sin_cached, mint.sin(emb))


class DeepseekV3DynamicNTKScalingRotaryEmbedding(DeepseekV3RotaryEmbedding):
    """DeepseekV3RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        ops.assign(self.max_seq_len_cached, seq_len)

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (mint.arange(0, self.dim, 2).float() / self.dim))
            ops.assign(self.inv_freq, inv_freq)

        t = mint.arange(self.max_seq_len_cached, dtype=ms.float32)

        freqs = mint.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = mint.cat((freqs, freqs), dim=-1)
        ops.assign(self.cos_cached, mint.cos(emb))
        ops.assign(self.sin_cached, mint.sin(emb))


# Inverse dim formula to find dim based on number of rotations
def yarn_find_correction_dim(
    num_rotations: int, dim: int, base: float = 10000.0, max_position_embeddings: int = 2048
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


# Find dim range bounds based on rotations
def yarn_find_correction_range(
    low_rot: int, high_rot: int, dim: int, base: float = 10000.0, max_position_embeddings: int = 2048
) -> Tuple[float, float]:
    low = math.floor(yarn_find_correction_dim(low_rot, dim, base=base, max_position_embeddings=max_position_embeddings))
    high = math.ceil(
        yarn_find_correction_dim(high_rot, dim, base=base, max_position_embeddings=max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_linear_ramp_mask(min: float, max: float, dim: int) -> ms.Tensor:
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (mint.arange(dim, dtype=ms.float32) - min) / (max - min)
    ramp_func = mint.clamp(linear_func, 0, 1)
    return ramp_func


class DeepseekV3YarnRotaryEmbedding(DeepseekV3RotaryEmbedding):
    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = mscale
        self.mscale_all_dim = mscale_all_dim
        super().__init__(dim, max_position_embeddings=max_position_embeddings, base=base)

    def _set_cos_sin_cache(self, seq_len: int) -> None:
        ops.assign(self.max_seq_len_cached, seq_len)
        dim = self.dim

        freq_extra = 1.0 / (self.base ** (mint.arange(0, dim, 2, dtype=ms.float32) / dim))
        freq_inter = 1.0 / (self.scaling_factor * self.base ** (mint.arange(0, dim, 2, dtype=ms.float32) / dim))

        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            dim,
            base=self.base,
            max_position_embeddings=self.original_max_position_embeddings,
        )
        inv_freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dim // 2)
        inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask
        ops.assign(self.inv_freq, inv_freq)

        t = mint.arange(seq_len, dtype=ms.float32)

        freqs = mint.outer(t, inv_freq)

        _mscale = float(
            yarn_get_mscale(self.scaling_factor, self.mscale)
            / yarn_get_mscale(self.scaling_factor, self.mscale_all_dim)
        )

        emb = mint.cat((freqs, freqs), dim=-1)
        ops.assign(self.cos_cached, mint.cos(emb) * _mscale)
        ops.assign(self.sin_cached, mint.sin(emb) * _mscale)


def apply_rotary_pos_emb(
    q: ms.Tensor, k: ms.Tensor, cos: ms.Tensor, sin: ms.Tensor, position_ids: ms.Tensor, unsqueeze_dim: int = 1
) -> Tuple[ms.Tensor, ms.Tensor]:
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`ms.Tensor`): The query tensor.
        k (`ms.Tensor`): The key tensor.
        cos (`ms.Tensor`): The cosine part of the rotary embedding.
        sin (`ms.Tensor`): The sine part of the rotary embedding.
        position_ids (`ms.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(ms.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    b, h, s, d = q.shape
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

    q_embed = ops.rotary_position_embedding(q, cos, sin)
    k_embed = ops.rotary_position_embedding(k, cos, sin)
    return q_embed, k_embed


class DeepseekV3MLP(nn.Cell):
    def __init__(
        self, config: DeepseekV3Config, hidden_size: Optional[int] = None, intermediate_size: Optional[int] = None
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = mint.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = mint.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class MoEGate(nn.Cell):
    def __init__(self, config: DeepseekV3Config) -> None:
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux
        self.topk_method = config.topk_method
        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = ms.Parameter(
            initializer(
                HeUniform(negative_slope=math.sqrt(5)), shape=(self.n_routed_experts, self.gating_dim), dtype=ms.float32
            )
        )
        if self.topk_method == "noaux_tc":
            self.e_score_correction_bias = ms.Parameter(np.zeros((self.n_routed_experts,), dtype=np.float32))

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(ms.float32), self.weight.type(ms.float32), bias=None)
        if self.scoring_func == "sigmoid":
            scores = mint.sigmoid(logits)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        # select top-k experts
        if self.topk_method == "noaux_tc":
            assert not self.training
            scores_for_choice = scores.view(bsz * seq_len, -1) + self.e_score_correction_bias.unsqueeze(0)
            group_scores = (
                scores_for_choice.view(bsz * seq_len, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
            )  # [n, n_group]
            group_idx = mint.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]
            group_mask = mint.zeros_like(group_scores)  # [n, n_group]
            group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
            score_mask = (
                group_mask.unsqueeze(-1)
                .expand((bsz * seq_len, self.n_group, self.n_routed_experts // self.n_group))
                .reshape(bsz * seq_len, -1)
            )  # [n, e]
            tmp_scores = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)  # [n, e]
            _, topk_idx = mint.topk(tmp_scores, k=self.top_k, dim=-1, sorted=False)
            topk_weight = scores.gather(1, topk_idx)
        elif self.topk_method == "greedy":
            topk_weight, topk_idx = mint.topk(scores, k=self.top_k, dim=-1, sorted=False)
        else:
            raise NotImplementedError(f"insupportable TopK function for MoE gating: {self.topk_method}")

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        topk_weight = topk_weight * self.routed_scaling_factor  # must multiply the scaling factor

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = mint.zeros(bsz, self.n_routed_experts)
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    mint.ones(bsz, seq_len * aux_topk),
                ).div_(seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(nn.Cell):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    def construct(self, x: ms.Tensor, loss: ms.Tensor):
        return NotImplementedError()


class DeepseekV3MoE(nn.Cell):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: DeepseekV3Config) -> None:
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config.num_experts_per_tok

        if hasattr(config, "ep_size") and config.ep_size > 1:
            assert config.ep_size == dist.get_world_size()
            self.ep_size = config.ep_size
            self.experts_per_rank = config.n_routed_experts // config.ep_size
            self.ep_rank = dist.get_rank()
            self.experts = nn.CellList(
                [
                    (
                        DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                        if i >= self.ep_rank * self.experts_per_rank and i < (self.ep_rank + 1) * self.experts_per_rank
                        else None
                    )
                    for i in range(config.n_routed_experts)
                ]
            )
        else:
            self.ep_size = 1
            self.experts_per_rank = config.n_routed_experts
            self.ep_rank = 0
            self.experts = nn.CellList(
                [
                    DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                    for i in range(config.n_routed_experts)
                ]
            )
        self.gate = MoEGate(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=intermediate_size)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, aux_loss = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if self.training:
            flat_topk_idx = topk_idx.view(-1)
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = mint.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        if self.config.n_shared_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x: ms.Tensor, topk_ids: ms.Tensor, topk_weight: ms.Tensor):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        if self.ep_size > 1:
            tokens_per_ep_rank = tokens_per_expert.view(self.ep_size, -1).sum(dim=1)
            tokens_per_expert_group = tokens_per_expert.new_empty((tokens_per_expert.shape[0],))
            dist.all_to_all_single(tokens_per_expert_group, tokens_per_expert)
            output_splits = tokens_per_expert_group.view(self.ep_size, -1).sum(1).tolist()
            gathered_tokens = sorted_tokens.new_empty(
                (tokens_per_expert_group.sum(dim=0).item(), sorted_tokens.shape[1])
            )
            input_split_sizes = tokens_per_ep_rank.tolist()
            dist.all_to_all(
                list(gathered_tokens.split(output_splits)),
                list(sorted_tokens.split(input_split_sizes)),
            )
            tokens_per_expert_post_gather = tokens_per_expert_group.view(self.ep_size, self.experts_per_rank).sum(dim=0)
            gatherd_idxs = np.zeros(shape=(gathered_tokens.shape[0],), dtype=np.int32)
            s = 0
            for i, k in enumerate(tokens_per_expert_group.tolist()):
                gatherd_idxs[s : s + k] = i % self.experts_per_rank
                s += k
            gatherd_idxs = gatherd_idxs.argsort()
            sorted_tokens = gathered_tokens[gatherd_idxs]
            tokens_per_expert = tokens_per_expert_post_gather
        tokens_per_expert = tokens_per_expert.tolist()

        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i + self.ep_rank * self.experts_per_rank]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx

        outs = mint.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty((0,))
        if self.ep_size > 1:
            new_x = mint.empty_like(outs)
            new_x[gatherd_idxs] = outs
            gathered_tokens = new_x.new_empty(sorted_tokens_shape)
            dist.all_to_all(
                list(gathered_tokens.split(input_split_sizes)),
                list(new_x.split(output_splits)),
            )
            outs = gathered_tokens

        new_x = mint.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class DeepseekV3Attention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: DeepseekV3Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the construct call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        self.is_causal = True

        if self.q_lora_rank is None:
            self.q_proj = mint.nn.Linear(self.hidden_size, self.num_heads * self.q_head_dim, bias=False)
        else:
            self.q_a_proj = mint.nn.Linear(self.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_a_layernorm = DeepseekV3RMSNorm(config.q_lora_rank)
            self.q_b_proj = mint.nn.Linear(config.q_lora_rank, self.num_heads * self.q_head_dim, bias=False)

        self.kv_a_proj_with_mqa = mint.nn.Linear(
            self.hidden_size,
            config.kv_lora_rank + config.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = DeepseekV3RMSNorm(config.kv_lora_rank)
        self.kv_b_proj = mint.nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = mint.nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self._init_rope()

        self.softmax_scale = self.q_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.softmax_scale = self.softmax_scale * mscale * mscale

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = DeepseekV3RotaryEmbedding(
                self.qk_rope_head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = DeepseekV3LinearScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = DeepseekV3DynamicNTKScalingRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                kwargs = {
                    key: self.config.rope_scaling[key]
                    for key in [
                        "original_max_position_embeddings",
                        "beta_fast",
                        "beta_slow",
                        "mscale",
                        "mscale_all_dim",
                    ]
                    if key in self.config.rope_scaling
                }
                self.rotary_emb = DeepseekV3YarnRotaryEmbedding(
                    self.qk_rope_head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: ms.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.v_head_dim).transpose(1, 2).contiguous()

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.shape

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = mint.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = mint.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = mint.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty((bsz, self.num_heads, q_len, self.q_head_dim))
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty((bsz, self.num_heads, q_len, self.q_head_dim))
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attn_weights = mint.matmul(query_states, key_states.transpose(2, 3)) * self.softmax_scale

        if attn_weights.shape != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.shape}"
            )
        assert attention_mask is not None
        if attention_mask.shape != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.shape}"
            )
        attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = mint.matmul(attn_weights, value_states)

        if attn_output.shape != (bsz, self.num_heads, q_len, self.v_head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.v_head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class DeepseekV3FlashAttention2(DeepseekV3Attention):
    """
    DeepseekV3 flash attention module. This module inherits from `DeepseekV3Attention` as the weights of the module stays
    untouched. The only required change would be on the construct pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.shape

        if self.q_lora_rank is None:
            q = self.q_proj(hidden_states)
        else:
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = mint.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        compressed_kv, k_pe = mint.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_b_proj(self.kv_a_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )

        k_nope, value_states = mint.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        kv_seq_len = value_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        query_states = k_pe.new_empty((bsz, self.num_heads, q_len, self.q_head_dim))
        query_states[:, :, :, : self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim :] = q_pe

        key_states = k_pe.new_empty((bsz, self.num_heads, q_len, self.q_head_dim))
        key_states[:, :, :, : self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim :] = k_pe

        if self.q_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.q_head_dim - self.v_head_dim])

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.is_causal and attention_mask is None:
            if q_len > 1:
                attention_mask = 1 - mint.tril(mint.ones((1, 1, q_len, kv_seq_len), dtype=ms.uint8))
            else:
                attention_mask = None

        attn_output = ops.flash_attention_score(
            query_states,
            key_states,
            value_states,
            self.num_heads,
            attn_mask=attention_mask,
            keep_prob=1 - self.attention_dropout,
            scalar_value=self.softmax_scale,
            input_layout="BNSD",
        )

        if self.q_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


ATTENTION_CLASSES = {
    "eager": DeepseekV3Attention,
    "flash_attention_2": DeepseekV3FlashAttention2,
}


class DeepseekV3DecoderLayer(nn.Cell):
    def __init__(self, config: DeepseekV3Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            DeepseekV3MoE(config)
            if (
                config.n_routed_experts is not None
                and layer_idx >= config.first_k_dense_replace
                and layer_idx % config.moe_layer_freq == 0
            )
            else DeepseekV3MLP(config)
        )
        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:
        """
        Args:
            hidden_states (`ms.Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`ms.Tensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(ms.Tensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class DeepseekV3PreTrainedModel(PreTrainedModel):
    config_class = DeepseekV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, mint.nn.Linear):
            normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, mint.nn.Embedding):
            normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0


class DeepseekV3Model(DeepseekV3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`DeepseekV3DecoderLayer`]

    Args:
        config: DeepseekV3Config
    """

    def __init__(self, config: DeepseekV3Config) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = mint.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.CellList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if use_cache:
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            position_ids = mint.arange(past_key_values_length, seq_length + past_key_values_length, dtype=ms.int64)
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
        else:
            # 4d mask is passed through the layers
            if past_key_values_length == 0:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
            else:
                # simply use a all zeros mask for decoding
                attention_mask = mint.zeros((batch_size, 1, 1, past_key_values_length + 1), dtype=inputs_embeds.dtype)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class DeepseekV3ForCausalLM(DeepseekV3PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = mint.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

        >>> model = DeepseekV3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: ms.Tensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[ms.Tensor] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        if past_key_values is not None:
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_seq_length()

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


class MoonVitVLProjector(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        merge_kernel_size: Tuple[int, int],
        hidden_act: str = "gelu",
        ln_eps: float = 1e-5,
        out_dim: int = 4096,
    ) -> None:
        super().__init__()
        self.hidden_size = in_channels * merge_kernel_size[0] * merge_kernel_size[1]

        self.pre_norm = mint.nn.LayerNorm(in_channels, eps=ln_eps)
        self.linear_1 = mint.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = ACT2FN[hidden_act]
        self.linear_2 = mint.nn.Linear(self.hidden_size, out_dim, bias=True)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.pre_norm(hidden_states).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


class MoonVitPretrainedModel(PreTrainedModel):
    config_class = MoonViTConfig
    model_type = "moonvit"
    _no_split_modules = ["PackingTransformer"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, config: MoonViTConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config = deepcopy(config)
        self.merge_kernel_size = config.merge_kernel_size
        self.patch_size = config.patch_size
        self.patch_embed = MoonVisionPatchEmbed(
            out_dim=config.hidden_size,
            patch_size=config.patch_size,
            pos_emb_height=config.init_pos_emb_height,
            pos_emb_width=config.init_pos_emb_width,
        )

        self.encoder = MoonVitEncoder(
            hidden_dim=config.hidden_size,
            num_layers=config.num_hidden_layers,
            block_cfg={
                "num_heads": config.num_attention_heads,
                "hidden_dim": config.hidden_size,
                "mlp_dim": config.intermediate_size,
                "activation": PytorchGELUTanh(),
                "attn_bias": True,
                "attn_implementation": config._attn_implementation,
            },
        )

    def construct(self, pixel_values: ms.Tensor, grid_hws: ms.Tensor) -> ms.Tensor:
        """
        Args:
            pixel_values (ms.Tensor): The input pixel values.
            grid_hws (ms.Tensor): The grid height and width.

        Returns:
            ms.Tensor: The output tokens.
        """
        hidden_states = self.patch_embed(pixel_values, grid_hws)
        hidden_states = self.encoder(hidden_states, grid_hws)
        hidden_states = patch_merger(hidden_states, grid_hws, merge_kernel_size=self.merge_kernel_size)
        return hidden_states


class KimiVLMultiModalProjector(nn.Cell):
    def __init__(self, config: KimiVLConfig) -> None:
        super().__init__()

        self.hidden_size = (
            config.vision_config.hidden_size
            * config.vision_config.merge_kernel_size[0]
            * config.vision_config.merge_kernel_size[1]
        )

        self.pre_norm = mint.nn.LayerNorm(config.vision_config.hidden_size, eps=1e-05)
        self.linear_1 = mint.nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.act = GELUActivation()
        self.linear_2 = mint.nn.Linear(self.hidden_size, config.text_config.hidden_size, bias=True)

    def construct(self, image_features: List[ms.Tensor]) -> ms.Tensor:
        image_features = mint.cat(image_features, dim=0)
        hidden_states = self.pre_norm(image_features).view(-1, self.hidden_size)
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


class KimiVLPreTrainedModel(PreTrainedModel, GenerationMixin):
    config_class = KimiVLConfig
    base_model_prefix = "model"
    _no_split_modules = ["MoonVitEncoderLayer", "DeepseekV3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        # important: this ported version of Llava isn't meant for training from scratch - only
        # inference and fine-tuning - so the proper init weights code has been removed - the original codebase
        # https://github.com/haotian-liu/LLaVA/tree/main/llava should serve for that purpose
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            normal_(module.class_embedding, mean=0.0, std=std)

        if isinstance(module, (mint.nn.Linear, mint.nn.Conv2d)):
            normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, mint.nn.Embedding):
            normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight[module.padding_idx] = 0

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        return self.language_model._supports_sdpa


class KimiVLForConditionalGeneration(KimiVLPreTrainedModel, GenerationMixin):
    def __init__(self, config: KimiVLConfig) -> None:
        super().__init__(config)
        vision_config: MoonViTConfig = config.vision_config
        self.vision_tower = MoonVitPretrainedModel(vision_config)
        self.multi_modal_projector = KimiVLMultiModalProjector(config)
        self.language_model = DeepseekV3ForCausalLM(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        return self.language_model.get_decoder()

    def tie_weights(self):
        return self.language_model.tie_weights()

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> mint.nn.Embedding:
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # update vocab size
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    def _merge_with_image_features(
        self,
        inputs_embeds: ms.Tensor,
        input_ids: ms.Tensor,
        image_features: ms.Tensor,
    ) -> ms.Tensor:
        """
        Args:
            inputs_embeds (:obj:`ms.Tensor` of shape :obj:`(batch_size, sequence_length, input_embed_dim)`):
                The input embeddings.
            input_ids (:obj:`ms.Tensor` of shape :obj:`(batch_size, sequence_length)`):
                The input ids.
            image_features (:obj:`ms.Tensor` of shape :obj:`(image_token_nums, image_feature_dim)`):
                The image features to merge with the input embeddings.
        """
        image_token_index = self.config.media_placeholder_token_id

        batch_size, sequence_length, input_embed_dim = inputs_embeds.shape
        image_feature_nums, image_feature_dim = image_features.shape

        assert image_feature_dim == input_embed_dim

        image_token_nums = (input_ids == image_token_index).sum().item()
        assert image_feature_nums == image_token_nums

        # (batch_size, sequence_length, input_embed_dim) -> (batch_size * sequence_length, input_embed_dim)
        inputs_embeds = inputs_embeds.reshape(-1, input_embed_dim)

        # (batch_size, sequence_length) -> (batch_size * sequence_length)
        input_ids = input_ids.flatten()

        inputs_embeds[input_ids == image_token_index] = image_features

        inputs_embeds = inputs_embeds.reshape((batch_size, sequence_length, input_embed_dim))

        return inputs_embeds

    def _extract_image_features(self, pixel_values: ms.Tensor, image_grid_hws: ms.Tensor) -> ms.Tensor:
        """
        Args:
            pixel_values (:obj:`ms.Tensor` of shape :obj:`(image_token_nums, 3, patch_size, patch_size)`):
                The pixel values of the images processed by image processor.

        Returns:
            image_features (:obj:`ms.Tensor` of shape :obj:`(image_token_nums, image_feature_dim)`):
                The selected image features to use as input to the projector head.
        """
        # [(image_token_nums_0, image_feature_dim), (image_token_nums_1, image_feature_dim), ...]
        image_features = self.vision_tower(pixel_values, image_grid_hws)
        # (image_token_nums_0 + image_token_nums_1 + ..., image_feature_dim)
        image_features = self.multi_modal_projector(image_features)
        return image_features

    def construct(
        self,
        input_ids: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Union[None, ms.Tensor, List[ms.Tensor]] = None,
        image_grid_hws: Optional[ms.Tensor] = None,
    ) -> Union[tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:
        ```python
        >>> from PIL import Image

        >>> # generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> # decode
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None and pixel_values.shape[0] > 0:
            pixel_values = pixel_values.to(self.vision_tower.dtype)

            image_features: ms.Tensor = self._extract_image_features(pixel_values, image_grid_hws)
            inputs_embeds = inputs_embeds.to(image_features[0].dtype)
            inputs_embeds = self._merge_with_image_features(inputs_embeds, input_ids, image_features)

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        attention_mask=None,
        image_grid_hws=None,
        cache_position=None,
        **kwargs,
    ):
        model_inputs = self.language_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            **kwargs,
        )

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["image_grid_hws"] = image_grid_hws

        return model_inputs
