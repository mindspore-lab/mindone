# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
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
import numbers
from typing import Dict, Optional, Tuple

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import initializer

from .activations import get_activation
from .embeddings import CombinedTimestepLabelEmbeddings, PixArtAlphaCombinedTimestepSizeEmbeddings
from .layers_compat import group_norm


class AdaLayerNorm(nn.Cell):
    r"""
    Norm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`, *optional*): The size of the embeddings dictionary.
        output_dim (`int`, *optional*):
        norm_elementwise_affine (`bool`, defaults to `False):
        norm_eps (`bool`, defaults to `False`):
        chunk_dim (`int`, defaults to `0`):
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        output_dim: Optional[int] = None,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-5,
        chunk_dim: int = 0,
    ):
        super().__init__()

        self.chunk_dim = chunk_dim
        output_dim = output_dim or embedding_dim * 2

        if num_embeddings is not None:
            self.emb = nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Dense(embedding_dim, output_dim)
        self.norm = LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def construct(
        self, x: ms.Tensor, timestep: Optional[ms.Tensor] = None, temb: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX for now.
            shift, scale = temb.chunk(2, axis=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, axis=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class AdaLayerNormZero(nn.Cell):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None, norm_type="layer_norm", bias=True):
        super().__init__()
        if num_embeddings is not None:
            self.emb = CombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = nn.Dense(embedding_dim, 6 * embedding_dim, has_bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        elif norm_type == "fp32_layer_norm":
            self.norm = FP32LayerNorm(embedding_dim, elementwise_affine=False, bias=False)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def construct(
        self,
        x: ms.Tensor,
        timestep: Optional[ms.Tensor] = None,
        class_labels: Optional[ms.Tensor] = None,
        hidden_dtype=None,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZeroSingle(nn.Cell):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type="layer_norm", bias=True):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Dense(embedding_dim, 3 * embedding_dim, has_bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(
                f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm', 'fp32_layer_norm'."
            )

    def construct(
        self,
        x: ms.Tensor,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class LuminaRMSNormZero(nn.Cell):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_eps: float, norm_elementwise_affine: bool):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Dense(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            has_bias=True,
        )
        self.norm = RMSNorm(embedding_dim, eps=norm_eps, elementwise_affine=norm_elementwise_affine)

    def construct(
        self,
        x: ms.Tensor,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        # emb = self.emb(timestep, encoder_hidden_states, encoder_mask)
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, axis=1)
        x = self.norm(x) * (1 + scale_msa[:, None])

        return x, gate_msa, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Cell):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = nn.SiLU()
        self.linear = nn.Dense(embedding_dim, 6 * embedding_dim, has_bias=True)

    def construct(
        self,
        timestep: ms.Tensor,
        added_cond_kwargs: Optional[Dict[str, ms.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype=None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(timestep, **added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_dtype)
        return self.linear(self.silu(embedded_timestep)), embedded_timestep


class AdaGroupNorm(nn.Cell):
    r"""
    GroupNorm layer modified to incorporate timestep embeddings.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
        num_groups (`int`): The number of groups to separate the channels into.
        act_fn (`str`, *optional*, defaults to `None`): The activation function to use.
        eps (`float`, *optional*, defaults to `1e-5`): The epsilon value to use for numerical stability.
    """

    def __init__(
        self, embedding_dim: int, out_dim: int, num_groups: int, act_fn: Optional[str] = None, eps: float = 1e-5
    ):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps

        if act_fn is None:
            self.act = None
        else:
            self.act = get_activation(act_fn)()

        self.linear = nn.Dense(embedding_dim, out_dim * 2)

    def construct(self, x: ms.Tensor, emb: ms.Tensor) -> ms.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, axis=1)

        x = group_norm(x, self.num_groups, None, None, self.eps)
        x = x * (1 + scale) + shift
        return x


class AdaLayerNormContinuous(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Dense(conditioning_embedding_dim, embedding_dim * 2, has_bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias=bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def construct(self, x: ms.Tensor, conditioning_embedding: ms.Tensor) -> ms.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = ops.chunk(emb, 2, axis=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class LuminaLayerNormContinuous(nn.Cell):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        out_dim: Optional[int] = None,
    ):
        super().__init__()
        # AdaLN
        self.silu = nn.SiLU()
        self.linear_1 = nn.Dense(conditioning_embedding_dim, embedding_dim, has_bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")
        # linear_2
        if out_dim is not None:
            self.linear_2 = nn.Dense(
                embedding_dim,
                out_dim,
                has_bias=bias,
            )

    def construct(
        self,
        x: ms.Tensor,
        conditioning_embedding: ms.Tensor,
    ) -> ms.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear_1(self.silu(conditioning_embedding).to(x.dtype))
        scale = emb
        x = self.norm(x) * (1 + scale)[:, None, :]

        if self.linear_2 is not None:
            x = self.linear_2(x)

        return x


class CogVideoXLayerNormZero(nn.Cell):
    def __init__(
        self,
        conditioning_dim: int,
        embedding_dim: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = nn.Dense(conditioning_dim, 6 * embedding_dim, has_bias=bias)
        self.norm = LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def construct(
        self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor, temb: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, axis=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


class LayerNorm(nn.Cell):
    r"""Applies Layer Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
    is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
    is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
    the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `ops.var(input, unbiased=False)`.

    .. note::
        Unlike Batch Normalization and Instance Normalization, which applies
        scalar scale and bias for each entire channel/plane with the
        :attr:`affine` option, Layer Normalization applies per-element scale and
        bias with :attr:`elementwise_affine`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        normalized_shape (int or list): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the last dimension which is expected to be of that specific size.
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine: a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
            The values are initialized to 1.
        bias:   the learnable bias of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 0.

    Shape:
        - Input: :math:`(N, *)`
        - Output: :math:`(N, *)` (same shape as input)

    Examples::

        >>> # NLP Example
        >>> batch, sentence_length, embedding_dim = 20, 5, 10
        >>> embedding = ops.randn(batch, sentence_length, embedding_dim)
        >>> layer_norm = LayerNorm(embedding_dim)
        >>> # Activate module
        >>> layer_norm(embedding)
        >>>
        >>> # Image Example
        >>> N, C, H, W = 20, 5, 10, 10
        >>> input = ops.randn(N, C, H, W)
        >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
        >>> # as shown in the image below
        >>> layer_norm = LayerNorm([C, H, W])
        >>> output = layer_norm(input)
    """

    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, bias=True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        _weight = np.ones(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        _bias = np.zeros(normalized_shape, dtype=ms.dtype_to_nptype(dtype))
        if self.elementwise_affine:
            self.weight = Parameter(ms.Tensor.from_numpy(_weight), name="weight")
            if bias:
                self.bias = Parameter(ms.Tensor.from_numpy(_bias), name="bias")
            else:
                self.bias = ms.Tensor.from_numpy(_bias)
        else:
            self.weight = ms.Tensor.from_numpy(_weight)
            self.bias = ms.Tensor.from_numpy(_bias)
        # TODO: In fact, we need -len(normalized_shape) instead of -1, but LayerNorm doesn't allow it.
        #  For positive axis, the ndim of input is needed. Put it in construct?
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: Tensor):
        x, _, _ = self.layer_norm(x, self.weight.to(x.dtype), self.bias.to(x.dtype))
        return x


class FP32LayerNorm(LayerNorm):
    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        origin_dtype = inputs.dtype
        x, _, _ = self.layer_norm(
            inputs.float(),
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
        )
        return x.to(origin_dtype)


class GroupNorm(nn.Cell):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = ops.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True, dtype=ms.float32):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        weight = initializer("ones", num_channels, dtype=dtype)
        bias = initializer("zeros", num_channels, dtype=dtype)
        if self.affine:
            self.weight = Parameter(weight, name="weight")
            self.bias = Parameter(bias, name="bias")
        else:
            self.weight = None
            self.bias = None

    def construct(self, x: Tensor):
        if self.affine:
            x = group_norm(x, self.num_groups, self.weight.to(x.dtype), self.bias.to(x.dtype), self.eps)
        else:
            x = group_norm(x, self.num_groups, self.weight, self.bias, self.eps)
        return x


class RMSNorm(nn.Cell):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = dim

        if elementwise_affine:
            self.weight = ms.Parameter(ops.ones(dim), name="weight")
        else:
            self.weight = None

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(ms.float32).pow(2).mean(-1, keep_dims=True)
        hidden_states = hidden_states * ops.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [ms.float16, ms.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class GlobalResponseNorm(nn.Cell):
    # Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = ms.Parameter(ops.zeros(size=(1, 1, 1, dim)), name="gamma")
        self.beta = ms.Parameter(ops.zeros(size=(1, 1, 1, dim)), name="beta")

    def construct(self, x):
        gx = ops.norm(x, ord=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(axis=-1, keep_dims=True) + 1e-6)
        out = (self.gamma * (x * nx) + self.beta + x).to(x.dtype)
        return out
