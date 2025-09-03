# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

from packaging.version import parse

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, nn, ops

from .activations import get_activation
from .embeddings import CombinedTimestepLabelEmbeddings, PixArtAlphaCombinedTimestepSizeEmbeddings

MINDSPORE_VERSION = parse(ms.__version__)


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
            self.emb = mint.nn.Embedding(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, output_dim)
        self.norm = mint.nn.LayerNorm(output_dim // 2, norm_eps, norm_elementwise_affine)

    def construct(
        self, x: ms.Tensor, timestep: Optional[ms.Tensor] = None, temb: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        if self.emb is not None:
            temb = self.emb(timestep)

        temb = self.linear(self.silu(temb))

        if self.chunk_dim == 1:
            # This is a bit weird why we have the order of "shift, scale" here and "scale, shift" in the
            # other if-branch. This branch is specific to CogVideoX and OmniGen for now.
            shift, scale = temb.chunk(2, dim=1)
            shift = shift[:, None, :]
            scale = scale[:, None, :]
        else:
            scale, shift = temb.chunk(2, dim=0)

        x = self.norm(x) * (1 + scale) + shift
        return x


class FP32LayerNorm(mint.nn.LayerNorm):
    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(
            inputs.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(origin_dtype)


class SD35AdaLayerNormZeroX(nn.Cell):
    r"""
    Norm layer adaptive layer norm zero (AdaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, norm_type: str = "layer_norm", bias: bool = True) -> None:
        super().__init__()

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, 9 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = mint.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")

    def construct(
        self,
        hidden_states: ms.Tensor,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ...]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp, shift_msa2, scale_msa2, gate_msa2 = emb.chunk(
            9, dim=1
        )
        norm_hidden_states = self.norm(hidden_states)
        hidden_states = norm_hidden_states * (1 + scale_msa[:, None]) + shift_msa[:, None]
        norm_hidden_states2 = norm_hidden_states * (1 + scale_msa2[:, None]) + shift_msa2[:, None]
        return hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2


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

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = mint.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
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
        hidden_dtype: Optional[ms.Type] = None,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        # x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = self.norm(x) * (1 + scale_msa.expand_dims(axis=1)) + shift_msa.expand_dims(axis=1)
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

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        if norm_type == "layer_norm":
            self.norm = mint.nn.LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
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
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        # x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        x = self.norm(x) * (1 + scale_msa.expand_dims(axis=1)) + shift_msa.expand_dims(axis=1)
        return x, gate_msa


class LuminaRMSNormZero(nn.Cell):
    """
    Norm layer adaptive RMS normalization zero.

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
    """

    def __init__(self, embedding_dim: int, norm_eps: float, norm_elementwise_affine: bool):
        super().__init__()
        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(
            min(embedding_dim, 1024),
            4 * embedding_dim,
            bias=True,
        )
        self.norm = RMSNorm(embedding_dim, eps=norm_eps)

    def construct(
        self,
        x: ms.Tensor,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        emb = self.linear(self.silu(emb))
        scale_msa, gate_msa, scale_mlp, gate_mlp = emb.chunk(4, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None])

        return x, gate_msa, scale_mlp, gate_mlp


class AdaLayerNormSingle(nn.Cell):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://huggingface.co/papers/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 3, use_additional_conditions=use_additional_conditions
        )

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

    def construct(
        self,
        timestep: ms.Tensor,
        added_cond_kwargs: Optional[Dict[str, ms.Tensor]] = None,
        batch_size: Optional[int] = None,
        hidden_dtype=None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        # No modulation happening here.
        added_cond_kwargs = added_cond_kwargs or {"resolution": None, "aspect_ratio": None}
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
            self.act = get_activation(act_fn)

        self.linear = mint.nn.Linear(embedding_dim, out_dim * 2)

    def construct(self, x: ms.Tensor, emb: ms.Tensor) -> ms.Tensor:
        if self.act:
            emb = self.act(emb)
        emb = self.linear(emb)
        emb = emb[:, :, None, None]
        scale, shift = emb.chunk(2, dim=1)

        x = F.group_norm(x, self.num_groups, eps=self.eps)
        x = x * (1 + scale) + shift
        return x


class AdaLayerNormContinuous(nn.Cell):
    r"""
    Adaptive normalization layer with a norm layer (layer_norm or rms_norm).

    Args:
        embedding_dim (`int`): Embedding dimension to use during projection.
        conditioning_embedding_dim (`int`): Dimension of the input condition.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        eps (`float`, defaults to 1e-5): Epsilon factor.
        bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        norm_type (`str`, defaults to `"layer_norm"`):
            Normalization layer to use. Values supported: "layer_norm", "rms_norm".
    """

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
        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias=bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps, elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def construct(self, x: ms.Tensor, conditioning_embedding: ms.Tensor) -> ms.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = mint.chunk(emb, 2, dim=1)
        # x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        x = self.norm(x) * (1 + scale).expand_dims(axis=1) + shift.expand_dims(axis=1)
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
        self.silu = mint.nn.SiLU()
        self.linear_1 = mint.nn.Linear(conditioning_embedding_dim, embedding_dim, bias=bias)

        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

        self.linear_2 = None
        if out_dim is not None:
            self.linear_2 = mint.nn.Linear(embedding_dim, out_dim, bias=bias)

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


class CogView3PlusAdaLayerNormZeroTextImage(nn.Cell):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, dim: int):
        super().__init__()

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, 12 * dim, bias=True)
        self.norm_x = mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)
        self.norm_c = mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-5)

    def construct(
        self,
        x: ms.Tensor,
        context: ms.Tensor,
        emb: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor, ms.Tensor]:
        emb = self.linear(self.silu(emb))
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = emb.chunk(12, dim=1)
        normed_x = self.norm_x(x)
        normed_context = self.norm_c(context)
        x = normed_x * (1 + scale_msa[:, None]) + shift_msa[:, None]
        context = normed_context * (1 + c_scale_msa[:, None]) + c_shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp, context, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp


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

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(conditioning_dim, 6 * embedding_dim, bias=bias)
        self.norm = mint.nn.LayerNorm(embedding_dim, eps=eps, elementwise_affine=elementwise_affine)

    def construct(
        self, hidden_states: ms.Tensor, encoder_hidden_states: ms.Tensor, temb: ms.Tensor
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
        hidden_states = self.norm(hidden_states) * (1 + scale)[:, None, :] + shift[:, None, :]
        encoder_hidden_states = self.norm(encoder_hidden_states) * (1 + enc_scale)[:, None, :] + enc_shift[:, None, :]
        return hidden_states, encoder_hidden_states, gate[:, None, :], enc_gate[:, None, :]


if MINDSPORE_VERSION >= parse("2.3.0"):
    LayerNorm = mint.nn.LayerNorm
else:

    class LayerNorm(nn.Cell):
        r"""
        LayerNorm with the bias parameter.

        Args:
            dim (`int`): Dimensionality to use for the parameters.
            eps (`float`, defaults to 1e-5): Epsilon factor.
            elementwise_affine (`bool`, defaults to `True`):
                Boolean flag to denote if affine transformation should be applied.
            bias (`bias`, defaults to `True`): Boolean flag to denote if bias should be use.
        """

        def __init__(self, dim, eps: float = 1e-5, elementwise_affine: bool = True, bias: bool = True):
            super().__init__()

            self.eps = eps

            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            self.dim = tuple(dim)

            if elementwise_affine:
                self.weight = ms.Parameter(mint.ones(dim), name="weight")
                self.bias = ms.Parameter(mint.zeros(dim), name="bias") if bias else None
            else:
                self.weight = None
                self.bias = None

        def construct(self, input):
            return F.layer_norm(input, self.dim, self.weight, self.bias, self.eps)


if MINDSPORE_VERSION == parse("2.5.0"):

    class RMSNorm(nn.Cell):
        r"""
        RMS Norm as introduced in https://huggingface.co/papers/1910.07467 by Zhang et al.

        Args:
            dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
            eps (`float`): Small value to use when calculating the reciprocal of the square-root.
            elementwise_affine (`bool`, defaults to `True`):
                Boolean flag to denote if affine transformation should be applied.
            bias (`bool`, defaults to False): If also training the `bias` param.
        """

        def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
            super().__init__()

            self.eps = eps
            self.elementwise_affine = elementwise_affine

            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            self.dim = tuple(dim)

            self.weight = None
            self.bias = None

            if elementwise_affine:
                self.weight = ms.Parameter(mint.ones(dim), name="weight")
                if bias:
                    self.bias = ms.Parameter(mint.zeros(dim), name="bias")

        def construct(self, hidden_states):
            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [ms.float16, ms.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                weight = self.weight
            else:
                weight = mint.ones(hidden_states.shape[-1], dtype=hidden_states.dtype)
            hidden_states = ops.rms_norm(hidden_states, weight, epsilon=self.eps)[0]
            if self.bias is not None:
                hidden_states = hidden_states + self.bias

            return hidden_states

else:

    class RMSNorm(nn.Cell):
        def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
            super().__init__()

            self.eps = eps
            self.elementwise_affine = elementwise_affine

            if isinstance(dim, numbers.Integral):
                dim = (dim,)

            self.dim = dim

            self.weight = None
            self.bias = None

            if elementwise_affine:
                self.weight = ms.Parameter(mint.ones(dim), name="weight")
                if bias:
                    self.bias = ms.Parameter(mint.zeros(dim), name="bias")

        def construct(self, hidden_states):
            input_dtype = hidden_states.dtype
            # variance = hidden_states.to(ms.float32).pow(2).mean(-1, keep_dims=True)
            variance = mint.pow(hidden_states.to(ms.float32), 2).mean(-1, keepdim=True)
            hidden_states = hidden_states * mint.rsqrt(variance + self.eps)

            if self.weight is not None:
                # convert into half-precision if necessary
                if self.weight.dtype in [ms.float16, ms.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                hidden_states = hidden_states * self.weight
                if self.bias is not None:
                    hidden_states = hidden_states + self.bias
            else:
                hidden_states = hidden_states.to(input_dtype)

            return hidden_states


# TODO: (Dhruv) This can be replaced with regular RMSNorm in Mochi once `_keep_in_fp32_modules` is supported
# for sharded checkpoints, see: https://github.com/huggingface/diffusers/issues/10013
class MochiRMSNorm(nn.Cell):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = dim

        if elementwise_affine:
            self.weight = ms.Parameter(mint.ones(dim), name="weight")
        else:
            self.weight = None

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(ms.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * mint.rsqrt(variance + self.eps)

        if self.weight is not None:
            hidden_states = hidden_states * self.weight
        hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class GlobalResponseNorm(nn.Cell):
    r"""
    Global response normalization as introduced in ConvNeXt-v2 (https://huggingface.co/papers/2301.00808).

    Args:
        dim (`int`): Number of dimensions to use for the `gamma` and `beta`.
    """

    # Taken from https://github.com/facebookresearch/ConvNeXt-V2/blob/3608f67cc1dae164790c5d0aead7bf2d73d9719b/models/utils.py#L105
    def __init__(self, dim):
        super().__init__()
        self.gamma = ms.Parameter(mint.zeros((1, 1, 1, dim)), name="gamma")
        self.beta = ms.Parameter(mint.zeros((1, 1, 1, dim)), name="beta")

    def construct(self, x):
        gx = mint.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        out = (self.gamma * (x * nx) + self.beta + x).to(x.dtype)
        return out


class LpNorm(nn.Cell):
    def __init__(self, p: int = 2, dim: int = -1, eps: float = 1e-12):
        super().__init__()

        self.p = p
        self.dim = dim
        self.eps = eps

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        return F.normalize(hidden_states, p=self.p, dim=self.dim, eps=self.eps)


def get_normalization(
    norm_type: str = "batch_norm",
    num_features: Optional[int] = None,
    eps: float = 1e-5,
    elementwise_affine: bool = True,
    bias: bool = True,
) -> nn.Cell:
    if norm_type == "rms_norm":
        norm = RMSNorm(num_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    elif norm_type == "layer_norm":
        norm = mint.nn.LayerNorm(num_features, eps=eps, elementwise_affine=elementwise_affine, bias=bias)
    elif norm_type == "batch_norm":
        norm = mint.nn.BatchNorm2d(num_features, eps=eps, affine=elementwise_affine)
    else:
        raise ValueError(f"{norm_type=} is not supported.")
    return norm
