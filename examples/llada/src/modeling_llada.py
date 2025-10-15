from __future__ import annotations

import logging
import math
import sys
import warnings
from abc import abstractmethod
from dataclasses import fields
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple, Union, cast

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor, mint
from mindspore.common.initializer import Constant, HeNormal, Normal, TruncatedNormal, initializer

from mindone.transformers import MSPreTrainedModel
from mindone.transformers.cache_utils import Cache
from mindone.transformers.mindspore_adapter.utils import _DTYPE_2_MAX, _DTYPE_2_MIN
from mindone.transformers.modeling_outputs import CausalLMOutputWithPast
from mindone.transformers.models.auto import AutoModel
from mindone.transformers.utils import is_flash_attn_2_available
from mindone.utils.version_control import check_valid_flash_attention

FLASH_IS_AVAILABLE = is_flash_attn_2_available and check_valid_flash_attention()

if FLASH_IS_AVAILABLE:
    from mindone.models.modules.flash_attention import MSFlashAttention

from .configuration_llada import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    InitFnType,
    LayerNormType,
    LLaDAConfig,
    ModelConfig,
    StrEnum,
)

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "GemmaRMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "LLaDABlock",
    "LLaDASequentialBlock",
    "LLaDAModel",
    "LLaDAOutput",
    "LLaDAGenerateOutput",
]


log = logging.getLogger(__name__)


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, dtype=None, training=True
):
    # force dtype(fp16 or bf16) precision calculation
    ori_dtype = query.dtype
    if dtype is not None:
        query, key, value = query.astype(dtype), key.astype(dtype), value.astype(dtype)

    if attn_mask is not None:
        if attn_mask.dtype == ms.bool_:
            attn_mask = attn_mask.to(ms.float32)
            attn_mask = attn_mask.masked_fill((1 - attn_mask).to(ms.bool_), _DTYPE_2_MIN[ms.float16])
        attn_mask = attn_mask.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.transpose(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_mask,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)
    else:
        L, S = query.shape[-2], key.shape[-2]
        attn_bias = mint.zeros((L, S), dtype=query.dtype)
        if is_causal:
            # assert attn_mask is None
            temp_mask = mint.ones((L, S), dtype=ms.bool_).tril(diagonal=0)
            attn_bias = ops.masked_fill(attn_bias, mint.logical_not(temp_mask), _DTYPE_2_MIN[ms.float16])
            attn_bias = attn_bias.to(query.dtype)

        attn_weight = mint.nn.functional.softmax(
            mint.matmul(query, mint.transpose(key, -2, -1)) / (query.shape[-1] ** 0.5) + attn_bias,
            dim=-1,
            dtype=ms.float32,
        ).astype(query.dtype)

    attn_weight = mint.nn.functional.dropout(attn_weight, p=dropout_p, training=training)

    out = mint.matmul(attn_weight, value)
    out = out.astype(ori_dtype)

    return out


def constant_(tensor: Parameter, val: float) -> None:
    tensor.set_data(initializer(Constant(val), tensor.shape, tensor.dtype))


def normal_(tensor: Parameter, mean: float = 0.0, std: float = 1.0) -> None:
    tensor.set_data(initializer(Normal(sigma=std, mean=mean), tensor.shape, tensor.dtype))


def trunc_normal_(tensor: Parameter, mean: float = 0.0, std: float = 1.0, a=-2, b=2) -> None:
    tensor.set_data(initializer(TruncatedNormal(sigma=std, mean=mean, a=a, b=b), tensor.shape, tensor.dtype))


def kaiming_normal_(tensor: Parameter, a: float = 0, mode: str = "fan_in", nonlinearity: str = "leaky_relu") -> None:
    tensor.set_data(
        initializer(HeNormal(negative_slope=a, mode=mode, nonlinearity=nonlinearity), tensor.shape, tensor.dtype)
    )


def init_weights(
    config: ModelConfig,
    module: Union[mint.nn.Linear, mint.nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, mint.nn.Linear):
        if module.bias is not None:
            constant_(module.bias, 0.0)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            module.weight.set_data(module.weight.value() / math.sqrt(2 * config.n_layers))


def ensure_finite_(x: Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x = ops.masked_fill(x, x == float("-inf"), _DTYPE_2_MIN[x.dtype])
    if check_pos_inf:
        x = ops.masked_fill(x, x == float("inf"), _DTYPE_2_MAX[x.dtype])
    return x


class BufferCache(dict, MutableMapping[str, Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> str:
    return None


class LayerNormBase(nn.Cell):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = Parameter(mint.ones(self.normalized_shape, dtype=ms.float32))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = Parameter(mint.zeros(self.normalized_shape, dtype=ms.float32))
            else:
                self.bias = None
        else:
            self.bias = None
            self.weight = None

    @abstractmethod
    def construct(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.gemma_rms:
            return GemmaRMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def reset_parameters(self):
        if self.weight is not None:
            constant_(self.weight, 1.0)
        if self.bias is not None:
            constant_(self.bias, 0.0)


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: Tensor) -> Tensor:
        ori_dtype = x.dtype
        if self.low_precision:
            x = x.to(ms.bfloat16)
            weight = self.weight.to(ms.bfloat16) if self.weight is not None else None
            bias = self.bias.to(ms.bfloat16) if self.bias is not None else None
            return self.layer_norm(x, weight, bias)[0].to(ori_dtype)

        else:
            return self.layer_norm(x.to(ms.float32), self.weight.to(ms.float32), self.bias.to(ms.float32))[0].to(
                ori_dtype
            )


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def construct(self, x: Tensor) -> Tensor:
        og_dtype = x.dtype
        x = x.to(ms.float32)
        variance = mint.mean(mint.pow(x, 2), -1, keepdim=True)
        x = x * mint.rsqrt(variance + self.eps)
        x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class GemmaRMSLayerNorm(LayerNormBase):
    """
    Gemma RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=config.rms_norm_eps)

    def construct(self, x: Tensor) -> Tensor:
        og_dtype = x.dtype
        x = x.to(ms.float32)
        variance = mint.mean(mint.pow(x, 2), -1, keepdim=True)
        x = x * mint.rsqrt(variance + self.eps)
        x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return x * (1 + self.weight) + self.bias
            else:
                return x * (1 + self.weight)
        else:
            return x


class RotaryEmbedding(nn.Cell):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.rope_theta = config.rope_theta

        self.dim = self.config.d_model // self.config.n_heads
        self.inv_freq = 1.0 / (self.rope_theta ** (mint.arange(0, self.dim, 2, dtype=ms.float32) / self.dim))

    def get_rotary_embedding(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        seq = mint.arange(seq_len, dtype=ms.float32)
        # freqs = einsum("i , j -> i j", seq, inv_freq)
        freqs = ops.outer(seq, self.inv_freq)
        positions = mint.cat((freqs, freqs), dim=-1)
        pos_sin, pos_cos = mint.sin(positions)[None, None, :, :], mint.cos(positions)[None, None, :, :]

        return pos_sin, pos_cos

    def rotate_half(self, x: Tensor) -> Tensor:
        B, nh, T, hs = x.shape
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = mint.unbind(x, -2)
        return mint.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: Tensor, pos_cos: Tensor, t: Tensor) -> Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def construct(self, q: Tensor, k: Tensor) -> Tuple[Tensor, Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
        pos_sin, pos_cos = self.get_rotary_embedding(key_len)
        pos_sin = pos_sin.to(q_.dtype)
        pos_cos = pos_cos.to(q_.dtype)
        if query_len == key_len:
            q_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, q_)
        else:
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
        k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.to(q.dtype), k_.to(k.dtype)


class Activation(nn.Cell):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def construct(self, x: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU())
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU())
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class GELU(mint.nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(mint.nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(mint.nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def construct(self, x: Tensor) -> Tensor:
        x, gate = mint.chunk(x, 2, dim=-1)
        return mint.nn.functional.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int) -> Tensor:
    att_bias = mint.triu(
        mint.ones((seq_len, seq_len), dtype=ms.float32),
        diagonal=1,
    )
    att_bias = ops.masked_fill(att_bias, att_bias == 1, _DTYPE_2_MIN[att_bias.dtype])
    return att_bias.view(1, 1, seq_len, seq_len)


def get_causal_attention_bias(seq_len: int) -> Tensor:
    causal_bias = causal_attention_bias(seq_len)

    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig) -> Tensor:
    alibi_bias = mint.arange(1 - seq_len, 1, dtype=ms.float32).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - mint.arange(1 - seq_len, 1, dtype=ms.float32).view(1, 1, seq_len, 1)
    alibi_bias = mint.abs(alibi_bias) * (-1)

    # shape: (n_heads,)
    m = mint.arange(1, config.n_heads + 1, dtype=ms.float32)
    m = m * (config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))


class LLaDABlock(nn.Cell):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        # Dropout.
        self.dropout = mint.nn.Dropout(p=config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = mint.nn.Linear(config.d_model, config.d_model, bias=config.include_bias)

        # Feed-forward output projection.
        self.ff_out = mint.nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)
        self.flash_attn_func = None
        if config.flash_attention:
            try:
                self.flash_attn_func = MSFlashAttention
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_channels,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    @classmethod
    def _cast_attn_bias(cls, bias: Tensor, input_dtype) -> Tensor:
        target_dtype = input_dtype

        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if self.flash_attn_func is not None and attn_mask is None:
            # q shape (B N S D)
            num_heads, head_dim = q.shape[1], q.shape[-1]
            r = self.flash_attn_func(
                head_dim=head_dim,
                head_num=num_heads,
                input_layout="BNSD",
                dtype=ms.float16,
                attention_dropout=dropout_p if self.training else 0.0,
            )(q, k, v)

            return r
        else:
            # sdpa doesn't support GQA, so we're doing this
            assert k.shape[1] == v.shape[1]
            num_kv_heads = k.shape[1]
            num_q_heads = q.shape[1]
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1)

            # Modify: MDM set causal to False, and with no attn_mask.
            return scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=False,
                training=self.training,
            )

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_bias: Optional[Tensor] = None,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, T, C = q.shape  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).astype(dtype)
            k = self.k_norm(k).astype(dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = mint.transpose(q.view(B, T, self.config.n_heads, C // self.config.n_heads), 1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = mint.transpose(k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads), 1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = mint.transpose(v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads), 1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = mint.cat((past_key, k), dim=-2)
            v = mint.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            if query_len == key_len:
                attention_bias = self._cast_attn_bias(attention_bias, dtype)
            else:
                attention_bias = self._cast_attn_bias(
                    attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
                )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=False,
        )

        # Re-assemble all head outputs side-by-side.
        att = mint.transpose(att, 1, 2).view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @abstractmethod
    def construct(
        self,
        x: Tensor,
        attention_bias: Optional[Tensor] = None,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> LLaDABlock:
        if config.block_type == BlockType.sequential:
            return LLaDASequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return LLaDALlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class LLaDASequentialBlock(LLaDABlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = mint.nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias | config.include_qkv_bias
        )
        # Feed-forward input projection.
        self.ff_proj = mint.nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

    def construct(
        self,
        x: Tensor,
        attention_bias: Optional[Tensor] = None,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        q, k, v = mint.split(self.att_proj(self.attn_norm(x)), self.fused_dims, dim=-1)

        # Get attention scores.
        # shape: (batch_size, seq_len, d_model)
        att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        x = self.ff_norm(x)
        x = self.ff_proj(x)
        x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class LLaDALlamaBlock(LLaDABlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `LLaDASequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * head_dim
        v_proj_out_dim = config.effective_n_kv_heads * head_dim
        self.q_proj = mint.nn.Linear(config.d_model, q_proj_out_dim, bias=config.include_bias | config.include_qkv_bias)
        self.k_proj = mint.nn.Linear(config.d_model, k_proj_out_dim, bias=config.include_bias | config.include_qkv_bias)
        self.v_proj = mint.nn.Linear(config.d_model, v_proj_out_dim, bias=config.include_bias | config.include_qkv_bias)

        # Feed-forward input projection.
        self.ff_proj = mint.nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias)
        # new add
        self.up_proj = mint.nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.up_proj, d=self.config.d_model, layer_id=None)  # new add

    def construct(
        self,
        x: Tensor,
        attention_bias: Optional[Tensor] = None,
        layer_past: Optional[Tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Get attention scores.
        # shape: (batch_size, seq_len, d_model)
        att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x)  # new add
        x = self.act(x)
        x = x * x_up  # new add
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class LLaDAOutput(NamedTuple):
    logits: Tensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[Tensor, Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[Tensor]]
    """
    Hidden states from each block.
    """


class LLaDAGenerateOutput(NamedTuple):
    token_ids: Tensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: Tensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class LLaDABlockGroup(nn.CellList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Cell]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None

    def construct(
        self,
        x: Tensor,
        attention_bias: Optional[Tensor] = None,
        layers_past: Optional[List[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[Tensor, Optional[List[Tuple[Tensor, Tensor]]]]:
        attn_key_values: Optional[List[Tuple[Tensor, Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            # if (
            #     (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
            #     or (
            #         self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
            #         and block_idx % 2 == 0
            #     )
            #     or (
            #         self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
            #         and block_idx % 3 == 0
            #     )
            #     or (
            #         self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
            #         and block_idx % 4 == 0
            #     )
            # ):
            #     # shape: (batch_size, seq_len, d_model)
            #     x, cache = self._activation_checkpoint_fn(  # type: ignore
            #         block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
            #     )
            # else:
            # shape: (batch_size, seq_len, d_model)
            x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class Transformer(nn.Cell):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        self.wte = mint.nn.Embedding(config.embedding_size or config.vocab_size, config.d_model)
        self.emb_drop = mint.nn.Dropout(p=config.embedding_dropout)
        self.ln_f = LayerNorm.build(config)

        blocks = [LLaDABlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                LLaDABlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.block_groups = nn.CellList(block_groups)
        else:
            self.blocks = nn.CellList(blocks)

        if not (self.config.alibi or self.config.rope):
            self.wpe = mint.nn.Embedding(config.max_sequence_length, config.d_model)
        if not config.weight_tying:
            self.ff_out = mint.nn.Linear(
                config.d_model,
                config.embedding_size or config.vocab_size,
                bias=config.include_bias,
            )


class LLaDAModel(nn.Cell):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise Exception("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise Exception("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise Exception("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        # self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)
        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise Exception("n layers must be divisible by block group size")
        # torch.backends.cuda.enable_flash_sdp(True)
        # torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = Transformer(config, self.__cache)
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

        # Warm up cache.
        if self.config.alibi:
            self.causal_bias = get_causal_attention_bias(config.max_sequence_length)
            self.alibi_bias = self.get_alibi_attention_bias(config.max_sequence_length)

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int) -> Tensor:
        if hasattr(self, "alibi_bias") and self.alibi_bias is not None and self.alibi_bias.shape[-1] >= seq_len:
            return self.alibi_bias
        alibi_bias = alibi_attention_bias(seq_len, self.config)
        return alibi_bias

    def construct(
        self,
        input_ids: Tensor,
        input_embeddings: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
    ) -> LLaDAOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        # Add Basic MDM Model config check
        assert not self.config.alibi, "Alibi length extrapolation is not supported for MDM."
        assert self.config.rope, "Rope must be used in Llama-Encoder for MDM."
        assert past_key_values is None and not use_cache, "The kvcache is not suppotred for MDM."

        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.shape if input_embeddings is None else input_embeddings.shape[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].shape[-2]

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings

        if self.config.input_emb_norm:
            x = x * (self.config.d_model**0.5)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = mint.arange(past_length, past_length + seq_len, dtype=ms.int32).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None and 0.0 in attention_mask:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.astype(ms.float32).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * _DTYPE_2_MIN[attention_mask.dtype]
        else:
            attention_mask = None

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache,
                    past_length + seq_len,
                ) + self.get_alibi_attention_bias(past_length + seq_len)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len)
            elif attention_bias.dtype in (ms.int8, ms.bool_):
                attention_bias = attention_bias.astype(ms.float32)
                attention_bias = ops.masked_fill(
                    attention_bias, attention_bias == 0.0, _DTYPE_2_MIN[attention_bias.dtype]
                )

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].astype(ms.float32)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                attention_bias = ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[Tensor, Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                # if (
                #     (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                #     or (
                #         self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                #         and block_idx % 2 == 0
                #     )
                #     or (
                #         self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                #         and block_idx % 3 == 0
                #     )
                #     or (
                #         self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                #         and block_idx % 4 == 0
                #     )
                # ):
                #     # shape: (batch_size, seq_len, d_model)
                #     x, cache = self._activation_checkpoint_fn(
                #         block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                #     )
                # else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache)
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = mint.nn.functional.linear(x, self.transformer.wte.embedding_table, None)
        else:
            logits = self.transformer.ff_out(x)
        if self.config.scale_logits:
            logits = logits * (1 / math.sqrt(self.config.d_model))

        return LLaDAOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )


def create_model_config_from_pretrained_config(config: LLaDAConfig):
    """
    Utility function
    """

    kwargs = {}
    for field in fields(ModelConfig):
        kwargs[field.name] = getattr(config, field.name)

    model_config = ModelConfig(**kwargs)
    return model_config


class LLaDAModelLM(MSPreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = LLaDAConfig
    base_model_prefix = "model"
    _no_split_modules = ["LLaDABlock", "LLaDASequentialBlock", "LLaDALlamaBlock"]

    def __init__(self, config: LLaDAConfig, model: Optional[LLaDAModel] = None, init_params: bool = False):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            self.model = LLaDAModel(model_config, init_params=init_params)
        else:
            self.model = model

    def construct(
        self,
        input_ids: Tensor = None,
        inputs_embeds: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        past_key_values: Optional[List[Tensor]] = None,
        labels: Optional[Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: bool = True,
        cache_position: Optional[Cache] = None,  # This is a hack mitigation of an issue in transformers `4.39.x`
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        if output_attentions:
            raise ValueError("output_attentions is not yet supported in LLaDA")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.construct(
            input_ids=input_ids,
            input_embeddings=inputs_embeds,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states

        loss = None
        if labels is not None:
            warnings.warn("Note that for LLaDA, you cannot calculate the loss here.", UserWarning)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.attn_key_values,
            hidden_states=hidden_states,
        )

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(self, input_ids: Tensor, past_key_values: Optional[List[Tuple]] = None, **kwargs):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    def get_input_embeddings(self) -> nn.Cell:
        return self.model.transformer.wte

    def set_input_embeddings(self, value: nn.Cell):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        if self.config.weight_tying:
            return self.model.transformer.wte
        else:
            return self.model.transformer.ff_out

    def set_output_embeddings(self, value: nn.Cell):
        if self.config.weight_tying:
            self.model.transformer.wte = value
        else:
            self.model.transformer.ff_out = value

    def tie_weights(self):
        if self.config.weight_tying:
            self.model.transformer.ff_out = self.model.transformer.wte


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModel.register(LLaDAConfig, LLaDAModelLM)
