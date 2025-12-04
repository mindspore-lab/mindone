# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import contextlib
import functools
import inspect
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

import mindspore as ms
from mindspore import mint, ops

from ..utils import get_logger
from ..utils.constants import DIFFUSERS_ATTN_BACKEND, DIFFUSERS_ATTN_CHECKS
from .layers_compat import scaled_dot_product_attention

if TYPE_CHECKING:
    from ._modeling_parallel import ParallelConfig

_CAN_USE_FLASH_ATTN = True
_CAN_USE_FLASH_ATTN_3 = False
_CAN_USE_SAGE_ATTN = False
_CAN_USE_FLEX_ATTN = False
_CAN_USE_NPU_ATTN = False
_CAN_USE_XLA_ATTN = False
_CAN_USE_XFORMERS_ATTN = False


if _CAN_USE_FLASH_ATTN_3:
    raise RuntimeError("Flash Attention 3 is not usable.")
else:
    flash_attn_3_func = None
    flash_attn_3_varlen_func = None
    flash_attn_3_func_hub = None


if _CAN_USE_SAGE_ATTN:
    raise RuntimeError("Sage Attention is not usable.")
else:
    sageattn = None
    sageattn_qk_int8_pv_fp16_cuda = None
    sageattn_qk_int8_pv_fp16_triton = None
    sageattn_qk_int8_pv_fp8_cuda = None
    sageattn_qk_int8_pv_fp8_cuda_sm90 = None
    sageattn_varlen = None


if _CAN_USE_FLEX_ATTN:
    # We cannot import the flex_attention function from the package directly because it is expected (from the
    # pytorch documentation) that the user may compile it. If we import directly, we will not have access to the
    # compiled function.
    raise RuntimeError("Flex Attention is not usable.")
else:
    flex_attention = None


if _CAN_USE_NPU_ATTN:
    raise RuntimeError("NPU Fusion Attention is not usable.")
else:
    npu_fusion_attention = None


if _CAN_USE_XLA_ATTN:
    raise RuntimeError("XLA Attention is not usable.")
else:
    xla_flash_attention = None


if _CAN_USE_XFORMERS_ATTN:
    raise RuntimeError("Xformers Attention is not usable.")
else:
    xops = None


def custom_op_no_op(name, fn=None, /, *, mutates_args, device_types=None, schema=None):
    def wrap(func):
        return func

    return wrap if fn is None else fn


def register_fake_no_op(op, fn=None, /, *, lib=None, _stacklevel=1):
    def wrap(func):
        return func

    return wrap if fn is None else fn


_custom_op = custom_op_no_op
_register_fake = register_fake_no_op


logger = get_logger(__name__)  # pylint: disable=invalid-name

# TODO(aryan): Add support for the following:
# - Sage Attention++
# - block sparse, radial and other attention methods
# - CP with sage attention, flex, xformers, other missing backends
# - Add support for normal and CP training with backends that don't support it yet

_SAGE_ATTENTION_PV_ACCUM_DTYPE = Literal["fp32", "fp32+fp32"]
_SAGE_ATTENTION_QK_QUANT_GRAN = Literal["per_thread", "per_warp"]
_SAGE_ATTENTION_QUANTIZATION_BACKEND = Literal["cuda", "triton"]


class AttentionBackendName(str, Enum):
    # EAGER = "eager"

    # `flash-attn`
    FLASH = "flash"
    FLASH_VARLEN = "flash_varlen"
    _FLASH_3 = "_flash_3"
    _FLASH_VARLEN_3 = "_flash_varlen_3"
    _FLASH_3_HUB = "_flash_3_hub"
    _FLASH_VARLEN_3_HUB = "_flash_varlen_3_hub"  # not supported yet.

    # PyTorch native
    FLEX = "flex"
    NATIVE = "native"
    _NATIVE_CUDNN = "_native_cudnn"
    _NATIVE_EFFICIENT = "_native_efficient"
    _NATIVE_FLASH = "_native_flash"
    _NATIVE_MATH = "_native_math"
    _NATIVE_NPU = "_native_npu"
    _NATIVE_XLA = "_native_xla"

    # `sageattention`
    SAGE = "sage"
    SAGE_VARLEN = "sage_varlen"
    _SAGE_QK_INT8_PV_FP8_CUDA = "_sage_qk_int8_pv_fp8_cuda"
    _SAGE_QK_INT8_PV_FP8_CUDA_SM90 = "_sage_qk_int8_pv_fp8_cuda_sm90"
    _SAGE_QK_INT8_PV_FP16_CUDA = "_sage_qk_int8_pv_fp16_cuda"
    _SAGE_QK_INT8_PV_FP16_TRITON = "_sage_qk_int8_pv_fp16_triton"
    # TODO: let's not add support for Sparge Attention now because it requires tuning per model
    # We can look into supporting something "autotune"-ing in the future
    # SPARGE = "sparge"

    # `xformers`
    XFORMERS = "xformers"


class _AttentionBackendRegistry:
    _backends = {}
    _constraints = {}
    _supported_arg_names = {}
    _supports_context_parallel = {}
    _active_backend = AttentionBackendName(DIFFUSERS_ATTN_BACKEND)
    _checks_enabled = DIFFUSERS_ATTN_CHECKS

    @staticmethod
    def register(
        backend: AttentionBackendName,
        constraints: Optional[List[Callable]] = None,
        supports_context_parallel: bool = False,
    ):
        Registry = _AttentionBackendRegistry
        logger.debug(f"Registering attention backend: {backend} with constraints: {constraints}")

        def decorator(func):
            Registry._backends[backend] = func
            Registry._constraints[backend] = constraints or []
            Registry._supported_arg_names[backend] = set(inspect.signature(func).parameters.keys())
            Registry._supports_context_parallel[backend] = supports_context_parallel
            return func

        return decorator

    @staticmethod
    def get_active_backend():
        Registry = _AttentionBackendRegistry
        return Registry._active_backend, Registry._backends[Registry._active_backend]

    @staticmethod
    def list_backends():
        Registry = _AttentionBackendRegistry
        return list(Registry._backends.keys())

    @staticmethod
    def _is_context_parallel_enabled(
        backend: AttentionBackendName, parallel_config: Optional["ParallelConfig"]
    ) -> bool:
        Registry = _AttentionBackendRegistry
        supports_context_parallel = backend in Registry._supports_context_parallel
        is_degree_greater_than_1 = parallel_config is not None and (
            parallel_config.context_parallel_config.ring_degree > 1
            or parallel_config.context_parallel_config.ulysses_degree > 1
        )
        return supports_context_parallel and is_degree_greater_than_1


@contextlib.contextmanager
def attention_backend(backend: Union[str, AttentionBackendName] = AttentionBackendName.NATIVE):
    """
    Context manager to set the active attention backend.
    """
    if backend not in _AttentionBackendRegistry._backends:
        raise ValueError(f"Backend {backend} is not registered.")

    backend = AttentionBackendName(backend)
    _check_attention_backend_requirements(backend)

    old_backend = _AttentionBackendRegistry._active_backend
    _AttentionBackendRegistry._active_backend = backend

    try:
        yield
    finally:
        _AttentionBackendRegistry._active_backend = old_backend


def dispatch_attention_fn(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    *,
    backend: Optional[AttentionBackendName] = None,
    parallel_config: Optional["ParallelConfig"] = None,
) -> ms.Tensor:
    attention_kwargs = attention_kwargs or {}

    if backend is None:
        # If no backend is specified, we either use the default backend (set via the DIFFUSERS_ATTN_BACKEND environment
        # variable), or we use a custom backend based on whether user is using the `attention_backend` context manager
        backend_name, backend_fn = _AttentionBackendRegistry.get_active_backend()
    else:
        backend_name = AttentionBackendName(backend)
        backend_fn = _AttentionBackendRegistry._backends.get(backend_name)

    if parallel_config is not None and not _AttentionBackendRegistry._is_context_parallel_enabled(
        backend_name, parallel_config
    ):
        raise ValueError(
            f"Backend {backend_name} either does not support context parallelism or context parallelism "
            f"was enabled with a world size of 1."
        )

    kwargs = {
        "query": query,
        "key": key,
        "value": value,
        "attn_mask": attn_mask,
        "dropout_p": dropout_p,
        "is_causal": is_causal,
        "scale": scale,
        **attention_kwargs,
        "_parallel_config": parallel_config,
    }
    kwargs["enable_gqa"] = enable_gqa

    if _AttentionBackendRegistry._checks_enabled:
        removed_kwargs = set(kwargs) - set(_AttentionBackendRegistry._supported_arg_names[backend_name])
        if removed_kwargs:
            logger.warning(f"Removing unsupported arguments for attention backend {backend_name}: {removed_kwargs}.")
        for check in _AttentionBackendRegistry._constraints.get(backend_name):
            check(**kwargs)

    kwargs = {k: v for k, v in kwargs.items() if k in _AttentionBackendRegistry._supported_arg_names[backend_name]}
    return backend_fn(**kwargs)


# ===== Checks =====
# A list of very simple functions to catch common errors quickly when debugging.


def _check_attn_mask_or_causal(attn_mask: Optional[ms.Tensor], is_causal: bool, **kwargs) -> None:
    if attn_mask is not None and is_causal:
        raise ValueError("`is_causal` cannot be True when `attn_mask` is not None.")


def _check_device(query: ms.Tensor, key: ms.Tensor, value: ms.Tensor, **kwargs) -> None:
    if query.device != key.device or query.device != value.device:
        raise ValueError("Query, key, and value must be on the same device.")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("Query, key, and value must have the same dtype.")


def _check_qkv_dtype_match(query: ms.Tensor, key: ms.Tensor, value: ms.Tensor, **kwargs) -> None:
    if query.dtype != key.dtype:
        raise ValueError("Query and key must have the same dtype.")
    if query.dtype != value.dtype:
        raise ValueError("Query and value must have the same dtype.")


def _check_qkv_dtype_bf16_or_fp16(query: ms.Tensor, key: ms.Tensor, value: ms.Tensor, **kwargs) -> None:
    _check_qkv_dtype_match(query, key, value)
    if query.dtype not in (ms.bfloat16, ms.float16):
        raise ValueError("Query, key, and value must be either bfloat16 or float16.")


def _check_shape(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    **kwargs,
) -> None:
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("Query and key must have the same last dimension.")
    if query.shape[-2] != value.shape[-2]:
        raise ValueError("Query and value must have the same second to last dimension.")
    if attn_mask is not None and attn_mask.shape[-1] != key.shape[-2]:
        raise ValueError("Attention mask must match the key's second to last dimension.")


# ===== Helper functions =====


def _check_attention_backend_requirements(backend: AttentionBackendName) -> None:
    if backend in [AttentionBackendName.FLASH, AttentionBackendName.FLASH_VARLEN]:
        if not _CAN_USE_FLASH_ATTN:
            raise RuntimeError(f"Flash Attention backend '{backend.value}' is not usable.")

    elif backend in [AttentionBackendName._FLASH_3, AttentionBackendName._FLASH_VARLEN_3]:
        if not _CAN_USE_FLASH_ATTN_3:
            raise RuntimeError(f"Flash Attention 3 backend '{backend.value}' is not usable.")

    # TODO: add support Hub variant of FA3 varlen later
    elif backend in [AttentionBackendName._FLASH_3_HUB]:
        raise RuntimeError(f"Flash Attention 3 Hub backend '{backend.value}' is not usable.")

    elif backend in [
        AttentionBackendName.SAGE,
        AttentionBackendName.SAGE_VARLEN,
        AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA,
        AttentionBackendName._SAGE_QK_INT8_PV_FP8_CUDA_SM90,
        AttentionBackendName._SAGE_QK_INT8_PV_FP16_CUDA,
        AttentionBackendName._SAGE_QK_INT8_PV_FP16_TRITON,
    ]:
        if not _CAN_USE_SAGE_ATTN:
            raise RuntimeError(f"Sage Attention backend '{backend.value}' is not usable.")

    elif backend == AttentionBackendName.FLEX:
        if not _CAN_USE_FLEX_ATTN:
            raise RuntimeError(f"Flex Attention backend '{backend.value}' is not usable.")

    elif backend == AttentionBackendName._NATIVE_NPU:
        if not _CAN_USE_NPU_ATTN:
            raise RuntimeError(f"NPU Attention backend '{backend.value}' is not usable.")

    elif backend == AttentionBackendName._NATIVE_XLA:
        if not _CAN_USE_XLA_ATTN:
            raise RuntimeError(f"XLA Attention backend '{backend.value}' is not usable.")

    elif backend == AttentionBackendName.XFORMERS:
        if not _CAN_USE_XFORMERS_ATTN:
            raise RuntimeError(f"Xformers Attention backend '{backend.value}' is not usable.")


@functools.lru_cache(maxsize=128)
def _prepare_for_flash_attn_or_sage_varlen_without_mask(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
):
    seqlens_q = mint.full((batch_size,), seq_len_q, dtype=ms.int32)
    seqlens_k = mint.full((batch_size,), seq_len_kv, dtype=ms.int32)
    cu_seqlens_q = mint.zeros(batch_size + 1, dtype=ms.int32)
    cu_seqlens_k = mint.zeros(batch_size + 1, dtype=ms.int32)
    cu_seqlens_q[1:] = mint.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = mint.cumsum(seqlens_k, dim=0)
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def _prepare_for_flash_attn_or_sage_varlen_with_mask(
    batch_size: int,
    seq_len_q: int,
    attn_mask: ms.Tensor,
):
    seqlens_q = mint.full((batch_size,), seq_len_q, dtype=ms.int32)
    seqlens_k = attn_mask.sum(dim=1, dtype=ms.int32)
    cu_seqlens_q = mint.zeros(batch_size + 1, dtype=ms.int32)
    cu_seqlens_k = mint.zeros(batch_size + 1, dtype=ms.int32)
    cu_seqlens_q[1:] = mint.cumsum(seqlens_q, dim=0)
    cu_seqlens_k[1:] = mint.cumsum(seqlens_k, dim=0)
    max_seqlen_q = seqlens_q.max().item()
    max_seqlen_k = seqlens_k.max().item()
    return (seqlens_q, seqlens_k), (cu_seqlens_q, cu_seqlens_k), (max_seqlen_q, max_seqlen_k)


def _prepare_for_flash_attn_or_sage_varlen(
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    attn_mask: Optional[ms.Tensor] = None,
) -> None:
    if attn_mask is None:
        return _prepare_for_flash_attn_or_sage_varlen_without_mask(batch_size, seq_len_q, seq_len_kv)
    return _prepare_for_flash_attn_or_sage_varlen_with_mask(batch_size, seq_len_q, attn_mask)


def _normalize_attn_mask(attn_mask: ms.Tensor, batch_size: int, seq_len_k: int) -> ms.Tensor:
    """
    Normalize an attention mask to shape [batch_size, seq_len_k] (bool) suitable for inferring seqlens_[q|k] in
    FlashAttention/Sage varlen.

    Supports 1D to 4D shapes and common broadcasting patterns.
    """
    if attn_mask.dtype != ms.bool:
        raise ValueError(f"Attention mask must be of type bool, got {attn_mask.dtype}.")

    if attn_mask.ndim == 1:
        # [seq_len_k] -> broadcast across batch
        attn_mask = attn_mask.unsqueeze(0).expand((batch_size, seq_len_k))

    elif attn_mask.ndim == 2:
        # [batch_size, seq_len_k]. Maybe broadcast across batch
        if attn_mask.shape[0] not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 2D attention mask."
            )
        attn_mask = attn_mask.expand((batch_size, seq_len_k))

    elif attn_mask.ndim == 3:
        # [batch_size, seq_len_q, seq_len_k] -> reduce over query dimension
        # We do this reduction because we know that arbitrary QK masks is not supported in Flash/Sage varlen.
        if attn_mask.shape[0] not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 3D attention mask."
            )
        attn_mask = attn_mask.any(dim=1)
        attn_mask = attn_mask.expand((batch_size, seq_len_k))

    elif attn_mask.ndim == 4:
        # [batch_size, num_heads, seq_len_q, seq_len_k] or broadcastable versions
        if attn_mask.shape[0] not in [1, batch_size]:
            raise ValueError(
                f"attn_mask.shape[0] ({attn_mask.shape[0]}) must be 1 or {batch_size} for 4D attention mask."
            )
        attn_mask = attn_mask.expand((batch_size, -1, -1, seq_len_k))  # [B, H, Q, K]
        attn_mask = attn_mask.any(dim=(1, 2))  # [B, K]

    else:
        raise ValueError(f"Unsupported attention mask shape: {attn_mask.shape}")

    if attn_mask.shape != (batch_size, seq_len_k):
        raise ValueError(
            f"Normalized attention mask shape mismatch: got {attn_mask.shape}, expected ({batch_size}, {seq_len_k})"
        )

    return attn_mask


def _flex_attention_causal_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
    return q_idx >= kv_idx


# ===== Helper functions to use attention backends with templated CP autograd functions =====


def _native_attention_forward_op(
    ctx,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    # Native attention does not return_lse
    if return_lse:
        raise ValueError("Native attention does not support return_lse=True")

    # used for backward pass
    if _save_ctx:
        ctx.save_for_backward(query, key, value)
        ctx.attn_mask = attn_mask
        ctx.dropout_p = dropout_p
        ctx.is_causal = is_causal
        ctx.scale = scale
        ctx.enable_gqa = enable_gqa

    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)

    return out


def _native_attention_backward_op(
    ctx,
    grad_out: ms.Tensor,
    *args,
    **kwargs,
):
    query, key, value = ctx.saved_tensors

    query._requires_grad = True
    key._requires_grad = True
    value._requires_grad = True

    query_t, key_t, value_t = (x.permute(0, 2, 1, 3) for x in (query, key, value))

    def forward_fn(q, k, v):
        out = scaled_dot_product_attention(
            query=query_t,
            key=key_t,
            value=value_t,
            attn_mask=ctx.attn_mask,
            dropout_p=ctx.dropout_p,
            is_causal=ctx.is_causal,
            scale=ctx.scale,
            enable_gqa=ctx.enable_gqa,
        )
        out = out.permute(0, 2, 1, 3)
        return out

    grad_out_t = grad_out.permute(0, 2, 1, 3)  # noqa
    grad_query_t, grad_key_t, grad_value_t = ms.grad(forward_fn, grad_position=(0, 1, 2))(query_t, key_t, value_t)

    grad_query = grad_query_t.permute(0, 2, 1, 3)
    grad_key = grad_key_t.permute(0, 2, 1, 3)
    grad_value = grad_value_t.permute(0, 2, 1, 3)

    return grad_query, grad_key, grad_value


# Adapted from: https://github.com/Dao-AILab/flash-attention/blob/fd2fc9d85c8e54e5c20436465bca709bc1a6c5a1/flash_attn/flash_attn_interface.py#L807
def _flash_attention_forward_op(
    ctx,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _save_ctx: bool = True,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    # if attn_mask is not None:
    #     raise ValueError("`attn_mask` is not yet supported for flash-attn.")
    if enable_gqa:
        raise ValueError("`enable_gqa` is not yet supported for flash-attn.")

    # Hardcoded for now
    window_size = (-1, -1)
    softcap = 0.0
    alibi_slopes = None
    deterministic = False
    grad_enabled = any(x._requires_grad for x in (query, key, value))

    if scale is None:
        scale = query.shape[-1] ** (-0.5)

    if is_causal:
        sparse_mode = 2
    else:
        sparse_mode = 0

    # flash-attn only returns LSE if dropout_p > 0. So, we need to workaround.
    if grad_enabled or (_parallel_config is not None and _parallel_config.context_parallel_config._world_size > 1):
        dropout_p = dropout_p if dropout_p > 0 else 1e-30

    input_layout = "BSND"
    head_num = query.shape[2]

    softmax_max, softmax_sum, _, out = ops.operations.nn_ops.FlashAttentionScore(
        head_num=head_num,
        keep_prob=1 - dropout_p,
        scale_value=scale,
        input_layout=input_layout,
        sparse_mode=sparse_mode,
    )(query, key, value, None, None, None, attn_mask)
    lse = softmax_max[..., 0] + mint.log(softmax_sum[..., 0])
    lse = lse.permute(0, 2, 1)

    if _save_ctx:
        ctx.save_for_backward(query, key, value, out, lse)
        ctx.dropout_p = dropout_p
        ctx.scale = scale
        ctx.is_causal = is_causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic

    return (out, lse) if return_lse else out


def _flash_attention_backward_op(
    ctx,
    grad_out: ms.Tensor,
    *args,
    **kwargs,
):
    query, key, value, out, lse = ctx.saved_tensors
    grad_query, grad_key, grad_value = mint.empty_like(query), mint.empty_like(key), mint.empty_like(value)

    # Head dimension may have been padded
    grad_query = grad_query[..., : grad_out.shape[-1]]
    grad_key = grad_key[..., : grad_out.shape[-1]]
    grad_value = grad_value[..., : grad_out.shape[-1]]

    return grad_query, grad_key, grad_value


# ===== Context parallel =====


def _all_to_all_single(x: ms.Tensor, group) -> ms.Tensor:
    shape = x.shape
    # HACK: We need to flatten because despite making tensors contiguous, torch single-file-ization
    # to benchmark triton codegen fails somewhere:
    # buf25 = torch.ops._c10d_functional.all_to_all_single.default(buf24, [1, 1], [1, 1], '3')
    # ValueError: Tensors must be contiguous
    x = x.flatten()
    # `all_to_all_single` writes the result into output in-place.
    x_output = mint.zeros_like(x)
    mint.distributed.all_to_all_single(x_output, x, group=group)
    x_output = x_output.reshape(shape)
    return x_output


def permute_tensor(
    tensor: ms.Tensor,
    src_dst: list[int],
    group: None,
) -> ms.Tensor:
    """
    Permutes the elements of the tensor according to the given source/destination pairs. `src_dst` should
    be defined such that src_dst[m] == n means m sends to n.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one
    """
    rank = mint.distributed.get_rank(group)
    world_size = mint.distributed.get_world_size(group)

    output_split_sizes = [0] * world_size
    input_split_sizes = [0] * world_size

    dst = src_dst[rank]
    input_split_sizes[dst] = tensor.size

    for m, n in enumerate(src_dst):
        if n == rank:
            output_split_sizes[m] = tensor.size

    output = mint.zeros_like(tensor)

    mint.distributed.all_to_all_single(
        output, tensor, input_split_sizes=input_split_sizes, output_split_sizes=output_split_sizes, group=group
    )

    return output


class TemplatedRingAttention(ms.common._Function):
    @staticmethod
    def forward(
        ctx,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        ring_mesh = _parallel_config.context_parallel_config._ring_mesh
        rank = _parallel_config.context_parallel_config._ring_local_rank
        world_size = _parallel_config.context_parallel_config.ring_degree
        next_rank = (rank + 1) % world_size
        prev_out = prev_lse = None

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx.q_shape = query.shape
        ctx.kv_shape = key.shape
        ctx._parallel_config = _parallel_config

        kv_buffer = mint.cat([key.flatten(), value.flatten()]).contiguous()
        group_size = mint.distributed.get_world_size(ring_mesh)
        kv_buffer_output = mint.cat([mint.zeros_like(kv_buffer) for _ in range(group_size)], dim=0)
        # `all_gather_into_tensor` performs in-place all-gather into kv_buffer_output.
        _ = mint.distributed.all_gather_into_tensor(kv_buffer_output, kv_buffer, group=ring_mesh)
        kv_buffer = kv_buffer_output.chunk(world_size)

        for i in range(world_size):
            if i > 0:
                kv = kv_buffer[next_rank]
                key_numel = key.numel()
                key = kv[:key_numel].reshape_as(key)
                value = kv[key_numel:].reshape_as(value)
                next_rank = (next_rank + 1) % world_size

            out, lse = forward_op(
                ctx,
                query,
                key,
                value,
                attn_mask,
                dropout_p,
                is_causal,
                scale,
                enable_gqa,
                True,
                _save_ctx=i == 0,
                _parallel_config=_parallel_config,
            )

            if _parallel_config.context_parallel_config.convert_to_fp32:
                out = out.to(ms.float32)
                lse = lse.to(ms.float32)

            lse = lse.unsqueeze(-1)
            if prev_out is not None:
                out = prev_out - mint.nn.functional.sigmoid(lse - prev_lse) * (prev_out - out)
                lse = prev_lse - mint.nn.functional.logsigmoid(prev_lse - lse)
            prev_out = out
            prev_lse = lse

        out = out.to(query.dtype)
        lse = lse.squeeze(-1)

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx,
        grad_out: ms.Tensor,
        *args,
    ):
        ring_mesh = ctx._parallel_config.context_parallel_config._ring_mesh
        rank = ctx._parallel_config.context_parallel_config._ring_local_rank
        world_size = ctx._parallel_config.context_parallel_config.ring_degree
        next_rank = (rank + 1) % world_size
        next_ranks = list(range(1, world_size)) + [0]

        accum_dtype = ms.float32 if ctx._parallel_config.context_parallel_config.convert_to_fp32 else grad_out.dtype
        grad_query = mint.zeros(ctx.q_shape, dtype=accum_dtype)
        grad_key = mint.zeros(ctx.kv_shape, dtype=accum_dtype)
        grad_value = mint.zeros(ctx.kv_shape, dtype=accum_dtype)
        next_grad_kv = None

        query, key, value, *_ = ctx.saved_tensors
        kv_buffer = mint.cat([key.flatten(), value.flatten()]).contiguous()
        group_size = mint.distributed.get_world_size(ring_mesh)
        kv_buffer_output = mint.cat([mint.zeros_like(kv_buffer) for _ in range(group_size)], dim=0)
        _ = mint.distributed.all_gather_into_tensor(kv_buffer_output, kv_buffer, group=ring_mesh)
        kv_buffer = kv_buffer_output.chunk(world_size)

        for i in range(world_size):
            if i > 0:
                kv = kv_buffer[next_rank]
                key_numel = key.numel()
                key = kv[:key_numel].reshape_as(key)
                value = kv[key_numel:].reshape_as(value)
                next_rank = (next_rank + 1) % world_size

            grad_query_op, grad_key_op, grad_value_op, *_ = ctx.backward_op(ctx, grad_out)

            if i > 0:
                grad_kv_buffer = next_grad_kv
                grad_key_numel = grad_key.numel()
                grad_key = grad_kv_buffer[:grad_key_numel].reshape_as(grad_key)
                grad_value = grad_kv_buffer[grad_key_numel:].reshape_as(grad_value)

            grad_query += grad_query_op
            grad_key += grad_key_op
            grad_value += grad_value_op

            if i < world_size - 1:
                grad_kv_buffer = mint.cat([grad_key.flatten(), grad_value.flatten()]).contiguous()
                next_grad_kv = permute_tensor(grad_kv_buffer, next_ranks, group=ring_mesh)

        grad_query, grad_key, grad_value = (x.to(grad_out.dtype) for x in (grad_query, grad_key, grad_value))

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


class TemplatedUlyssesAttention(ms.common._Function):
    @staticmethod
    def forward(
        ctx,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor],
        dropout_p: float,
        is_causal: bool,
        scale: Optional[float],
        enable_gqa: bool,
        return_lse: bool,
        forward_op,
        backward_op,
        _parallel_config: Optional["ParallelConfig"] = None,
    ):
        ulysses_mesh = _parallel_config.context_parallel_config._ulysses_mesh
        world_size = _parallel_config.context_parallel_config.ulysses_degree
        group = ulysses_mesh

        ctx.forward_op = forward_op
        ctx.backward_op = backward_op
        ctx._parallel_config = _parallel_config

        B, S_Q_LOCAL, H, D = query.shape
        _, S_KV_LOCAL, _, _ = key.shape
        H_LOCAL = H // world_size
        query = query.reshape(B, S_Q_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        key = key.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        value = value.reshape(B, S_KV_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        query, key, value = (_all_to_all_single(x, group) for x in (query, key, value))
        query, key, value = (x.flatten(0, 1).permute(1, 0, 2, 3).contiguous() for x in (query, key, value))

        out = forward_op(
            ctx,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=True,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse, *_ = out

        out = out.reshape(B, world_size, S_Q_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
        out = _all_to_all_single(out, group)
        out = out.flatten(0, 1).permute(1, 2, 0, 3).contiguous()

        if return_lse:
            lse = lse.reshape(B, world_size, S_Q_LOCAL, H_LOCAL).permute(1, 3, 0, 2).contiguous()
            lse = _all_to_all_single(lse, group)
            lse = lse.flatten(0, 1).permute(1, 2, 0).contiguous()
        else:
            lse = None

        return (out, lse) if return_lse else out

    @staticmethod
    def backward(
        ctx,
        grad_out: ms.Tensor,
        *args,
    ):
        ulysses_mesh = ctx._parallel_config.context_parallel_config._ulysses_mesh
        world_size = ctx._parallel_config.context_parallel_config.ulysses_degree
        group = ulysses_mesh

        B, S_LOCAL, H, D = grad_out.shape
        H_LOCAL = H // world_size

        grad_out = grad_out.reshape(B, S_LOCAL, world_size, H_LOCAL, D).permute(2, 1, 0, 3, 4).contiguous()
        grad_out = _all_to_all_single(grad_out, group)
        grad_out = grad_out.flatten(0, 1).permute(1, 0, 2, 3).contiguous()

        grad_query_op, grad_key_op, grad_value_op, *_ = ctx.backward_op(ctx, grad_out)

        grad_query, grad_key, grad_value = (
            x.reshape(B, world_size, S_LOCAL, H_LOCAL, D).permute(1, 3, 0, 2, 4).contiguous()
            for x in (grad_query_op, grad_key_op, grad_value_op)
        )
        grad_query, grad_key, grad_value = (_all_to_all_single(x, group) for x in (grad_query, grad_key, grad_value))
        grad_query, grad_key, grad_value = (
            x.flatten(0, 1).permute(1, 2, 0, 3).contiguous() for x in (grad_query, grad_key, grad_value)
        )

        return grad_query, grad_key, grad_value, None, None, None, None, None, None, None, None


def _templated_context_parallel_attention(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    *,
    forward_op,
    backward_op,
    _parallel_config: Optional["ParallelConfig"] = None,
):
    if attn_mask is not None:
        raise ValueError("Attention mask is not yet supported for templated attention.")
    if is_causal:
        raise ValueError("Causal attention is not yet supported for templated attention.")
    if enable_gqa:
        raise ValueError("GQA is not yet supported for templated attention.")

    # TODO: add support for unified attention with ring/ulysses degree both being > 1
    if _parallel_config.context_parallel_config.ring_degree > 1:
        return TemplatedRingAttention.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op,
            backward_op,
            _parallel_config,
        )
    elif _parallel_config.context_parallel_config.ulysses_degree > 1:
        return TemplatedUlyssesAttention.apply(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op,
            backward_op,
            _parallel_config,
        )
    else:
        raise ValueError("Reaching this branch of code is unexpected. Please report a bug.")


# ===== Attention backends =====


@_AttentionBackendRegistry.register(
    AttentionBackendName.FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
    supports_context_parallel=True,
)
def _flash_attention(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> ms.Tensor:
    lse = None
    if _parallel_config is None:
        if scale is None:
            scale = query.shape[-1] ** (-0.5)

        if is_causal:
            sparse_mode = 2
        else:
            sparse_mode = 0

        input_layout = "BSND"
        head_num = query.shape[2]

        softmax_max, softmax_sum, _, out = ops.operations.nn_ops.FlashAttentionScore(
            head_num=head_num,
            keep_prob=1 - dropout_p,
            scale_value=scale,
            input_layout=input_layout,
            sparse_mode=sparse_mode,
        )(query, key, value)
        lse = softmax_max[..., 0] + mint.log(softmax_sum[..., 0])
        lse = lse.permute(0, 2, 1)
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            None,
            dropout_p,
            is_causal,
            scale,
            False,
            return_lse,
            forward_op=_flash_attention_forward_op,
            backward_op=_flash_attention_backward_op,
            _parallel_config=_parallel_config,
        )
        if return_lse:
            out, lse = out

    return (out, lse) if return_lse else out


@_AttentionBackendRegistry.register(
    AttentionBackendName.NATIVE,
    constraints=[_check_device, _check_shape],
    supports_context_parallel=True,
)
def _native_attention(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> ms.Tensor:
    if return_lse:
        raise ValueError("Native attention backend does not support setting `return_lse=True`.")
    if _parallel_config is None:
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        out = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
            enable_gqa=enable_gqa,
        )
        out = out.permute(0, 2, 1, 3)
    else:
        out = _templated_context_parallel_attention(
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            forward_op=_native_attention_forward_op,
            backward_op=_native_attention_backward_op,
            _parallel_config=_parallel_config,
        )

    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_FLASH,
    constraints=[_check_device, _check_qkv_dtype_bf16_or_fp16, _check_shape],
)
def _native_flash_attention(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> ms.Tensor:
    if return_lse:
        raise ValueError("Native flash attention backend does not support setting `return_lse=True`.")
    if enable_gqa:
        raise ValueError("Native flash attention backend does not support setting `enable_gqa=True`.")
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=None,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    out = out.permute(0, 2, 1, 3)
    return out


@_AttentionBackendRegistry.register(
    AttentionBackendName._NATIVE_MATH,
    constraints=[_check_device, _check_shape],
)
def _native_math_attention(
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attn_mask: Optional[ms.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    _parallel_config: Optional["ParallelConfig"] = None,
) -> ms.Tensor:
    if return_lse:
        raise ValueError("Native math attention backend does not support setting `return_lse=True`.")
    query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
    out = scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
        backend="math",
    )
    out = out.permute(0, 2, 1, 3)
    return out
