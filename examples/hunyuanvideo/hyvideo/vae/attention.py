# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import math
from typing import Callable, Optional

import mindspore as ms
from mindspore import nn, ops

from mindone.diffusers.models.attention_processor import (
    AttentionProcessor,
    AttnProcessor,
    SpatialNorm,
    XFormersAttnProcessor,
)
from mindone.diffusers.utils import is_mindspore_version, logging
from mindone.diffusers.utils.mindspore_utils import dtype_to_min

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class Attention(nn.Cell):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        kv_heads (`int`,  *optional*, defaults to `None`):
            The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
            `kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
            Query Attention (MQA) otherwise GQA is used.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        kv_heads: Optional[int] = None,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,
        added_kv_proj_dim: Optional[int] = None,
        added_proj_bias: Optional[bool] = True,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
        pre_only=False,
    ):
        super().__init__()

        # To prevent circular import.
        from .normalization import FP32LayerNorm, GroupNorm, LayerNorm, RMSNorm

        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.inner_kv_dim = self.inner_dim if kv_heads is None else dim_head * kv_heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0
        # use `scale_sqrt` for ops.baddbmm to get same outputs with torch.baddbmm in fp16
        self.scale_sqrt = float(math.sqrt(self.scale))

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."  # noqa: E501
            )

        if norm_num_groups is not None:
            self.group_norm = GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = LayerNorm(dim_head, eps=eps)
            self.norm_k = LayerNorm(dim_head, eps=eps)
        elif qk_norm == "fp32_layer_norm":
            self.norm_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            self.norm_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
        elif qk_norm == "layer_norm_across_heads":
            # Lumina applys qk norm across all heads
            self.norm_q = LayerNorm(dim_head * heads, eps=eps)
            self.norm_k = LayerNorm(dim_head * kv_heads, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm'")

        self.cross_attention_norm = cross_attention_norm
        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Dense(query_dim, self.inner_dim, has_bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Dense(self.cross_attention_dim, self.inner_kv_dim, has_bias=bias)
            self.to_v = nn.Dense(self.cross_attention_dim, self.inner_kv_dim, has_bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Dense(added_kv_proj_dim, self.inner_kv_dim, has_bias=added_proj_bias)
            self.add_v_proj = nn.Dense(added_kv_proj_dim, self.inner_kv_dim, has_bias=added_proj_bias)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Dense(added_kv_proj_dim, self.inner_dim, has_bias=added_proj_bias)

        if not self.pre_only:
            self.to_out = nn.CellList(
                [nn.Dense(self.inner_dim, self.out_dim, has_bias=out_bias), nn.Dropout(p=dropout)]
            )

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = nn.Dense(self.inner_dim, self.out_dim, has_bias=out_bias)

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # MindSpore flash attention settings
        # flash attention only supports fp16 and bf 16, force cast to fp16 defaultly
        self.fa_op_available = (
            is_mindspore_version(">=", "2.3.0") and ms.get_context("device_target").lower() == "ascend"
        )
        self._enable_flash_sdp = True
        self.set_flash_attention_force_cast_dtype(ms.float16)

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = AttnProcessor2_0() if self.scale_qk else AttnProcessor()
        self.processor = processor

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                Not supported for now.
        """
        if use_memory_efficient_attention_xformers:
            if not hasattr(ops.operations.nn_ops, "FlashAttentionScore"):
                raise ModuleNotFoundError(
                    f"Memory efficient attention on mindspore uses flash attention under the hoods. "
                    f"The implementation of flash attention is `FlashAttentionScore`, "
                    f"which should be available in `mindspore.ops.operations.nn_ops`. "
                    f"However, we cannot find it in current environment(mindspore version: {ms.__version__})."
                )
            elif ms.get_context("device_target") != "Ascend":
                raise ValueError(
                    f"Memory efficient attention is only available for Ascend, "
                    f"but got current device: {ms.get_context('device_target')}"
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    flash_attn = ops.operations.nn_ops.FlashAttentionScore(1, input_layout="BSH")
                    _ = flash_attn(
                        ops.randn(1, 16, 64, dtype=ms.float16),
                        ops.randn(1, 16, 64, dtype=ms.float16),
                        ops.randn(1, 16, 64, dtype=ms.float16),
                    )
                except Exception as e:
                    raise e

            # The following lines is a patch for flash attn, which calculates implicit padding on head_dim.
            # TODO: Remove it if flash attention has better supports.
            import bisect

            self.flash_attn_valid_head_dims = [64, 80, 96, 120, 128, 256]
            self.head_dim = self.inner_dim // self.heads
            if self.head_dim in self.flash_attn_valid_head_dims:
                self.head_dim_padding = 0
            else:
                minimum_larger_index = bisect.bisect_right(self.flash_attn_valid_head_dims, self.head_dim)
                if minimum_larger_index >= len(self.flash_attn_valid_head_dims):
                    self.head_dim_padding = -1  # head_dim is bigger than the largest one, we cannot do padding
                else:
                    self.head_dim_padding = self.flash_attn_valid_head_dims[minimum_larger_index] - self.head_dim

            if self.head_dim_padding == 0:
                logger.info(
                    f"The head dimension of '{self.to_q.weight.name[:-12]}' is {self.head_dim}. "
                    f"Successfully set to use the flash attention."
                )
                processor = XFormersAttnProcessor(attention_op=attention_op)
            elif self.head_dim_padding > 0:
                logger.warning(
                    f"Flash attention requires that the head dimension must be one of "
                    f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
                    f"We will implicitly pad the head dimension to {self.head_dim + self.head_dim_padding}."
                )
                processor = XFormersAttnProcessor(attention_op=attention_op)
            else:
                logger.warning(
                    f"Flash attention requires that the head dimension must be one of "
                    f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
                    f"Fallback to the vanilla implementation of attention."
                )
                processor = AttnProcessor()

            # # The following lines is a patch for flash attn, which fallbacks to vanilla attn if head_dim is invalid.
            # # TODO: Remove it if flash attention has better supports.
            # self.flash_attn_valid_head_dims = [64, 80, 96, 120, 128, 256]
            # self.head_dim = self.inner_dim // self.heads
            # self.head_dim_padding = 0
            # if self.head_dim in self.flash_attn_valid_head_dims:
            #     logger.info(
            #         f"The head dimension of '{self.to_q.weight.name[:-12]}' is {self.head_dim}. "
            #         f"Successfully set to use the flash attention."
            #     )
            #     processor = XFormersAttnProcessor(attention_op=attention_op)
            # else:
            #     logger.warning(
            #         f"Flash attention requires that the head dimension must be one of "
            #         f"{self.flash_attn_valid_head_dims}, but got {self.head_dim} in '{self.to_q.weight.name[:-12]}'. "
            #         f"Fallback to the vanilla implementation of attention."
            #     )
            #     processor = AttnProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = AttnProcessor()

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if hasattr(self, "processor") and isinstance(self.processor, nn.Cell) and not isinstance(processor, nn.Cell):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._cells.pop("processor")

        self.processor = processor

    def get_processor(self) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        return self.processor

    def construct(
        self,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        **cross_attention_kwargs,
    ) -> ms.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`ms.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`ms.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`ms.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `ms.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty
        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: ms.Tensor) -> ms.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`ms.Tensor`): The tensor to reshape.

        Returns:
            `ms.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: ms.Tensor, out_dim: int = 3) -> ms.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`ms.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `ms.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: ms.Tensor, key: ms.Tensor, attention_mask: Optional[ms.Tensor] = None
    ) -> ms.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`ms.Tensor`): The query tensor.
            key (`ms.Tensor`): The key tensor.
            attention_mask (`ms.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `ms.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            attention_scores = ops.bmm(
                query * self.scale_sqrt,
                key.swapaxes(-1, -2) * self.scale_sqrt,
            )
        else:
            attention_scores = ops.baddbmm(
                attention_mask.to(query.dtype),
                query * self.scale_sqrt,
                key.swapaxes(-1, -2) * self.scale_sqrt,
                beta=1,
                alpha=1,
            )

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = ops.softmax(attention_scores, axis=-1)

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: ms.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> Optional[ms.Tensor]:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`ms.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `ms.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length = attention_mask.shape[-1]
        if current_length != target_length:
            # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
            #       we want to instead pad by (0, remaining_length), where remaining_length is:
            #       remaining_length: int = target_length - current_length
            # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
            attention_mask = ops.Pad(paddings=((0, 0),) * (attention_mask.ndim - 1) + ((0, target_length),))(
                attention_mask
            )

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: ms.Tensor) -> ms.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`ms.Tensor`): Hidden states of the encoder.

        Returns:
            `ms.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if self.cross_attention_norm == "layer_norm":
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif self.cross_attention_norm == "group_norm":
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.swapaxes(1, 2)
        else:
            assert False

        return encoder_hidden_states

    def fuse_projections(self, fuse=True):
        dtype = self.to_q.weight.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = ops.cat([self.to_q.weight, self.to_k.weight, self.to_v.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = nn.Dense(in_features, out_features, has_bias=self.use_bias, dtype=dtype)
            self.to_qkv.weight.set_data(concatenated_weights)
            if self.use_bias:
                concatenated_bias = ops.cat([self.to_q.bias, self.to_k.bias, self.to_v.bias])
                self.to_qkv.bias.set_data(concatenated_bias)

        else:
            concatenated_weights = ops.cat([self.to_k.weight, self.to_v.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Dense(in_features, out_features, has_bias=self.use_bias, dtype=dtype)
            self.to_kv.weight.set_data(concatenated_weights)
            if self.use_bias:
                concatenated_bias = ops.cat([self.to_k.bias, self.to_v.bias])
                self.to_kv.bias.set_data(concatenated_bias)

        # handle added projections for SD3 and others.
        if hasattr(self, "add_q_proj") and hasattr(self, "add_k_proj") and hasattr(self, "add_v_proj"):
            concatenated_weights = ops.cat([self.add_q_proj.weight, self.add_k_proj.weight, self.add_v_proj.weight])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_added_qkv = nn.Dense(in_features, out_features, has_bias=self.added_proj_bias, dtype=dtype)
            self.to_added_qkv.weight.set_data(concatenated_weights)
            if self.added_proj_bias:
                concatenated_bias = ops.cat([self.add_q_proj.bias, self.add_k_proj.bias, self.add_v_proj.bias])
                self.to_added_qkv.bias.set_data(concatenated_bias)

        self.fused_projections = fuse

    def enable_flash_sdp(self, enabled: bool):
        r"""
        .. warning:: This flag is beta and subject to change.

        Enables or disables flash scaled dot product attention.
        """
        self._enable_flash_sdp = enabled

    def set_flash_attention_force_cast_dtype(self, force_cast_dtype: Optional[ms.Type]):
        r"""
        Since the flash-attention operator in MindSpore only supports float16 and bfloat16 data types,
        we need to manually set whether to force data type conversion.

        When the attention interface encounters data of an unsupported data type, if `force_cast_dtype`
        is not None, the function will forcibly convert the data to `force_cast_dtype` for computation
        and then restore it to the original data type afterward. If `force_cast_dtype` is None, it will
        fall back to the original attention calculation using mathematical formulas.

        Parameters:
            force_cast_dtype (Optional): The data type to which the input data should be forcibly converted.
                                         If None, no forced conversion is performed.
        """
        self.fa_force_dtype = force_cast_dtype

    def scaled_dot_product_attention(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        r"""
        Perform scaled dot-product attention using either the flash attention operator or the mathematical
        formula-based attention, depending on the availability of the flash attention operator and the
        data-types of the inputs.

        Parameters:
            query (ms.Tensor): The query tensor.
            key (ms.Tensor): The key tensor.
            value (ms.Tensor): The value tensor.
            attn_mask (Optional[ms.Tensor], optional): The attention mask tensor. Defaults to None.
            dropout_p (float, optional): The dropout probability. Defaults to 0.0.
            is_causal (bool): If true, attn_mask must be None.
            scale (float, optional): scaled value

        Returns:
            ms.Tensor: The result of the scaled dot-product attention.

        Notes:
            - If the flash attention operator is not available (`self.fa_op_available` is False),
              the function falls back to the mathematical formula-based attention.
            - If the data types of `query`, `key`, and `value` are either `float16` or `bfloat16`, the
              flash-attention operator is used directly.
            - If `self.fa_force_dtype` is set to `float16` or `bfloat16`, the input tensors are cast to
              this data-type, the flash attention operator is applied, and the result is cast back to the
              original data type of `query`.
            - Otherwise, the function falls back to the mathematical formula-based attention.
        """
        if is_causal:
            assert attn_mask is None, "attn_mask must be None when is_causal is True"

        if not (self.fa_op_available and self._enable_flash_sdp):
            if is_causal:
                attn_mask = self.create_causal_mask(query, key, return_fa_mask=False)
            return self.math_attention_op(query, key, value, attn_mask)
        elif query.dtype in (ms.float16, ms.bfloat16):
            if is_causal:
                attn_mask = self.create_causal_mask(query, key, return_fa_mask=True)
            return self.flash_attention_op(query, key, value, attn_mask, keep_prob=1 - dropout_p, scale=scale)
        elif self.fa_force_dtype in (ms.float16, ms.bfloat16):
            if is_causal:
                attn_mask = self.create_causal_mask(query, key, return_fa_mask=True)
            return self.flash_attention_op(
                query.to(self.fa_force_dtype),
                key.to(self.fa_force_dtype),
                value.to(self.fa_force_dtype),
                attn_mask,
                keep_prob=1 - dropout_p,
                scale=scale,
            ).to(query.dtype)
        else:
            if is_causal:
                attn_mask = self.create_causal_mask(query, key, return_fa_mask=False)
            return self.math_attention_op(query, key, value, attn_mask)

    def math_attention_op(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
    ):
        # Adapted from mindone.diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.construct
        if attn_mask is not None and attn_mask.dtype == ms.bool_:
            attn_mask = ops.logical_not(attn_mask) * dtype_to_min(query.dtype)

        attention_probs = self.get_attention_scores(query, key, attn_mask)
        hidden_states = ops.bmm(attention_probs, value)

        return hidden_states

    def create_causal_mask(self, query: ms.Tensor, key: ms.Tensor, return_fa_mask=False):
        L, S = query.shape[-1], key.shape[-1]
        BS = query.shape[0]
        mask = ops.tril(ops.ones((L, S), dtype=ms.bool_), diagonal=0)
        if return_fa_mask:
            # flip mask, since FA treats 1 as discard, 0 as retain.
            mask = ~mask if mask.dtype == ms.bool_ else 1 - mask
            # reshape mask to be (bs, num_heads, L, S)
            mask = mask.unsqueeze(0).tile((BS, 1, 1))
            mask = mask.unsqueeze(1).tile((1, self.heads, 1, 1))
            return mask

        return mask.unsqueeze(0).tile((BS, 1, 1))

    def flash_attention_op(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        keep_prob: float = 1.0,
        scale: Optional[float] = None,
    ):
        # For most scenarios, qkv has been processed into a BNSD layout before sdp
        input_layout = "BNSD"
        head_num = self.heads

        # In case qkv is 3-dim after `head_to_batch_dim`
        if query.ndim == 3:
            input_layout = "BSH"
            head_num = 1

        # process `attn_mask` as logic is different between PyTorch and Mindspore
        # In MindSpore, False indicates retention and True indicates discard, in PyTorch it is the opposite
        if attn_mask is not None:
            attn_mask = ops.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
            attn_mask = ops.broadcast_to(
                attn_mask, (attn_mask.shape[0], attn_mask.shape[1], query.shape[-2], key.shape[-2])
            )[:, :1, :, :]

        return ops.operations.nn_ops.FlashAttentionScore(
            head_num=head_num, keep_prob=keep_prob, scale_value=scale or self.scale, input_layout=input_layout
        )(query, key, value, None, None, None, attn_mask)[3]


@ms.jit_class
class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        is_causal: bool = False,
        **kwargs,
    ) -> ms.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).swapaxes(1, 2)
        else:
            batch_size, channel, height, width = None, None, None, None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        if is_causal:
            assert attention_mask is None, "Cannot use attention mask when is_causal is True"
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.swapaxes(1, 2)).swapaxes(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).swapaxes(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        hidden_states = attn.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=is_causal
        )

        hidden_states = hidden_states.swapaxes(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.swapaxes(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
