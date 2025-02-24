import logging
import numbers
from typing import Optional, Tuple

from opensora.acceleration.communications import AllToAll_SBH
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.npu_config import npu_config

import mindspore as ms
from mindspore import Parameter, mint, nn, ops
from mindspore.common.initializer import initializer

from mindone.diffusers.models.attention import FeedForward
from mindone.diffusers.models.attention_processor import Attention as Attention_
from mindone.utils.version_control import check_valid_flash_attention

from ..common import PositionGetter3D, RoPE3D

logger = logging.getLogger(__name__)


class LayerNorm(nn.Cell):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine: bool = True, dtype=ms.float32):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = Parameter(initializer("ones", normalized_shape, dtype=dtype))
            self.beta = Parameter(initializer("zeros", normalized_shape, dtype=dtype))
        else:
            self.gamma = mint.ones(normalized_shape, dtype=dtype)
            self.beta = mint.zeros(normalized_shape, dtype=dtype)
        self.layer_norm = ops.LayerNorm(-1, -1, epsilon=eps)

    def construct(self, x: ms.Tensor):
        x, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return x


class Attention(Attention_):
    def __init__(self, interpolation_scale_thw, sparse1d, sparse_n, sparse_group, is_cross_attn, **kwags):
        processor = OpenSoraAttnProcessor2_0(
            interpolation_scale_thw=interpolation_scale_thw,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=is_cross_attn,
            dim_head=kwags["dim_head"],
        )
        super().__init__(processor=processor, **kwags)
        if npu_config.enable_FA:
            self.set_use_memory_efficient_attention_xformers(True)
        self.processor = processor

    @staticmethod
    def prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n, head_num):
        attention_mask = attention_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        length = attention_mask.shape[-1]
        if length % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - length % (sparse_n * sparse_n)

        attention_mask_sparse = mint.nn.functional.pad(
            attention_mask, (0, pad_len, 0, 0), mode="constant", value=0
        )  # 0 for discard
        b = attention_mask_sparse.shape[0]
        k = sparse_n
        m = sparse_n
        # b 1 1 (g k) -> (k b) 1 1 g
        attention_mask_sparse_1d = (
            attention_mask_sparse.reshape(b, 1, 1, -1, k).permute(4, 0, 1, 2, 3).reshape(b * k, 1, 1, -1)
        )
        # b 1 1 (n m k) -> (m b) 1 1 (n k)
        attention_mask_sparse_1d_group = (
            attention_mask_sparse.reshape(b, 1, 1, -1, m, k).permute(4, 0, 1, 2, 3, 5).reshape(m * b, 1, 1, -1)
        )
        encoder_attention_mask_sparse = encoder_attention_mask.tile((sparse_n, 1, 1, 1))

        # get attention mask dtype, and shape
        attention_mask_sparse_1d = npu_config.get_attention_mask(
            attention_mask_sparse_1d, attention_mask_sparse_1d.shape[-1]
        )
        attention_mask_sparse_1d_group = npu_config.get_attention_mask(
            attention_mask_sparse_1d_group, attention_mask_sparse_1d_group.shape[-1]
        )

        encoder_attention_mask_sparse_1d = npu_config.get_attention_mask(
            encoder_attention_mask_sparse, attention_mask_sparse_1d.shape[-1]
        )
        encoder_attention_mask_sparse_1d_group = encoder_attention_mask_sparse_1d

        return {
            False: (attention_mask_sparse_1d, encoder_attention_mask_sparse_1d),
            True: (attention_mask_sparse_1d_group, encoder_attention_mask_sparse_1d_group),
        }

    def prepare_attention_mask(
        self, attention_mask: ms.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> ms.Tensor:
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
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if get_sequence_parallel_state():
            head_size = head_size // hccl_info.world_size  # e.g, 24 // 8

        if attention_mask is None:  # b 1 t*h*w in sa, b 1 l in ca
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            attention_mask = mint.nn.functional.pad(attention_mask, (0, target_length), mode="constant", value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, 0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, 1)

        return attention_mask


@ms.jit_class
class OpenSoraAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(
        self,
        interpolation_scale_thw=(1, 1, 1),
        sparse1d=False,
        sparse_n=2,
        sparse_group=False,
        is_cross_attn=True,
        dim_head=96,
    ):
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw

        self._init_rope(interpolation_scale_thw, dim_head=dim_head)

        # if npu_config.enable_FA:
        #     FLASH_IS_AVAILABLE = check_valid_flash_attention()
        #     npu_config.enable_FA = FLASH_IS_AVAILABLE and npu_config.enable_FA
        # if npu_config.enable_FA:
        #     npu_config.FA_dtype = FA_dtype
        #     assert FA_dtype in [ms.float16, ms.bfloat16], f"Unsupported flash-attention dtype: {FA_dtype}"
        # self.fa_mask_dtype = choose_flash_attention_dtype()

        if get_sequence_parallel_state():
            self.sp_size = hccl_info.world_size
            self.alltoall_sbh_q = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_k = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_v = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_out = AllToAll_SBH(scatter_dim=0, gather_dim=1)
        else:
            self.sp_size = 1
            self.alltoall_sbh_q = None
            self.alltoall_sbh_k = None
            self.alltoall_sbh_v = None
            self.alltoall_sbh_out = None

    def _init_rope(self, interpolation_scale_thw, dim_head):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw, dim_head=dim_head)
        self.position_getter = PositionGetter3D()

    # TODO: need consider shapes for parallel seq and non-parallel cases
    def _sparse_1d(self, x, frame, height, width):
        """
        require the shape of (ntokens x batch_size x dim)

        Convert to sparse groups
        Input:
            x: shape in S,B,D
        Output:
            x: shape if sparse_group: (S//sparse_n, sparse_n*B, D), else: (S//sparse_n, sparse_n*B, D)
            pad_len: 0 or padding
        """
        length = x.shape[0]
        assert length == frame * height * width
        pad_len = 0
        if length % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - length % (self.sparse_n * self.sparse_n)
        if pad_len != 0:
            x = mint.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_len), mode="constant", value=0.0)

        _, b, d = x.shape
        if not self.sparse_group:
            # (g k) b d -> g (k b) d
            k = self.sparse_n
            x = x.reshape(-1, k, b, d).reshape(-1, k * b, d)
        else:
            # (n m k) b d -> (n k) (m b) d
            m = self.sparse_n
            k = self.sparse_n
            x = x.reshape(-1, m, k, b, d).permute(0, 2, 1, 3, 4).reshape(-1, m * b, d)

        return x, pad_len

    def _reverse_sparse_1d(self, x, frame, height, width, pad_len):
        """
        require the shape of (ntokens x batch_size x dim)

        Convert sparse groups back to original dimension
        Input:
            x: shape in S,B,D
        Output:
            x: shape if sparse_group: (S*sparse_n, B//sparse_n, D), else: (S*sparse_n, B//sparse_n, D)
        """
        assert x.shape[0] == (frame * height * width + pad_len) // self.sparse_n
        g, _, d = x.shape
        if not self.sparse_group:
            # g (k b) d -> (g k) b d
            k = self.sparse_n
            x = x.reshape(g, k, -1, d).reshape(g * k, -1, d)
        else:
            # (n k) (m b) d -> (n m k) b d
            m = self.sparse_n
            k = self.sparse_n
            assert g % k == 0
            n = g // k
            x = x.reshape(n, k, m, -1, d).permute(0, 2, 1, 3, 4).reshape(n * m * k, -1, d)
        x = x[: frame * height * width, :, :]
        return x

    def _sparse_1d_kv(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        # s b d -> s (k b) d
        # x = repeat(x, 's b d -> s (k b) d', k = self.sparse_n) # original
        # x = x.repeat(self.sparse_n, axis = 1) # WRONG!!!
        x = x.tile((1, self.sparse_n, 1))
        return x

    def __call__(
        self,
        attn: Attention,
        hidden_states: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        frame: int = 8,
        height: int = 16,
        width: int = 16,
    ) -> ms.Tensor:
        # residual = hidden_states
        sequence_length, batch_size, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # attention_mask shape
        if attention_mask.ndim == 3:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size, out_dim=4)

        # print(f"hidden_states.shape {hidden_states.shape}") #BSH
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        FA_head_num = attn.heads
        total_frame = frame

        if get_sequence_parallel_state():  # TODO: to test
            sp_size = hccl_info.world_size
            FA_head_num = attn.heads // sp_size
            total_frame = frame * sp_size
            # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
            query = self.alltoall_sbh_q(query.view(-1, attn.heads, head_dim))
            key = self.alltoall_sbh_k(key.view(-1, attn.heads, head_dim))
            value = self.alltoall_sbh_v(value.view(-1, attn.heads, head_dim))

        # print(f'batch: {batch_size}, FA_head_num: {FA_head_num}, head_dim: {head_dim}, total_frame:{total_frame}')
        query = query.view(-1, batch_size, FA_head_num, head_dim)
        key = key.view(-1, batch_size, FA_head_num, head_dim)

        # print(f'q {query.shape}, k {key.shape}, v {value.shape}')
        if not self.is_cross_attn:
            # require the shape of (ntokens x batch_size x nheads x dim) or (batch_size x ntokens x nheads x dim)
            pos_thw = self.position_getter(batch_size, t=total_frame, h=height, w=width)
            # print(f'pos_thw {pos_thw}')
            query = self.rope(query, pos_thw)
            key = self.rope(key, pos_thw)

        query = query.view(-1, batch_size, FA_head_num * head_dim)
        key = key.view(-1, batch_size, FA_head_num * head_dim)
        value = value.view(-1, batch_size, FA_head_num * head_dim)

        if self.sparse1d:
            query, pad_len = self._sparse_1d(query, total_frame, height, width)
            if self.is_cross_attn:
                key = self._sparse_1d_kv(key)
                value = self._sparse_1d_kv(value)
            else:
                key, pad_len = self._sparse_1d(key, total_frame, height, width)
                value, pad_len = self._sparse_1d(value, total_frame, height, width)

        # print(f'q {query.shape}, k {key.shape}, v {value.shape}')
        query = query.swapaxes(0, 1)  # SBH to BSH
        key = key.swapaxes(0, 1)
        value = value.swapaxes(0, 1)
        hidden_states = npu_config.run_attention(
            query, key, value, attention_mask, input_layout="BNSD", head_dim=head_dim, head_num=FA_head_num
        )
        hidden_states = hidden_states.swapaxes(0, 1)  # BSH -> SBH

        if self.sparse1d:
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frame, height, width, pad_len)

        # [s, b, h // sp * d] -> [s // sp * b, h, d] -> [s // sp, b, h * d]
        if get_sequence_parallel_state():
            hidden_states = self.alltoall_sbh_out(hidden_states.reshape(-1, FA_head_num, head_dim))
            hidden_states = hidden_states.view(-1, batch_size, inner_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class BasicTransformerBlock(nn.Cell):
    @ms.lazy_inline(policy="front")
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        interpolation_scale_thw: Tuple[int] = (1, 1, 1),
        sparse1d: bool = False,
        sparse_n: int = 2,
        sparse_group: bool = False,
        FA_dtype=ms.bfloat16,
    ):
        super().__init__()

        if npu_config.enable_FA:
            FLASH_IS_AVAILABLE = check_valid_flash_attention()
            npu_config.enable_FA = FLASH_IS_AVAILABLE and npu_config.enable_FA
        if npu_config.enable_FA:
            npu_config.FA_dtype = FA_dtype
            assert FA_dtype in [ms.float16, ms.bfloat16], f"Unsupported flash-attention dtype: {FA_dtype}"

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.norm1 = LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            interpolation_scale_thw=interpolation_scale_thw,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=False,
        )

        # 2. Cross-Attn
        self.norm2 = LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
            interpolation_scale_thw=interpolation_scale_thw,
            sparse1d=sparse1d,
            sparse_n=sparse_n,
            sparse_group=sparse_group,
            is_cross_attn=True,
        )

        # 3. Feed-forward
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Scale-shift.
        self.scale_shift_table = Parameter(ops.randn((6, dim)) / dim**0.5)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor],
        encoder_hidden_states: Optional[ms.Tensor],
        encoder_attention_mask: Optional[ms.Tensor],
        timestep: Optional[ms.Tensor],
        frame: int,
        height: int,
        width: int,
    ) -> ms.Tensor:
        # 0. Self-Attention
        batch_size = hidden_states.shape[1]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(
            self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1), 6, dim=0
        )

        norm_hidden_states = self.norm1(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask,
            frame=frame,
            height=height,
            width=width,
        )

        attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 3. Cross-Attention
        norm_hidden_states = hidden_states

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=encoder_attention_mask,
            frame=frame,
            height=height,
            width=width,
        )

        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states
