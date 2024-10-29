import logging
import numbers
from typing import Optional, Tuple

import numpy as np
from opensora.acceleration.communications import AllToAll_SBH
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info

import mindspore as ms
from mindspore import Parameter, mint, nn, ops
from mindspore.common.initializer import initializer

from mindone.diffusers.models.attention import FeedForward
from mindone.diffusers.models.attention_processor import Attention as Attention_
from mindone.utils.version_control import check_valid_flash_attention, choose_flash_attention_dtype

from .rope import PositionGetter3D, RoPE3D

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

# Different from v1.2
class PatchEmbed2D(nn.Cell):
    """2D Image to Patch Embedding but with video"""

    def __init__(
        self,
        patch_size=16, #2
        in_channels=3, #8
        embed_dim=768, # 24*96=2304
        bias=True,
    ):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size), has_bias=bias
        )

    def construct(self, latent):
        b, c, t, h, w = latent.shape # b, c=in_channels, t, h, w
        # b c t h w -> (b t) c h w
        latent = latent.permute(0, 2, 1, 3, 4).reshape(b*t, c, h, w) # b*t, c, h, w
        latent = self.proj(latent)  # b*t, embed_dim, h, w
        # (b t) c h w -> b (t h w) c
        _, c, h, w = latent.shape
        latent = latent.reshape(b, -1, c, h, w).permute(0, 1, 3, 4, 2).reshape(b, -1, c) # b, t*h*w, embed_dim
        
        return latent


def get_attention_mask(attention_mask, repeat_num, attention_mode="xformers"):
    if attention_mask is not None:
        if attention_mode != "math":
            attention_mask = attention_mask.to(ms.bool_)
        else:
            attention_mask = attention_mask.repeat_interleave(repeat_num, dim=-2)
    return attention_mask


class Attention(Attention_):
    def __init__(
            self, interpolation_scale_thw, sparse1d, sparse_n, 
            sparse_group, is_cross_attn, attention_mode="xformers", **kwags
            ):
        FA_dtype = kwags.pop("FA_dtype", ms.bfloat16)
        processor = OpenSoraAttnProcessor2_0(
            interpolation_scale_thw=interpolation_scale_thw, sparse1d=sparse1d, sparse_n=sparse_n, 
            sparse_group=sparse_group, is_cross_attn=is_cross_attn,
            attention_mode=attention_mode,
            FA_dtype=FA_dtype, dim_head=kwags["dim_head"]
            )
        kwags["processor"] = processor
        super().__init__(**kwags)
        if attention_mode == "xformers":
            self.set_use_memory_efficient_attention_xformers(True)
        self.processor = processor
    
    @staticmethod
    def prepare_sparse_mask(attention_mask, encoder_attention_mask, sparse_n, head_num):
        
        attention_mask = attention_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
        l = attention_mask.shape[-1]
        if l % (sparse_n * sparse_n) == 0:
            pad_len = 0
        else:
            pad_len = sparse_n * sparse_n - l % (sparse_n * sparse_n)

        attention_mask_sparse = mint.nn.functional.pad(attention_mask, (0, pad_len, 0, 0), mode="constant", value=-9980.0)
        b = attention_mask_sparse.shape[0]
        k = sparse_n
        m = sparse_n
        # b 1 1 (g k) -> (k b) 1 1 g
        attention_mask_sparse_1d = attention_mask_sparse.reshape(b, 1, 1, -1, k).permute(4, 0, 1, 2, 3).reshape(b*k, 1, 1, -1)
        # b 1 1 (n m k) -> (m b) 1 1 (n k)
        attention_mask_sparse_1d_group = attention_mask_sparse.reshape(b, 1, 1, -1, m, k).permute(4, 0, 1, 2, 3, 5).reshape(m*b, 1, 1, -1)
        encoder_attention_mask_sparse = encoder_attention_mask.tile((sparse_n, 1, 1, 1))
        # if npu_config is not None:
        attention_mask_sparse_1d = get_attention_mask(
            attention_mask_sparse_1d, attention_mask_sparse_1d.shape[-1]
            )
        attention_mask_sparse_1d_group = get_attention_mask(
            attention_mask_sparse_1d_group, attention_mask_sparse_1d_group.shape[-1]
            )
        
        encoder_attention_mask_sparse_1d = get_attention_mask(
            encoder_attention_mask_sparse, attention_mask_sparse_1d.shape[-1]
            )
        encoder_attention_mask_sparse_1d_group = encoder_attention_mask_sparse_1d
        # else:
            # attention_mask_sparse_1d = attention_mask_sparse_1d.repeat_interleave(head_num, dim=1)
            # attention_mask_sparse_1d_group = attention_mask_sparse_1d_group.repeat_interleave(head_num, dim=1)

            # encoder_attention_mask_sparse_1d = encoder_attention_mask_sparse.repeat_interleave(head_num, dim=1)
            # encoder_attention_mask_sparse_1d_group = encoder_attention_mask_sparse_1d
        
        return {
                    False: (attention_mask_sparse_1d, encoder_attention_mask_sparse_1d),
                    True: (attention_mask_sparse_1d_group, encoder_attention_mask_sparse_1d_group)
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

    def __init__(self, interpolation_scale_thw=(1, 1, 1), 
                 sparse1d=False, sparse_n=2, sparse_group=False, is_cross_attn=True, 
                 FA_dtype=ms.bfloat16, dim_head=64, attention_mode = "xformers"):
        self.sparse1d = sparse1d
        self.sparse_n = sparse_n
        self.sparse_group = sparse_group
        self.is_cross_attn = is_cross_attn
        self.interpolation_scale_thw = interpolation_scale_thw
        self.attention_mode = attention_mode
        
        self._init_rope(interpolation_scale_thw, dim_head=dim_head)

        self.attention_mode = "xformers"  #TBD
        # Currently we only support setting attention_mode to `flash` or `math`
        assert self.attention_mode in [
            "xformers",
            "math",
        ], f"Unsupported attention mode {self.attention_mode}. Currently we only support ['xformers', 'math']!"
        self.enable_FA = self.attention_mode == "xformers"
        self.FA_dtype = FA_dtype
        assert self.FA_dtype in [ms.float16, ms.bfloat16], f"Unsupported flash-attention dtype: {self.FA_dtype}"
        if self.enable_FA:
            FLASH_IS_AVAILABLE = check_valid_flash_attention()
            self.enable_FA = FLASH_IS_AVAILABLE and self.enable_FA

        self.fa_mask_dtype = choose_flash_attention_dtype()
        if get_sequence_parallel_state():
            self.sp_size = hccl_info.world_size
            self.alltoall_sbh_q = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_k = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_v = AllToAll_SBH(scatter_dim=1, gather_dim=0)
            self.alltoall_sbh_out = AllToAll_SBH(scatter_dim=1, gather_dim=0)
        else:
            self.sp_size = 1
            self.alltoall_sbh_q = None
            self.alltoall_sbh_k = None
            self.alltoall_sbh_v = None
            self.alltoall_sbh_out = None

    def _init_rope(self, interpolation_scale_thw, dim_head):
        self.rope = RoPE3D(interpolation_scale_thw=interpolation_scale_thw, dim_head=dim_head)
        self.position_getter = PositionGetter3D()

    def run_ms_flash_attention(
        self,
        attn,
        query,
        key,
        value,
        attention_mask,
        input_layout="BSH",
        attention_dropout: float = 0.0,
    ):
        # Memory efficient attention on mindspore uses flash attention under the hoods.
        # Flash attention implementation is called `FlashAttentionScore`
        # which is an experimental api with the following limitations:
        # 1. Sequence length of query must be divisible by 16 and in range of [1, 32768].
        # 2. Head dimensions must be one of [64, 80, 96, 120, 128, 256].
        # 3. The input dtype must be float16 or bfloat16.
        # Sequence length of query must be checked in runtime.
        if input_layout not in ["BSH", "BNSD"]:
            raise ValueError(f"input_layout must be in ['BSH', 'BNSD'], but get {input_layout}.")
        Bs, query_tokens, _ = query.shape
        assert query_tokens % 16 == 0, f"Sequence length of query must be divisible by 16, but got {query_tokens=}."
        key_tokens = key.shape[1]
        heads = attn.heads if not get_sequence_parallel_state() else attn.heads // hccl_info.world_size
        query = query.view(Bs, query_tokens, heads, -1)
        key = key.view(Bs, key_tokens, heads, -1)
        value = value.view(Bs, key_tokens, heads, -1)
        # Head dimension is checked in Attention.set_use_memory_efficient_attention_xformers. We maybe pad on head_dim.
        if attn.head_dim_padding > 0:
            query_padded = mint.nn.functional.pad(query, (0, attn.head_dim_padding), mode="constant", value=0.0)
            key_padded = mint.nn.functional.pad(key, (0, attn.head_dim_padding), mode="constant", value=0.0)
            value_padded = mint.nn.functional.pad(value, (0, attn.head_dim_padding), mode="constant", value=0.0)
        else:
            query_padded, key_padded, value_padded = query, key, value
        flash_attn = ops.operations.nn_ops.FlashAttentionScore(
            scale_value=attn.scale, head_num=heads, input_layout=input_layout, keep_prob=1 - attention_dropout
        )
        if attention_mask is not None:
            # flip mask, since ms FA treats 1 as discard, 0 as retain.
            attention_mask = ~attention_mask if attention_mask.dtype == ms.bool_ else 1 - attention_mask
            # (b, 1, 1, k_n) - > (b, 1, q_n, k_n), manual broadcast
            if attention_mask.shape[-2] == 1:
                attention_mask = mint.tile(attention_mask.bool(), (1, 1, query_tokens, 1))            
            attention_mask = attention_mask.to(self.fa_mask_dtype)

        if input_layout == "BNSD":
            # (b s n d) -> (b n s d)
            query_padded = query_padded.swapaxes(1, 2)
            key_padded = key_padded.swapaxes(1, 2)
            value_padded = value_padded.swapaxes(1, 2)
        elif input_layout == "BSH":
            query_padded = query_padded.view(Bs, query_tokens, -1)
            key_padded = key_padded.view(Bs, key_tokens, -1)
            value_padded = value_padded.view(Bs, key_tokens, -1)
        hidden_states_padded = flash_attn(
            query_padded.to(self.FA_dtype),
            key_padded.to(self.FA_dtype),
            value_padded.to(self.FA_dtype),
            None,
            None,
            None,
            attention_mask,
        )[3]
        # If we did padding before calculate attention, undo it!
        if attn.head_dim_padding > 0:
            if input_layout == "BNSD":
                hidden_states = hidden_states_padded[..., : attn.head_dim]
            else:
                hidden_states = hidden_states_padded.view(Bs, query_tokens, heads, -1)[..., : attn.head_dim]
                hidden_states = hidden_states.view(Bs, query_tokens, -1)
        else:
            hidden_states = hidden_states_padded
        if input_layout == "BNSD":
            # b n s d -> b s n d
            hidden_states = hidden_states.swapaxes(1, 2)
        hidden_states = hidden_states.reshape(Bs, query_tokens, -1)
        hidden_states = hidden_states.to(query.dtype)
        return hidden_states

    def run_math_attention(self, attn, query, key, value, attention_mask):
        _head_size = attn.heads if not get_sequence_parallel_state() else attn.heads // hccl_info.world_size
        query = self._head_to_batch_dim(_head_size, query)
        key = self._head_to_batch_dim(_head_size, key)
        value = self._head_to_batch_dim(_head_size, value)

        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqeeuze(1)
            assert attention_mask.shape[1] == 1
            attention_mask = attention_mask.repeat_interleave(_head_size, 1)
            attention_mask = attention_mask.reshape(-1, attention_mask.shape[-2], attention_mask.shape[-1])
            attention_mask = mint.zeros(attention_mask.shape).masked_fill(attention_mask.to(ms.bool_), -10000.0)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = mint.bmm(attention_probs, value)
        hidden_states = self._batch_to_head_dim(_head_size, hidden_states)
        return hidden_states

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
        l = x.shape[0]
        assert l == frame*height*width
        pad_len = 0
        if l % (self.sparse_n * self.sparse_n) != 0:
            pad_len = self.sparse_n * self.sparse_n - l % (self.sparse_n * self.sparse_n)
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
            x = x.reshape(-1, m, k, b, d).permute(0, 2, 1, 3, 4).reshape(-1, m*b, d)
            
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
        assert x.shape[0] == (frame*height*width+pad_len) // self.sparse_n
        g, _, d = x.shape
        if not self.sparse_group:
            # g (k b) d -> (g k) b d
            k = self.sparse_n
            x = x.reshape(g, k, -1, d).reshape(g*k, -1, d)
        else:
            # (n k) (m b) d -> (n m k) b d
            m = self.sparse_n
            k = self.sparse_n
            assert g % k == 0
            n = g // k
            x = x.reshape(n, k, m, -1, d).permute(0, 2, 1, 3, 4).reshape(n*m*k, -1, d)
        x = x[:frame*height*width, :, :]
        return x
    
    def _sparse_1d_kv(self, x):
        """
        require the shape of (ntokens x batch_size x dim)
        """
        # s b d -> s (k b) d
        x = x.repeat(self.sparse_n, axis = 1)
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
        *args,
        **kwargs,
    ) -> ms.Tensor:

        residual = hidden_states 

        if get_sequence_parallel_state():
            sequence_length, batch_size, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )
        else:
            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            ) #BSH

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

        if get_sequence_parallel_state(): #TODO: to test
            sp_size = hccl_info.world_size
            FA_head_num = attn.heads // sp_size
            total_frame = frame * sp_size
            # apply all_to_all to gather sequence and split attention heads [s // sp * b, h, d] -> [s * b, h // sp, d]
            query = self.alltoall_sbh_q(query.view(-1, attn.heads, head_dim))
            key = self.alltoall_sbh_k(key.view(-1, attn.heads, head_dim))
            value = self.alltoall_sbh_v(value.view(-1, attn.heads, head_dim))
        
            # print(f'batch: {batch_size}, FA_head_num: {FA_head_num}, head_dim: {head_dim}, total_frame:{total_frame}')
            query = query.view(-1, batch_size, FA_head_num, head_dim)# BUG? TODO: to test
            key = key.view(-1, batch_size, FA_head_num, head_dim) #BUG ?

            # print(f'q {query.shape}, k {key.shape}, v {value.shape}')
            if not self.is_cross_attn:
                # require the shape of (ntokens x batch_size x nheads x dim) 
                pos_thw = self.position_getter(batch_size, t=total_frame, h=height, w=width)
                # print(f'pos_thw {pos_thw}')
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)
            
            query = query.view(-1, batch_size, FA_head_num * head_dim)
            key = key.view(-1, batch_size, FA_head_num * head_dim)
            value = value.view(-1, batch_size, FA_head_num * head_dim)
        else:
            # print(f'batch: {batch_size}, FA_head_num: {FA_head_num}, head_dim: {head_dim}, total_frame:{total_frame}')
            query = query.view(batch_size, -1, FA_head_num, head_dim)
            key = key.view(batch_size, -1, FA_head_num, head_dim) 
            # (batch_size x ntokens x nheads x dim)
            
            # print(f'q {query.shape}, k {key.shape}, v {value.shape}')
            if not self.is_cross_attn:
                # require the shape of (batch_size x ntokens x nheads x dim)
                pos_thw = self.position_getter(batch_size, t=total_frame, h=height, w=width)
                # print(f'pos_thw {pos_thw}')
                query = self.rope(query, pos_thw)
                key = self.rope(key, pos_thw)
            
            query = query.view(batch_size, -1, FA_head_num * head_dim).swapaxes(0, 1)
            key = key.view(batch_size, -1, FA_head_num * head_dim).swapaxes(0, 1)
            value = value.swapaxes(0, 1)
            
        # print(f'q {query.shape}, k {key.shape}, v {value.shape}') #(SBH) 

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
        if self.attention_mode == "math":
            # FIXME: shape error
            hidden_states = self.run_math_attention(attn, query, key, value, attention_mask)
        elif self.attention_mode == "xformers":
            hidden_states = self.run_ms_flash_attention(attn, query, key, value, attention_mask)
        # if npu_config is not None:
        #     hidden_states = npu_config.run_attention(query, key, value, attention_mask, "SBH", head_dim, FA_head_num)
        # else:
        #     query = rearrange(query, 's b (h d) -> b h s d', h=FA_head_num)
        #     key = rearrange(key, 's b (h d) -> b h s d', h=FA_head_num)
        #     value = rearrange(value, 's b (h d) -> b h s d', h=FA_head_num)
        #     # 0, -10000 ->(bool) False, True ->(any) True ->(not) False
        #     # 0, 0 ->(bool) False, False ->(any) False ->(not) True
        #     # if attention_mask is None or not torch.any(attention_mask.bool()):  # 0 mean visible
        #     #     attention_mask = None
        #     # the output of sdp = (batch, num_heads, seq_len, head_dim)
        #     with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=False, enable_mem_efficient=True):
        #         hidden_states = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask) # dropout_p=0.0, is_causal=False
        #     hidden_states = rearrange(hidden_states, 'b h s d -> s b (h d)', h=FA_head_num)

        if self.sparse1d:
            hidden_states = hidden_states.swapaxes(0, 1) # BSH -> SBH
            hidden_states = self._reverse_sparse_1d(hidden_states, total_frame, height, width, pad_len)
            hidden_states = hidden_states.swapaxes(0, 1) # SBH -> BSH

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
        attention_mode: str = "xformers",
        FA_dtype=ms.bfloat16,
    ):
        super().__init__()
        self.FA_dtype = FA_dtype

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
            attention_mode=attention_mode,
            FA_dtype=self.FA_dtype,
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
            attention_mode=attention_mode,
            FA_dtype=self.FA_dtype,
        )  # is self-attn if encoder_hidden_states is none

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
        attention_mask: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        timestep: Optional[ms.Tensor] = None,
        frame: int = None, 
        height: int = None, 
        width: int = None, 
    ) -> ms.Tensor:
        
        # 0. Self-Attention
        if get_sequence_parallel_state():
            batch_size = hidden_states.shape[1]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(
                self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1), 6, dim=0)
        else:
            batch_size = hidden_states.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1), 6, dim=1
            )        

        norm_hidden_states = self.norm1(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=None,
            attention_mask=attention_mask, frame=frame, height=height, width=width, 
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
            attention_mask=encoder_attention_mask, frame=frame, height=height, width=width,
        )

        hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm2(hidden_states)

        norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states

        return hidden_states