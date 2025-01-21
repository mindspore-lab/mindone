from typing import Optional
import mindspore as ms
from mindspore import nn, ops
from .embed_layers import TimestepEmbedder, TextProjection
from .norm_layers import LayerNorm, FP32LayerNorm
from .mlp_layers import MLP
from .attention import VanillaAttention, FlashAttention
from .norm_layers import get_norm_layer
from .activation_layers import get_activation_layer
from .modulate_layers import apply_gate


def rearrange_qkv(qkv, heads_num):
    # qkv: shape (B L K*H*D), K=3
    # B L (K H D) -> B L K H D -> K B L H D
    # return q/k/v: (B L H D)
    B, L, KHD = qkv.shape
    H = heads_num
    # D = head_dim # KHD // (K * self.heads_num)
    D = KHD // (3 * H)
    qkv = ops.reshape(qkv, (B, L, 3, H, D))
    q, k, v = ops.split(qkv, 1, axis=2)
    q = ops.squeeze(q, axis=2)
    k = ops.squeeze(k, axis=2)
    v = ops.squeeze(v, axis=2)

    return q, k, v


class IndividualTokenRefinerBlock(nn.Cell):
    def __init__(
        self,
        hidden_size,
        heads_num,
        mlp_width_ratio: str = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = 'flash',
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.norm1 = FP32LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        self.self_attn_qkv = nn.Dense(
            hidden_size, hidden_size * 3, has_bias=qkv_bias,
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.self_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.self_attn_proj = nn.Dense(
            hidden_size, hidden_size, has_bias=qkv_bias,
        )

        self.norm2 = FP32LayerNorm(
            hidden_size, elementwise_affine=True, eps=1e-6, **factory_kwargs
        )
        act_layer = get_activation_layer(act_type)
        self.mlp = MLP(
            in_channels=hidden_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop_rate,
            **factory_kwargs,
        )

        self.adaLN_modulation = nn.SequentialCell(
            act_layer(),
            nn.Dense(hidden_size, 2 * hidden_size, has_bias=True, weight_init='zeros', bias_init='zeros'),
        )

        if attn_mode == 'vanilla':
            self.compute_attention = VanillaAttention(head_dim)
        elif attn_mode == 'flash':
            self.compute_attention = FlashAttention(heads_num, head_dim)
        else:
            raise NotImplementedError

    def construct(
        self,
        x: ms.Tensor,
        c: ms.Tensor,  # timestep_aware_representations + context_aware_representations
        attn_mask: ms.Tensor = None,
    ):
        gate_msa, gate_mlp = self.adaLN_modulation(c).chunk(2, axis=1)

        norm_x = self.norm1(x)
        qkv = self.self_attn_qkv(norm_x)

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q, k, v = rearrange_qkv(qkv, self.heads_num)

        # Apply QK-Norm if needed
        q = self.self_attn_q_norm(q) # .to(v)
        k = self.self_attn_k_norm(k) # .to(v)

        # Self-Attention
        # TODO; support attn_mask
        #  import pdb; pdb.set_trace()
        attn = self.compute_attention(q, k, v, mask=attn_mask)

        x = x + apply_gate(self.self_attn_proj(attn), gate_msa)

        # FFN Layer
        x = x + apply_gate(self.mlp(self.norm2(x)), gate_mlp)

        return x


class IndividualTokenRefiner(nn.Cell):
    def __init__(
        self,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = 'flash',
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.blocks = nn.CellList(
            [
                IndividualTokenRefinerBlock(
                    hidden_size=hidden_size,
                    heads_num=heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_drop_rate=mlp_drop_rate,
                    act_type=act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    attn_mode=attn_mode,
                    **factory_kwargs,
                )
                for _ in range(depth)
            ]
        )

    def construct(
        self,
        x: ms.Tensor,
        c: ms.Tensor,
        mask: Optional[ms.Tensor] = None,
    ):
        self_attn_mask = None
        if mask is not None:
            # mask shape: (b, s)
            batch_size = mask.shape[0]
            seq_len = mask.shape[1]
            # TODO: check tile op
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_1 = mask.reshape((batch_size, 1, 1, seq_len)).tile(
                (1, 1, seq_len, 1),
            )
            # print('D--: attn mask 1 shape ', self_attn_mask_1.shape)
            # batch_size x 1 x seq_len x seq_len
            self_attn_mask_2 = self_attn_mask_1.transpose((0, 1, 3, 2))
            # batch_size x 1 x seq_len x seq_len, 1 for broadcasting of heads_num
            self_attn_mask = ops.logical_and(self_attn_mask_1, self_attn_mask_2)
            # self_attn_mask = (self_attn_mask_1 & self_attn_mask_2).bool()
            # avoids self-attention weight being NaN for padding tokens

            # TODO: this slicing operation can be slow
            self_attn_mask[:, :, :, 0] = True

        for block in self.blocks:
            x = block(x, c, self_attn_mask)
        return x


class SingleTokenRefiner(nn.Cell):
    """
    A single token refiner block for llm text embedding refine.
    """
    def __init__(
        self,
        in_channels,
        hidden_size,
        heads_num,
        depth,
        mlp_width_ratio: float = 4.0,
        mlp_drop_rate: float = 0.0,
        act_type: str = "silu",
        qk_norm: bool = False,
        qk_norm_type: str = "layer",
        qkv_bias: bool = True,
        attn_mode: str = "flash",
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()
        self.attn_mode = attn_mode

        self.input_embedder = nn.Dense(
            in_channels, hidden_size, has_bias=True,
        )

        act_layer = get_activation_layer(act_type)
        # Build timestep embedding layer
        self.t_embedder = TimestepEmbedder(hidden_size, act_layer, **factory_kwargs)
        # Build context embedding layer
        self.c_embedder = TextProjection(
            in_channels, hidden_size, act_layer, **factory_kwargs
        )

        self.individual_token_refiner = IndividualTokenRefiner(
            hidden_size=hidden_size,
            heads_num=heads_num,
            depth=depth,
            mlp_width_ratio=mlp_width_ratio,
            mlp_drop_rate=mlp_drop_rate,
            act_type=act_type,
            qk_norm=qk_norm,
            qk_norm_type=qk_norm_type,
            qkv_bias=qkv_bias,
            attn_mode=attn_mode,
            **factory_kwargs,
        )

        self.dtype = dtype

    def construct(
        self,
        x: ms.Tensor,
        t: ms.Tensor,
        mask: Optional[ms.Tensor] = None,
    ):
        '''
        Inputs:
            x: float16, (B, S_token_padded, emb_dim), text embedding (from llama)
            t: float32, (1,), e.g. [1000.]
            mask: int, (B, S_token_padded)
        Output:
            (B, S_token_padded, out_emb_dim)
        '''
        # import pdb; pdb.set_trace()

        # AMP: t (fp32) -> TimestepEmbed (sinusoidal, mlp) -> bf16
        # (B, hidden_dim)
        timestep_aware_representations = self.t_embedder(t)

        if mask is None:
            context_aware_representations = x.mean(axis=1)
        else:
            mask_float = mask.float().unsqueeze(-1)  # [b, s1, 1]
            # TODO: AMP: x fp16, mask_float fp32, should we upcast x to fp32 manully?
            context_aware_representations = (x * mask_float).sum(
                axis=1
            ) / mask_float.sum(axis=1)
        # AMP: car -> c_embedder mlp (bf16) -> bf16
        context_aware_representations = self.c_embedder(context_aware_representations.to(self.dtype))
        c = timestep_aware_representations + context_aware_representations
        
        # AMP: linear bf16, out x bf16
        x = self.input_embedder(x.to(self.dtype))
        
        # AMP: x bf16, c float32; c -> adaLN_modulation (silu, linear)
        x = self.individual_token_refiner(x, c.to(self.dtype), mask)

        return x
