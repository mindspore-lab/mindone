import logging
from typing import Optional, Union

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import XavierUniform, Zero, initializer

from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention

from .dit import GELU, Attention, FinalLayer, LayerNorm, Mlp, PatchEmbed, TimestepEmbedder
from .modules import apply_2d_rotary_pos, create_sinusoidal_positions, get_2d_sincos_pos_embed
from .utils import constant_, modulate, normal_, xavier_uniform_

logger = logging.getLogger(__name__)

__all__ = [
    "MMDiT",
    "MMDiT_models",
    "MMDiT_XL_2",
    "MMDiT_XL_4",
    "MMDiT_XL_8",
    "MMDiT_L_2",
    "MMDiT_L_4",
    "MMDiT_L_8",
    "MMDiT_B_2",
    "MMDiT_B_4",
    "MMDiT_B_8",
    "MMDiT_S_2",
    "MMDiT_S_4",
    "MMDiT_S_8",
]


class RMSNorm(nn.Cell):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
        Root Mean Square Layer Normalization, https://arxiv.org/abs/1910.07467
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = Parameter(ops.ones(d))

        if self.bias:
            self.offset = Parameter(ops.zeros(d))

    def construct(self, x: Tensor):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = ops.split(x, [partial_size, self.d - partial_size], axis=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class JointAttention(nn.Cell):
    """
    Args:
        dim (int): hidden size.
        num_heads (int): number of heads
        qkv_bias (int): whether to use bias
        attn_drop (bool): attention dropout
        proj_drop (bool): projection dropout
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        dtype=ms.float32,
        enable_flash_attention=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dtype = dtype
        self.num_heads = num_heads
        self.dim = dim
        head_dim = dim // num_heads

        # q k v projection
        self.qkv_x = nn.Dense(dim, dim * 3, has_bias=qkv_bias, weight_init=XavierUniform(), bias_init=Zero()).to_float(
            self.dtype
        )
        self.qkv_c = nn.Dense(dim, dim * 3, has_bias=qkv_bias, weight_init=XavierUniform(), bias_init=Zero()).to_float(
            self.dtype
        )
        # output projection
        self.proj_x = nn.Dense(dim, dim, weight_init=XavierUniform(), bias_init=Zero()).to_float(self.dtype)
        self.proj_c = nn.Dense(dim, dim, weight_init=XavierUniform(), bias_init=Zero()).to_float(self.dtype)
        # dropout
        self.proj_drop_x = nn.Dropout(p=proj_drop)
        self.proj_drop_c = nn.Dropout(p=proj_drop)

        # QK-Normalization, use RMSNorm
        self.q_norm_x = RMSNorm(head_dim)
        self.k_norm_x = RMSNorm(head_dim)
        self.q_norm_c = RMSNorm(head_dim)
        self.k_norm_c = RMSNorm(head_dim)

        self.attention = Attention(head_dim, attn_drop=attn_drop)

        self.enable_flash_attention = (
            enable_flash_attention and FLASH_IS_AVAILABLE and (ms.context.get_context("device_target") == "Ascend")
        )

        if self.enable_flash_attention:
            self.flash_attention = MSFlashAttention(
                head_dim=head_dim, head_num=num_heads, fix_head_dims=[72], attention_dropout=attn_drop
            )
        else:
            self.flash_attention = None

    def construct(self, x, c, mask=None, spatial_freq=None):
        x_dtype = x.dtype
        h = self.num_heads

        # latent
        B, T, C = x.shape
        hd = C // h  # head dim
        q_x, k_x, v_x = self.qkv_x(x).split(self.dim, axis=2)
        q_x = q_x.view(B, T, h, hd).swapaxes(1, 2)
        k_x = k_x.view(B, T, h, hd).swapaxes(1, 2)
        v_x = v_x.view(B, T, h, hd).swapaxes(1, 2)
        # QK-RMSNorm over head dim
        q_x = self.q_norm_x(q_x)
        k_x = self.k_norm_x(k_x)

        if spatial_freq is not None:
            q_x, k_x = apply_2d_rotary_pos(q_x, k_x, freqs_cis=spatial_freq)

        # text
        _, L, _ = c.shape
        hd = C // h  # head dim
        q_c, k_c, v_c = self.qkv_c(c).split(self.dim, axis=2)
        q_c = q_c.view(B, L, h, hd).swapaxes(1, 2)
        k_c = k_c.view(B, L, h, hd).swapaxes(1, 2)
        v_c = v_c.view(B, L, h, hd).swapaxes(1, 2)
        # QK-RMSNorm over head dim
        q_c = self.q_norm_c(q_c)
        k_c = self.k_norm_c(k_c)

        # (B, num_head, T+L, head_dim)
        q = ops.cat([q_x, q_c], axis=2)
        k = ops.cat([k_x, k_c], axis=2)
        v = ops.cat([v_x, v_c], axis=2)

        if mask is not None:
            mask = ops.reshape(mask, (mask.shape[0], -1))
            attn_mask = ops.zeros((B, T + L, T + L))
            mask = ops.expand_dims(mask, axis=1)
            attn_mask = attn_mask.masked_fill(~mask, -ms.numpy.inf)
            mask = attn_mask

        if self.enable_flash_attention and (T + L) % 16 == 0 and hd <= 256:
            out = self.flash_attention(q, k, v, mask)
            out = out.swapaxes(1, 2).view(B, T + L, C)

        else:
            # b, nh, T+L, hd -> b*nh, T+L, hd
            q = ops.reshape(q, (B * h, T + L, hd))
            k = ops.reshape(k, (B * h, T + L, hd))
            v = ops.reshape(v, (B * h, T + L, hd))
            if mask is not None and mask.shape[0] != q.shape[0]:
                mask = mask.repeat_interleave(h, dim=0)
            out = self.attention(q, k, v, mask)
            out = out.swapaxes(1, 2).view(B, T + L, C)  # b, nh, T+L, hd -> b, T+L, nh*hd=C

        # split to latent_out and text_out
        x = out[:, :T, :]  # B T C
        c = out[:, T:, :]  # B L C

        # output projection
        x = self.proj_drop_x(self.proj_x(x)).to(x_dtype)
        c = self.proj_drop_c(self.proj_c(c)).to(x_dtype)

        return x, c


class MMDiTBlock(nn.Cell):
    """
    A MMDiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()

        self.norm_x_1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_c_1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = JointAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm_x_2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_c_2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp_x = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.mlp_c = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        self.adaLN_modulation_x = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.adaLN_modulation_c = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))

    def construct(self, x, c, t, mask=None, spatial_freq=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_x(t).chunk(
            6, axis=1
        )
        shift_msa_c, scale_msa_c, gate_msa_c, shift_mlp_c, scale_mlp_c, gate_mlp_c = self.adaLN_modulation_c(t).chunk(
            6, axis=1
        )

        x1 = modulate(self.norm_x_1(x), shift_msa_x, scale_msa_x)
        c1 = modulate(self.norm_c_1(c), shift_msa_c, scale_msa_c)
        x1, c1 = self.attn(x1, c1, mask, spatial_freq)

        x = x + gate_msa_x.unsqueeze(1) * x1
        c = c + gate_msa_c.unsqueeze(1) * c1

        x = x + gate_mlp_x.unsqueeze(1) * self.mlp_x(modulate(self.norm_x_2(x), shift_mlp_x, scale_mlp_x))
        c = c + gate_mlp_c.unsqueeze(1) * self.mlp_c(modulate(self.norm_c_2(c), shift_mlp_c, scale_mlp_c))

        return x, c


class MMDiT(nn.Cell):
    """A diffusion model with a Transformer backbone.
    Args:
        input_size (int, default=32): The size of the input latent.
        patch_size (int, default=2): The size of each patch in the input latent. The input latent is divided into patches of patch_size x patch_size.
        in_channels (int, default=4): The number of input channels in the input latent.
        text_in_channels (int, default=4096): The number of input channels of the text_embedding_projection, which is the text emb dim.
        text_pooled_in_channels (int, default=2048): The number of input channels of the timestep_projection, which is the text pooled emb dim.
        hidden_size (int, default=1152): The hidden size of the Transformer model.
        depth (int, default=28): The number of blocks in this Transformer.
        num_heads (int, default=16): The number of attention heads.
        mlp_ratio (float, default=4.0): The expansion ratio for the hidden dimension in the MLP of the Transformer.
        num_classes (int, default=1000): The number of classes of the input labels.
        learn_sigma (bool, default=True): Whether to learn the diffusion model's sigma parameter.
        block_kwargs (dict, default={}): Additional keyword arguments for the Transformer blocks. for example, {'enable_flash_attention':True}
        use_rel_pos (bool, default=False): Whether to use Rotary Position Embedding (RoPE) in attention.
        use_recompute (bool, default=False): Whether to use recompute.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        text_in_channels=4096,
        text_pooled_in_channels=2048,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
        block_kwargs={},
        use_rel_pos=False,
        use_recompute=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.text_in_channels = text_in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.attention_head_dim = hidden_size // num_heads
        self.use_recompute = use_recompute

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.num_patches = self.x_embedder.num_patches

        self.base_size = input_size // patch_size
        self.use_rel_pos = use_rel_pos

        if self.use_rel_pos:
            # use relative sin-cos embbedding
            spatial_position_id = [
                [i for i in range(self.base_size) for j in range(self.base_size)],
                [j for i in range(self.base_size) for j in range(self.base_size)],
            ]
            embed_positions_h = create_sinusoidal_positions(self.base_size, self.attention_head_dim // 2)
            embed_positions_w = create_sinusoidal_positions(self.base_size, self.attention_head_dim // 2)
            sincos_h = embed_positions_h[spatial_position_id[0]]
            sincos_w = embed_positions_w[spatial_position_id[1]]
            self.spatial_freq = (sincos_h, sincos_w)
        else:
            # use fixed sin-cos embbedding
            self.spatial_freq = None
            self.pos_embed = Parameter(ops.zeros(size=(1, self.num_patches, hidden_size)), requires_grad=False)

        self.text_embedding_projection = nn.Dense(text_in_channels, hidden_size, has_bias=True)
        self.timestep_projection = nn.Dense(text_pooled_in_channels, hidden_size, has_bias=True)

        self.blocks = nn.CellList(
            [MMDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        if self.use_recompute:
            for block in self.blocks:
                self.recompute(block)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        if not self.use_rel_pos:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
            self.pos_embed.set_data(Tensor(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        # xavier_uniform_(w.view(w.shape[0], -1))
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).init_data().reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize text_embedding_projection, caption-timestep projection:
        normal_(self.text_embedding_projection.weight, std=0.02)
        normal_(self.timestep_projection.weight, std=0.02)

        # Zero-out adaLN modulation layers in MMDiT blocks:
        for block in self.blocks:
            constant_(block.adaLN_modulation_x[-1].weight, 0)
            constant_(block.adaLN_modulation_x[-1].bias, 0)
            constant_(block.adaLN_modulation_c[-1].weight, 0)
            constant_(block.adaLN_modulation_c[-1].bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: Tensor):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = ops.transpose(x, (0, 5, 1, 3, 2, 4))
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        clip_emb: Tensor,
        clip_pooled_emb: Tensor,
        t5_emb: Optional[Tensor] = None,
    ):
        """
        Forward pass of MMDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        clip_emb: (N, L, D) tensor of text embeddings of 2 clips (concated), (N, 77, 2048)
        clip_pooled_emb: tensor of pooled text embeddings of 2 clips (concated)
        t5_emb: (N, L, D) tensor of t5 text embeddings, t5xxl (N, 77, 4096)
        """
        x = self.x_embedder(x)
        if not self.use_rel_pos:
            x = x + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        if t5_emb is None:
            text_emb = clip_emb  # L = 77
        else:
            # pad and concat if use t5 emb
            _, _, d_clip = clip_emb.shape
            _, _, d_t5 = t5_emb.shape
            clip_emb = ops.pad(clip_emb, (0, d_t5 - d_clip), mode="constant", value=0)
            text_emb = ops.cat([clip_emb, t5_emb], axis=1)  # L = 77+77 = 154

        c = self.text_embedding_projection(text_emb)  # (N, L, D)

        t = self.t_embedder(t)  # (N, D)
        timesteps_proj = self.timestep_projection(clip_pooled_emb)
        t = t + timesteps_proj

        for block in self.blocks:
            x, c = block(x, c, t, spatial_freq=self.spatial_freq)  # (N, T, D)

        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(
        self,
        x: Tensor,
        t: Tensor,
        clip_emb: Tensor,
        clip_pooled_emb: Tensor,
        t5_emb: Optional[Tensor] = None,
        cfg_scale: Union[float, Tensor] = 4.0,
    ):
        """
        Forward pass of MMDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.forward(combined, t, clip_emb, clip_pooled_emb, t5_emb)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)

    def construct(self, x, t, clip_emb, clip_pooled_emb, t5_emb=None, cfg_scale=None):
        if cfg_scale is None:
            return self.forward(x, t, clip_emb, clip_pooled_emb, t5_emb)
        else:
            return self.forward_with_cfg(x, t, clip_emb, clip_pooled_emb, t5_emb, cfg_scale)


# configs


def MMDiT_XL_2(**kwargs):
    return MMDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def MMDiT_XL_4(**kwargs):
    return MMDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def MMDiT_XL_8(**kwargs):
    return MMDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def MMDiT_L_2(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def MMDiT_L_4(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def MMDiT_L_8(**kwargs):
    return MMDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def MMDiT_B_2(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def MMDiT_B_4(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def MMDiT_B_8(**kwargs):
    return MMDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def MMDiT_S_2(**kwargs):
    return MMDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def MMDiT_S_4(**kwargs):
    return MMDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def MMDiT_S_8(**kwargs):
    return MMDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


MMDiT_models = {
    "MMDiT-XL/2": MMDiT_XL_2,
    "MMDiT-XL/4": MMDiT_XL_4,
    "MMDiT-XL/8": MMDiT_XL_8,
    "MMDiT-L/2": MMDiT_L_2,
    "MMDiT-L/4": MMDiT_L_4,
    "MMDiT-L/8": MMDiT_L_8,
    "MMDiT-B/2": MMDiT_B_2,
    "MMDiT-B/4": MMDiT_B_4,
    "MMDiT-B/8": MMDiT_B_8,
    "MMDiT-S/2": MMDiT_S_2,
    "MMDiT-S/4": MMDiT_S_4,
    "MMDiT-S/8": MMDiT_S_8,
}


if __name__ == "__main__":
    # ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
    ms.set_context(mode=ms.GRAPH_MODE)

    import numpy as np

    N, C, H, W = 2, 16, 64, 64
    x = Tensor(np.random.rand(N, C, H, W), dtype=ms.float32)
    t = Tensor(np.random.randint(low=0, high=100, size=(N,)), dtype=ms.int64)

    clip_emb = Tensor(np.random.rand(N, 77, 2048), dtype=ms.float32)
    clip_pooled_emb = Tensor(np.random.rand(N, 2048), dtype=ms.float32)
    t5_emb = Tensor(np.random.rand(N, 77, 4096), dtype=ms.float32)

    model = MMDiT_S_2(input_size=64, in_channels=C, block_kwargs={"enable_flash_attention": True})
    output = model(x, t, clip_emb, clip_pooled_emb, t5_emb)
    print(output.shape)
