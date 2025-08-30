import math
from typing import Optional, Tuple, Type, Union

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn, ops
from mindspore.common.initializer import XavierUniform, Zero, initializer

from mindone.models.modules.flash_attention import FLASH_IS_AVAILABLE, MSFlashAttention

from .modules import get_2d_sincos_pos_embed
from .utils import constant_, exists, modulate, normal_, xavier_uniform_

__all__ = [
    "DiT",
    "DiT_models",
    "DiT_XL_2",
    "DiT_XL_4",
    "DiT_XL_8",
    "DiT_L_2",
    "DiT_L_4",
    "DiT_L_8",
    "DiT_B_2",
    "DiT_B_4",
    "DiT_B_8",
    "DiT_S_2",
    "DiT_S_4",
    "DiT_S_8",
]


class GELU(mint.nn.GELU):
    def __init__(self, approximate: str = "none"):
        if approximate == "none":
            super().__init__(False)
        elif approximate == "tanh":
            super().__init__()
        else:
            raise ValueError(f"approximate must be one of ['none', 'tanh'], but got {approximate}.")


class PatchEmbed(nn.Cell):
    """Image to Patch Embedding

    Args:
        image_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(
        self,
        image_size: Optional[int] = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size: Tuple = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        if image_size is not None:
            self.image_size: Optional[Tuple] = (image_size, image_size) if isinstance(image_size, int) else image_size
            self.patches_resolution: Optional[Tuple] = tuple([s // p for s, p in zip(self.image_size, self.patch_size)])
            self.num_patches: Optional[int] = self.patches_resolution[0] * self.patches_resolution[1]
        else:
            self.image_size: Optional[Tuple] = None
            self.patches_resolution: Optional[Tuple] = None
            self.num_patches: Optional[int] = None
        self.embed_dim = embed_dim
        self.proj = mint.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def construct(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if self.image_size is not None:
            assert (h, w) == (
                self.image_size[0],
                self.image_size[1],
            ), f"Input height and width ({h},{w}) doesn't match model ({self.image_size[0]},{self.image_size[1]})."
        x = self.proj(x)
        x = mint.reshape(x, (b, self.embed_dim, -1))
        x = mint.permute(x, (0, 2, 1))  # B Ph*Pw C
        return x


class LinearPatchEmbed(nn.Cell):
    """Image to Patch Embedding: using a linear layer instead of conv2d layer for projection

    Args:
        image_size (int): Image size. Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(
        self,
        image_size: Optional[int] = 224,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size: Tuple = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        if image_size is not None:
            self.image_size: Optional[Tuple] = (image_size, image_size) if isinstance(image_size, int) else image_size
            self.patches_resolution: Optional[Tuple] = tuple([s // p for s, p in zip(self.image_size, self.patch_size)])
            self.num_patches: Optional[int] = self.patches_resolution[0] * self.patches_resolution[1]
        else:
            self.image_size: Optional[Tuple] = None
            self.patches_resolution: Optional[Tuple] = None
            self.num_patches: Optional[int] = None
        self.embed_dim = embed_dim
        self.proj = mint.nn.Linear(patch_size * patch_size * in_chans, embed_dim, bias=bias)

    def construct(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        if self.image_size is not None:
            assert (h, w) == (
                self.image_size[0],
                self.image_size[1],
            ), f"Input height and width ({h},{w}) doesn't match model ({self.image_size[0]},{self.image_size[1]})."
        ph, pw = h // self.patch_size[0], w // self.patch_size[1]
        x = x.reshape((b, c, self.patch_size[0], ph, self.patch_size[1], pw))  # (B, C, P, Ph, P, Pw)
        x = x.transpose((0, 3, 5, 2, 4, 1))  # (B, Ph, Pw, P, P, C)
        x = x.reshape((b, ph * pw, self.patch_size[0] * self.patch_size[1] * c))  # (B, Ph*Pw, P*P*C)

        x = self.proj(x)  # B Ph*Pw C_out
        return x


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Cell] = mint.nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = mint.nn.Linear(in_features=in_features, out_features=hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = mint.nn.Linear(in_features=hidden_features, out_features=out_features, bias=True)
        self.drop = mint.nn.Dropout(p=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Cell):
    def __init__(self, dim_head, attn_drop=0.0, upcast_softmax=True):
        super().__init__()
        self.scale = dim_head**-0.5
        self.attn_drop = mint.nn.Dropout(p=attn_drop)
        self.upcast_softmax = upcast_softmax

    def construct(self, q, k, v, mask=None):
        """
        q: (b*h n_q d), h - num_head, n_q - seq_len of q
        k v: (b*h n_k d), h - num_head, n_k - seq_len of k;
        mask: (b*h n_q n_k), -inf means to discard, 0 means to keep.
        """
        sim = mint.matmul(q, mint.permute(k, (0, 2, 1))) * self.scale
        if self.upcast_softmax:
            sim = sim.astype(ms.float32)
        if exists(mask):
            sim += mask

        # use fp32 for exponential inside
        attn = mint.nn.functional.softmax(sim, dim=-1).astype(v.dtype)
        attn = self.attn_drop(attn)

        out = mint.matmul(attn, v)

        return out


class SelfAttention(nn.Cell):
    """Attention adopted from :
    https://github.com/pprp/timm/blob/master/timm/models/vision_transformer.py
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
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = mint.nn.Linear(dim, dim * 3, bias=qkv_bias, weight_init=XavierUniform(), bias_init=Zero()).to_float(
            self.dtype
        )
        self.proj = mint.nn.Linear(dim, dim, weight_init=XavierUniform(), bias_init=Zero()).to_float(self.dtype)
        self.proj_drop = mint.nn.Dropout(p=proj_drop)

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

    @staticmethod
    def _rearange_in(x, h):
        # (b, n, h*d) -> (b*h, n, d)
        b, n, d = x.shape
        d = d // h

        x = mint.reshape(x, (b, n, h, d))
        x = mint.permute(x, (0, 2, 1, 3))
        x = mint.reshape(x, (b * h, n, d))
        return x

    @staticmethod
    def _rearange_out(x, h):
        # (b*h, n, d) -> (b, n, h*d)
        b, n, d = x.shape
        b = b // h

        x = mint.reshape(x, (b, h, n, d))
        x = mint.permute(x, (0, 2, 1, 3))
        x = mint.reshape(x, (b, n, h * d))
        return x

    def construct(self, x, mask=None):
        """
        x: (b, seq_len, c)
        mask: (b n_k) or (b 1 n_k) or (b n_q n_k). 1 means to keep. 0 means to discard (mask).
        """
        x_dtype = x.dtype
        h = self.num_heads
        B, N, C = x.shape
        # (b, n, 3*h*d) -> (b, n, 3, h*d)  -> (3, b, n, h*d)
        qkv = self.qkv(x).reshape(B, N, 3, -1).permute((2, 0, 1, 3))
        q, k, v = qkv.unbind(0)
        q_b, q_n, _ = q.shape  # (b n h*d)
        k_b, k_n, _ = k.shape
        v_b, v_n, _ = v.shape

        head_dim = q.shape[-1] // h
        # convert sequence mask to attention mask: (b, q_n) to (b, q_n, k_n)
        if mask is not None:
            if mask.ndim == 2:
                # mask shape is (batch_size, key_len)
                mask = mint.unsqueeze(mask, dim=1).repeat_interleave(
                    q_n, dim=1
                )  # (b, k_n) -> (b, 1, k_n) -> (b, q_n, k_n)
                mask = mint.select(
                    ~mask,
                    mint.ones((q_b, q_n, k_n), self.dtype) * (-ms.numpy.inf),
                    mint.zeros((q_b, q_n, k_n), self.dtype),
                )
            elif mask.ndim == 3:
                # mask shape is (batch_size, query_len, key_len), the query_len maybe one
                if mask.shape[-2] == 1:
                    mask = mask.repeat_interleave(
                        q_n, dim=-1
                    )  # manually broadcast to key length to avoid FA shape error
                assert mask.shape[-2] == q_n, "Expect mask shape to be (bs, query_len, key_len), "
                f"but the mask query length {mask.shape[-2]} is different from the input query length {q_n}"
                assert mask.shape[-1] == k_n, "Expect mask shape to be (bs, query_len, key_len), "
                f"but the mask key length {mask.shape[-1]} is different from the input key length {k_n}."
            else:
                raise ValueError(f"mask should be 2D or 3D, but got {mask.ndim}D")

        if (
            self.enable_flash_attention and q_n % 16 == 0 and k_n % 16 == 0 and head_dim <= 256
        ):  # TODO: why restrict head_dim?
            # reshape qkv shape ((b n h*d) -> (b h n d))and mask dtype for FA input format
            q = q.view(q_b, q_n, h, -1).transpose(0, 2, 1, 3)
            k = k.view(k_b, k_n, h, -1).transpose(0, 2, 1, 3)
            v = v.view(v_b, v_n, h, -1).transpose(0, 2, 1, 3)
            if mask is not None and mask.ndim == 3:
                mask = mint.unsqueeze(mask, dim=1)  # (q_b, 1, q_n, k_n), FA needs 4-dim mask
            out = self.flash_attention(q, k, v, mask)
            b, h, n, d = out.shape
            # reshape FA output to original attn input format, (b h n d) -> (b n h*d)
            out = out.transpose(0, 2, 1, 3).view(b, n, -1)
        else:
            # (b, n, h*d) -> (b*h, n, d)
            q = self._rearange_in(q, h)
            k = self._rearange_in(k, h)
            v = self._rearange_in(v, h)
            if mask is not None and mask.shape[0] != q.shape[0]:
                mask = mask.repeat_interleave(h, dim=0)  # (b, q_n, k_n) -> (b*h, q_n, k_n)
            out = self.attention(q, k, v, mask)
            # (b*h, n, d) -> (b, n, h*d)
            out = self._rearange_out(out, h)

        return self.proj_drop(self.proj(out)).to(x_dtype)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Cell):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            mint.nn.SiLU(),
            mint.nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = mint.exp(-math.log(max_period) * mint.arange(start=0, end=half, dtype=ms.float32) / half)
        args = t[:, None].float() * freqs[None]
        embedding = mint.cat([mint.cos(args), mint.sin(args)], dim=-1)
        if dim % 2:
            embedding = mint.cat([embedding, mint.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def construct(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Cell):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = mint.nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = ops.rand(labels.shape[0]) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = mint.where(drop_ids, self.num_classes, labels)
        return labels

    def construct(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Cell):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.SequentialCell(
            mint.nn.SiLU(), mint.nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def construct(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, axis=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Cell):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = mint.nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.SequentialCell(
            mint.nn.SiLU(), mint.nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def construct(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Cell):
    """A diffusion model with a Transformer backbone.
    Args:
        input_size (int, default=32): The size of the input latent.
        patch_size (int, default=2): The size of each patch in the input latent. The input latent is divided into patches of patch_size x patch_size.
        in_channels (int, default=4): The number of input channels in the input latent.
        hidden_size (int, default=1152): The hidden size of the Transformer model.
        depth (int, default=28): The number of blocks in this Transformer.
        num_heads (int, default=16): The number of attention heads.
        mlp_ratio (float, default=4.0): The expansion ratio for the hidden dimension in the MLP of the Transformer.
        class_dropout_prob (float, default=0.1): The dropout probability for the class labels in the label embedder.
        num_classes (int, default=1000): The number of classes of the input labels.
        learn_sigma (bool, default=True): Whether to learn the diffusion model's sigma parameter.
        block_kwargs (dict, default={}): Additional keyword arguments for the Transformer blocks. for example, {'enable_flash_attention':True}
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        block_kwargs={},
        patch_embedder="conv",
        use_recompute=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.patch_embedder = patch_embedder
        self.use_recompute = use_recompute
        if patch_embedder == "conv":
            PatchEmbedder = PatchEmbed
        else:
            PatchEmbedder = LinearPatchEmbed
        self.x_embedder = PatchEmbedder(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = Parameter(mint.zeros((1, num_patches, hidden_size)), requires_grad=False)

        self.blocks = nn.CellList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, **block_kwargs) for _ in range(depth)]
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
            if isinstance(module, mint.nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.set_data(Tensor(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        # xavier_uniform_(w.view(w.shape[0], -1))
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).init_data().reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            constant_(block.adaLN_modulation[-1].weight, 0)
            constant_(block.adaLN_modulation[-1].bias, 0)

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
        x = mint.permute(x, (0, 5, 1, 3, 2, 4))
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

    def construct(self, x: Tensor, t: Tensor, y: Tensor):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    @ms.jit
    def construct_with_cfg(self, x: Tensor, t: Tensor, y: Tensor, cfg_scale: Union[float, Tensor]):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = mint.cat([half, half], dim=0)
        model_out = self.construct(combined, t, y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = mint.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = mint.cat([half_eps, half_eps], dim=0)
        return mint.cat([eps, rest], dim=1)


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}


if __name__ == "__main__":
    ms.set_context(mode=ms.GRAPH_MODE)
    model = DiT_S_2(input_size=32, block_kwargs={"enable_flash_attention": True})
    print(model(ops.randn(2, 4, 32, 32), ops.randint(0, 50, (2,)), mint.arange(2)))
