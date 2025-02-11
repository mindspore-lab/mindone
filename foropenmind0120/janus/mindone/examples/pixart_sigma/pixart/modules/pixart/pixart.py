from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.common.initializer import XavierUniform, initializer

from mindone.models.utils import constant_, normal_, xavier_uniform_

from ._layers import GELU, LayerNorm
from .blocks import (
    CaptionEmbedder,
    CrossAttention,
    DropPath,
    KVCompressSelfAttention,
    Mlp,
    PatchEmbed,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    t2i_modulate,
)
from .pos import cal_2d_sincos_pos_embed, cal_omega


class PixArtBlock(nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        sampling: Literal[None, "conv", "ave", "uniform"] = None,
        sr_ratio: int = 1,
        qk_norm: bool = False,
        **block_kwargs: Any,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, epsilon=1e-6)
        self.attn = KVCompressSelfAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            sampling=sampling,
            sr_ratio=sr_ratio,
            qk_norm=qk_norm,
            **block_kwargs,
        )
        self.cross_attn = CrossAttention(hidden_size, num_heads, **block_kwargs)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, epsilon=1e-6)
        approx_gelu = lambda: GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = Parameter(ops.randn(6, hidden_size) / hidden_size**0.5)

    def construct(
        self, x: Tensor, y: Tensor, t: Tensor, mask_y: Tensor, HW: Optional[Tuple[int, int]] = None
    ) -> Tensor:
        B = x.shape[0]

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mint.chunk(
            self.scale_shift_table[None] + t.reshape(B, 6, -1), 6, dim=1
        )
        x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa), HW=HW))
        x = x + self.cross_attn(x, y, mask=mask_y)
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))
        return x


class PixArt(nn.Cell):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        pred_sigma: bool = True,
        drop_path: float = 0.0,
        caption_channels: int = 4096,
        pe_interpolation: float = 1.0,
        model_max_length: int = 120,
        qk_norm: bool = False,
        sampling: Literal[None, "conv", "ave", "uniform"] = None,
        scale_factor: int = 1,
        kv_compress_layer: Optional[List[int]] = None,
        recompute: bool = False,
        block_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.hidden_size = hidden_size

        self.kv_compress_layer: List[int] = list() if kv_compress_layer is None else kv_compress_layer
        self.block_kwargs: Dict[str, Any] = dict() if block_kwargs is None else block_kwargs

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        base_size = input_size // self.patch_size
        pos_embed = cal_2d_sincos_pos_embed(
            hidden_size, int(self.x_embedder.num_patches**0.5), scale=self.pe_interpolation, base_size=base_size
        )
        self.pos_embed = pos_embed[None, ...]

        approx_gelu = lambda: GELU(approximate="tanh")
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        drop_path = np.linspace(0, drop_path, depth).tolist()  # stochastic depth decay rule
        self.blocks = nn.CellList(
            [
                PixArtBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    sampling=sampling,
                    sr_ratio=int(scale_factor) if i in self.kv_compress_layer else 1,
                    qk_norm=qk_norm,
                    **self.block_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)

        self.initialize_weights()

        # TODO: somehow self.blocks.recompute() doesn't work
        if recompute:
            for blk in self.blocks:
                blk.recompute()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)
        normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            constant_(block.cross_attn.proj.weight, 0)
            constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        nh, nw = h // self.patch_size, w // self.patch_size
        x = x.reshape((x.shape[0], nh, nw, self.patch_size, self.patch_size, c))
        x = ops.transpose(x, (0, 5, 1, 3, 2, 4))
        imgs = x.reshape((x.shape[0], c, nh * self.patch_size, nw * self.patch_size))
        return imgs

    def construct(self, x: Tensor, t: Tensor, y: Tensor, mask_y: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, L, C') tensor of text embeddings
        mask_y: (N, L) tensor of text mask
        """
        _, _, h, w = x.shape
        x = self.x_embedder(x) + self.pos_embed.to(x.dtype)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)

        for block in self.blocks:
            x = block(x, y, t0, mask_y=mask_y)
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, h, w)  # (N, out_channels, H, W)
        return x

    @ms.jit
    def construct_with_cfg(
        self, x: Tensor, t: Tensor, y: Tensor, cfg_scale: Union[Tensor, float], mask_y: Optional[Tensor] = None
    ) -> Tensor:
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, t, y, mask_y=mask_y)
        eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
        cond_eps, uncond_eps = mint.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0).to(rest.dtype)
        return ops.cat([eps, rest], axis=1)

    @ms.jit
    def construct_with_dpmsolver(
        self,
        x: Tensor,
        t: Tensor,
        y: Tensor,
        mask_y: Optional[Tensor] = None,
    ) -> Tensor:
        model_out = self.construct(x, t, y, mask_y=mask_y)
        return mint.chunk(model_out, 2, dim=1)[0]


class PixArtMS(PixArt):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        pred_sigma: bool = True,
        drop_path: float = 0.0,
        caption_channels: int = 4096,
        pe_interpolation: float = 1.0,
        model_max_length: int = 120,
        micro_condition: bool = False,
        qk_norm: bool = False,
        sampling: Literal[None, "conv", "ave", "uniform"] = None,
        scale_factor: int = 1,
        kv_compress_layer: Optional[List[int]] = None,
        recompute: bool = False,
        block_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super(PixArt, self).__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pe_interpolation = pe_interpolation
        self.hidden_size = hidden_size
        self.base_size = input_size // self.patch_size

        self.kv_compress_layer: List[int] = list() if kv_compress_layer is None else kv_compress_layer
        self.block_kwargs: Dict[str, Any] = dict() if block_kwargs is None else block_kwargs

        self.x_embedder = PatchEmbed(None, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        approx_gelu = lambda: GELU(approximate="tanh")
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )
        self.micro_conditioning = micro_condition
        if self.micro_conditioning:
            self.csize_embedder = SizeEmbedder(hidden_size // 3)
            self.ar_embedder = SizeEmbedder(hidden_size // 3)

        drop_path = np.linspace(0, drop_path, depth).tolist()  # stochastic depth decay rule
        self.blocks = nn.CellList(
            [
                PixArtBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i],
                    sampling=sampling,
                    sr_ratio=int(scale_factor) if i in self.kv_compress_layer else 1,
                    qk_norm=qk_norm,
                    **self.block_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, patch_size, self.out_channels)
        self.omega = cal_omega(hidden_size // 2)

        self.initialize_weights()

        # TODO: somehow self.blocks.recompute() doesn't work
        if recompute:
            for blk in self.blocks:
                blk.recompute()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)
        normal_(self.t_block[1].weight, std=0.02)
        if self.micro_conditioning:
            normal_(self.csize_embedder.mlp[0].weight, std=0.02)
            normal_(self.csize_embedder.mlp[2].weight, std=0.02)
            normal_(self.ar_embedder.mlp[0].weight, std=0.02)
            normal_(self.ar_embedder.mlp[2].weight, std=0.02)

        # Initialize caption embedding MLP:
        normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            constant_(block.cross_attn.proj.weight, 0)
            constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)

    def _cal_pos_embed(self, x: Tensor) -> Tensor:
        _, _, h, w = x.shape

        nh = h // self.patch_size
        nw = w // self.patch_size
        pos_emb = cal_2d_sincos_pos_embed(
            self.hidden_size, nh, nw=nw, scale=self.pe_interpolation, base_size=self.base_size, omega=self.omega
        )
        return ops.stop_gradient(pos_emb)

    def construct(self, x: Tensor, t: Tensor, y: Tensor, mask_y: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, L, C') tensor of text embeddings
        mask_y: (N, L) tensor of text mask
        """
        _, _, h, w = x.shape
        pos_embed = self._cal_pos_embed(x)

        x = self.x_embedder(x) + pos_embed.to(x.dtype)  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)

        if self.micro_conditioning:
            csize = Tensor([[h * 8, w * 8]], dtype=x.dtype)
            csize = self.csize_embedder(csize)  # (1, D)
            ar = Tensor([[h / w]], dtype=x.dtype)
            ar = self.ar_embedder(ar)  # (1, D)
            t = t + ops.concat([csize, ar], axis=1)

        t0 = self.t_block(t)
        y = self.y_embedder(y)  # (N, L, D)

        for block in self.blocks:
            x = block(x, y, t0, mask_y=mask_y, HW=(h // self.patch_size, w // self.patch_size))
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, h, w)  # (N, out_channels, H, W)
        return x


def PixArt_XL_2(**kwargs):
    return PixArt(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def PixArtMS_XL_2(**kwargs):
    return PixArtMS(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)
