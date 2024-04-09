import logging
from typing import Any, Dict, Optional, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import mindspore as ms
from mindspore import Tensor, nn, ops

from .dit import FinalLayer, LabelEmbedder, TimestepEmbedder
from .fit import FiTBlock
from .modules.pos_embed import get_1d_sincos_pos_embed, precompute_freqs_cis_1d
from .utils import constant_, normal_, xavier_uniform_

logger = logging.getLogger(__name__)


class DynLatte(nn.Cell):
    """A diffusion Transform model for video generation
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
        condition (str, default=None): The type of conditions in [None, 'text', 'class']. If it is None, DynLatte is a un-conditional video generator.
            If it is 'text', it accepts text embeddings (B, T, D) as conditions, and generates videos. T: number of tokens. D: embedding dimension.
            If it is 'class', it accepts class labels (B, ) as conditions, and generates videos.
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        num_frames: int = 32,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        ffn: Literal["swiglu", "mlp"] = "swiglu",
        pos: Literal["rotate", "absolute"] = "absolute",
        block_kwargs: Dict[str, Any] = {},
        condition: Optional[str] = None,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.pos = pos

        if condition is not None:
            assert isinstance(condition, str), f"Expect that the condition type is a string, but got {type(condition)}"
            self.condition = condition.lower()
        else:
            self.condition = condition

        self.x_embedder = nn.Dense(self.in_channels * patch_size * patch_size, hidden_size, has_bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        assert self.condition in [None, "text", "class"], f"Unsupported condition type! {self.condition}"
        if self.condition == "class":
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)  # original dit param
        if self.condition == "text":
            self.text_embedding_projection = nn.SequentialCell(
                nn.SiLU(),
                nn.Dense(77 * 768, hidden_size, has_bias=True),
            )

        if self.pos == "absolute":
            self.pos_t = Tensor(get_1d_sincos_pos_embed(hidden_size, num_frames)[None, ...], dtype=ms.float32)
        else:
            self.pos_t = Tensor(
                precompute_freqs_cis_1d(hidden_size // num_heads, num_frames)[None, ...], dtype=ms.float32
            )

        self.blocks = nn.CellList(
            [
                FiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, ffn=ffn, pos=pos, k_norm=True, **block_kwargs)
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)
        # Initialize label embedding table:
        if self.condition == "class":
            normal_(self.y_embedder.embedding_table.embedding_table, std=0.02)

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

    def unpatchify(self, x: Tensor, bs: int, h: int, w: int) -> Tensor:
        c = self.out_channels
        nh, nw = h // self.patch_size, w // self.patch_size
        x = x.reshape((x.shape[0], nh, nw, self.patch_size, self.patch_size, c))
        x = x.permute((0, 5, 1, 3, 2, 4))
        imgs = x.reshape((bs, -1, c, nh * self.patch_size, nw * self.patch_size))
        return imgs

    def patchify(self, x: Tensor) -> Tensor:
        N, F, C, H, W = x.shape
        nh, nw = H // self.patch_size, W // self.patch_size
        x = ops.reshape(x, (-1, C, nh, self.patch_size, nw, self.patch_size))
        x = ops.transpose(x, (0, 2, 4, 3, 5, 1))  # N*F, nh, nw, patch, patch, C
        x = ops.reshape(x, (N * F, nh * nw, -1))
        return x

    def get_condition_embed(self, t_embed, y_embed=None, text_embed=None):
        # conditions can be (1) timestep embed, (2) class label embed, (3) text embed.
        if y_embed is None and text_embed is None:
            return t_embed
        elif y_embed is not None and text_embed is None:
            return t_embed + y_embed
        elif y_embed is None and text_embed is not None:
            return t_embed + text_embed
        else:
            raise ValueError("Incorrect embedding!")

    def spatial_to_temperoal(self, x: Tensor, bs: int, num_frames: int):
        _, num_patches, channels = x.shape
        x = ops.reshape(x, (bs, num_frames, num_patches, channels))
        x = ops.permute(x, (0, 2, 1, 3))
        x = ops.reshape(x, (bs * num_patches, num_frames, channels))
        return x

    def temporal_to_spatial(self, x: Tensor, bs: int, num_patches: int):
        _, num_frames, channels = x.shape
        x = ops.reshape(x, (bs, num_patches, num_frames, channels))
        x = ops.permute(x, (0, 2, 1, 3))
        x = ops.reshape(x, (bs * num_frames, num_patches, channels))
        return x

    def get_condition(
        self,
        num_frames: int,
        num_patches: int,
        bs: int,
        y: Optional[Tensor] = None,
        text_embed: Optional[Tensor] = None,
    ):
        if self.condition == "class":
            y = self.y_embedder(y, self.training)  # (N, D)
            # (N, D) -> (N*num_frames, D)
            y_spatial = y.repeat_interleave(repeats=num_frames, dim=0)
            # (N, D) -> (N*T, D)
            y_temp = y.repeat_interleave(repeats=num_patches, dim=0)
            text_embed_spatial, text_embed_temp = None, None
        elif self.condition == "text":
            text_embed = self.text_embedding_projection(text_embed.reshape(bs, -1))  # (N, L*D)
            # (N, D) -> (N*num_frames, D)
            text_embed_spatial = text_embed.repeat_interleave(repeats=num_frames, dim=0)
            # (N, D) -> (N*T, D)
            text_embed_temp = text_embed.repeat_interleave(repeats=num_patches, dim=0)
            y_spatial, y_temp = None, None
        else:
            y_spatial, y_temp = None, None
            text_embed_spatial, text_embed_temp = None, None
        return y_spatial, y_temp, text_embed_spatial, text_embed_temp

    @ms.jit
    def construct(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        mask_t: Tensor,
        mask_s: Tensor,
        y: Optional[Tensor] = None,
        text_embed: Optional[Tensor] = None,
    ):
        """
        Forward pass of DynLatte.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        pos: (N, T, D) tensor of positional embedding or precomputed cosine and sine frequencies
        mask: (N, T) tensor of valid mask
        y: (N,) tensor of class labels
        text_embed: (N, L, D), tensor of text embedding. L is the number of tokens, for CLIP it should be 77
        """
        bs, num_frames, _, h, w = x.shape
        x = self.patchify(x)
        _, num_patches, _ = x.shape

        # FIXME: drop cast when repeat_interleave support bool_
        if mask_s.shape[0] != 1:
            mask_s = mask_s.to(ms.int32).repeat_interleave(repeats=num_frames, dim=0).to(ms.bool_)
        if mask_t.shape[0] != 1:
            mask_t = mask_t.to(ms.int32).repeat_interleave(repeats=num_patches, dim=0).to(ms.bool_)

        if pos.shape[0] != 1:
            pos = pos.repeat_interleave(repeats=num_frames, dim=0)

        if self.pos == "absolute":
            x = self.x_embedder(x) + pos.to(x.dtype)  # (N, T, D), where T = H * W / patch_size ** 2
        else:
            x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        # (N, D) -> (N*num_frames, D)
        t_spatial = t.repeat_interleave(repeats=num_frames, dim=0)
        # (N, D) -> (N*T, D)
        t_temp = t.repeat_interleave(repeats=num_patches, dim=0)

        y_spatial, y_temp, text_embed_spatial, text_embed_temp = self.get_condition(
            num_frames, num_patches, bs, y, text_embed
        )

        if self.pos == "rotate":
            freqs_cis = pos
            freqs_cis_t = self.pos_t
        else:
            freqs_cis = None
            freqs_cis_t = None

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]
            c = self.get_condition_embed(t_spatial, y_spatial, text_embed_spatial)

            x = spatial_block(x, c, mask=mask_s, freqs_cis=freqs_cis)
            x = self.spatial_to_temperoal(x, bs, num_frames)

            # add time embed
            if i == 0 and self.pos == "absolute":
                x = x + self.pos_t.to(x.dtype)

            c = self.get_condition_embed(t_temp, y_temp, text_embed_temp)
            x = temp_block(x, c, mask=mask_t, freqs_cis=freqs_cis_t)
            x = self.temporal_to_spatial(x, bs, num_patches)

        c = self.get_condition_embed(t_spatial, y_spatial, None)
        x = self.final_layer(x, c)
        x = self.unpatchify(x, bs, h, w)  # (N, F, out_channels, H, W)
        return x

    @ms.jit
    def construct_with_cfg(
        self,
        x: Tensor,
        t: Tensor,
        pos: Tensor,
        mask_s: Tensor,
        mask_t: Tensor,
        y: Optional[Tensor] = None,
        text_embed: Optional[Tensor] = None,
        cfg_scale: Union[Tensor, float] = 4.0,
    ):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, t, pos, mask_s, mask_t, y=y, text_embed=text_embed)
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=2)


def DynLatte_XL_2(**kwargs):
    return DynLatte(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DynLatte_XL_4(**kwargs):
    return DynLatte(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DynLatte_XL_8(**kwargs):
    return DynLatte(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DynLatte_L_2(**kwargs):
    return DynLatte(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DynLatte_L_4(**kwargs):
    return DynLatte(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DynLatte_L_8(**kwargs):
    return DynLatte(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DynLatte_B_2(**kwargs):
    return DynLatte(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DynLatte_B_4(**kwargs):
    return DynLatte(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DynLatte_B_8(**kwargs):
    return DynLatte(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DynLatte_S_2(**kwargs):
    return DynLatte(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DynLatte_S_4(**kwargs):
    return DynLatte(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DynLatte_S_8(**kwargs):
    return DynLatte(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DynLatte_models = {
    "DynLatte-XL/2": DynLatte_XL_2,
    "DynLatte-XL/4": DynLatte_XL_4,
    "DynLatte-XL/8": DynLatte_XL_8,
    "DynLatte-L/2": DynLatte_L_2,
    "DynLatte-L/4": DynLatte_L_4,
    "DynLatte-L/8": DynLatte_L_8,
    "DynLatte-B/2": DynLatte_B_2,
    "DynLatte-B/4": DynLatte_B_4,
    "DynLatte-B/8": DynLatte_B_8,
    "DynLatte-S/2": DynLatte_S_2,
    "DynLatte-S/4": DynLatte_S_4,
    "DynLatte-S/8": DynLatte_S_8,
}
