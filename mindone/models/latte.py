import logging
import os

import mindspore as ms
from mindspore import Tensor, nn, ops, mint
from mindspore.common.initializer import XavierUniform, initializer

from .dit import DiTBlock, FinalLayer, LabelEmbedder, LinearPatchEmbed, PatchEmbed, TimestepEmbedder
from .modules import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from .utils import constant_, normal_, xavier_uniform_

logger = logging.getLogger(__name__)


class Latte(nn.Cell):
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
        condition (str, default=None): The type of conditions in [None, 'text', 'class']. If it is None, Latte is a un-conditional video generator.
            If it is 'text', it accepts text embeddings (B, T, D) as conditions, and generates videos. T: number of tokens. D: embedding dimension.
            If it is 'class', it accepts class labels (B, ) as conditions, and generates videos.
        use_recompute (bool, default=False): Whether to use recompute for transformer blocks. Recompute can save some memory while slowing down the process.
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
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        block_kwargs={},
        condition=None,
        patch_embedder="conv",
        use_recompute=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.use_recompute = use_recompute
        self.patch_embedder = patch_embedder

        if condition is not None:
            assert isinstance(condition, str), f"Expect that the condition type is a string, but got {type(condition)}"
            self.condition = condition.lower()
        else:
            self.condition = condition
        if patch_embedder == "conv":
            PatchEmbedder = PatchEmbed
        else:
            PatchEmbedder = LinearPatchEmbed
        self.x_embedder = PatchEmbedder(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        assert self.condition in [None, "text", "class"], f"Unsupported condition type! {self.condition}"
        if self.condition == "class":
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)  # original dit param
        if self.condition == "text":
            self.text_embedding_projection = nn.SequentialCell(
                nn.SiLU(),
                nn.Dense(77 * 768, hidden_size, has_bias=True),
            )
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = ms.Parameter(mint.zeros((1, num_patches, hidden_size), dtype=ms.float32), requires_grad=False)
        self.temp_embed = ms.Parameter(mint.zeros((1, num_frames, hidden_size), dtype=ms.float32), requires_grad=False)

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
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed (temp_embed) by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.set_data(Tensor(pos_embed).float().unsqueeze(0))
        temp_embed = get_1d_sincos_pos_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.set_data(Tensor(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        # xavier_uniform_(w.view(w.shape[0], -1))
        w_flatted = w.view(w.shape[0], -1)
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))
        constant_(self.x_embedder.proj.bias, 0)

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

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape((x.shape[0], h, w, p, p, c))
        x = x.permute((0, 5, 1, 3, 2, 4))
        imgs = x.reshape((x.shape[0], c, h * p, h * p))
        return imgs

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

    def construct(self, x, t, y=None, text_embed=None):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        text_embed: (N, L, D), tensor of text embedding. L is the number of tokens, for CLIP it should be 77
        """
        bs, num_frames, channels, height, width = x.shape
        # (b c f h w) -> (b*f c h w)
        x = x.reshape((bs * num_frames, channels, height, width))
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        # time embeddings
        t = self.t_embedder(t)  # (N, D)
        # (N, D) -> (N*num_frames, D)
        t_spatial = t.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
        # (N, D) -> (N*T, D)
        t_temp = t.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)

        if self.condition == "class":
            y = self.y_embedder(y, self.training)  # (N, D)
            # (N, D) -> (N*num_frames, D)
            y_spatial = y.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
            # (N, D) -> (N*T, D)
            y_temp = y.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)
            text_embed_spatial, text_embed_temp = None, None
        elif self.condition == "text":
            text_embed = self.text_embedding_projection(text_embed.reshape(bs, -1))  # (N, L*D)
            # (N, D) -> (N*num_frames, D)
            text_embed_spatial = text_embed.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
            # (N, D) -> (N*T, D)
            text_embed_temp = text_embed.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)
            y_spatial, y_temp = None, None
        else:
            y_spatial, y_temp = None, None
            text_embed_spatial, text_embed_temp = None, None

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i : i + 2]
            c = self.get_condition_embed(t_spatial, y_spatial, text_embed_spatial)
            x = spatial_block(x, c)
            _, num_patches, channels = x.shape
            x = (
                x.reshape((bs, num_frames, num_patches, channels))
                .permute((0, 2, 1, 3))
                .reshape((bs * num_patches, num_frames, channels))
            )
            # add time embed
            if i == 0:
                x = x + self.temp_embed

            c = self.get_condition_embed(t_temp, y_temp, text_embed_temp)
            x = temp_block(x, c)

            _, num_frames, channels = x.shape
            x = (
                x.reshape((bs, num_patches, num_frames, channels))
                .permute((0, 2, 1, 3))
                .reshape((bs * num_frames, num_patches, channels))
            )

        c = self.get_condition_embed(t_spatial, y_spatial, None)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        _, channels, h, w = x.shape
        x = x.reshape((bs, num_frames, channels, h, w))
        return x

    def construct_with_cfg(self, x, t, y=None, text_embed=None, cfg_scale=4.0):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = mint.cat([half, half], dim=0)
        model_out = self.construct(combined, t, y=y, text_embed=text_embed)
        eps, rest = model_out[:, :, : self.in_channels], model_out[:, :, self.in_channels :]
        cond_eps, uncond_eps = mint.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = mint.cat([half_eps, half_eps], dim=0)
        return mint.cat([eps, rest], dim=2)

    def load_params_from_ckpt(self, ckpt):
        # load param from a ckpt file path or a parameter dictionary
        if isinstance(ckpt, str):
            assert os.path.exists(ckpt), f"{ckpt} does not exist!"
            logger.info(f"Loading {ckpt} params into Latte model...")
            param_dict = ms.load_checkpoint(ckpt)
        elif isinstance(ckpt, dict):
            param_dict = ckpt
        else:
            raise ValueError("Expect to receive a ckpt path or parameter dictionary as input!")
        _, ckpt_not_load = ms.load_param_into_net(
            self,
            param_dict,
        )
        assert len(ckpt_not_load) == 0, f"ckpt params should be all loaded, but found {ckpt_not_load} are not loaded."


#################################################################################
#                                   Latte Configs                                  #
#################################################################################


def Latte_XL_2(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def Latte_XL_4(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def Latte_XL_8(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def Latte_L_2(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def Latte_L_4(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def Latte_L_8(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def Latte_B_2(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def Latte_B_4(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def Latte_B_8(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def Latte_S_2(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def Latte_S_4(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def Latte_S_8(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Latte_models = {
    "Latte-XL/2": Latte_XL_2,
    "Latte-XL/4": Latte_XL_4,
    "Latte-XL/8": Latte_XL_8,
    "Latte-L/2": Latte_L_2,
    "Latte-L/4": Latte_L_4,
    "Latte-L/8": Latte_L_8,
    "Latte-B/2": Latte_B_2,
    "Latte-B/4": Latte_B_4,
    "Latte-B/8": Latte_B_8,
    "Latte-S/2": Latte_S_2,
    "Latte-S/4": Latte_S_4,
    "Latte-S/8": Latte_S_8,
}
