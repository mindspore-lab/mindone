import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common import initializer as init

from .layers import DiTBlock, FinalLayer, LabelEmbedder, PatchEmbed, TimestepEmbedder, get_2d_sincos_pos_embed


class DiT(nn.Cell):
    """
    Diffusion model with a Transformer backbone.
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
        dtype=ms.float32,
        block_kwargs={},
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dtype = dtype

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, dtype=self.dtype)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=self.dtype)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, dtype=self.dtype)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = ms.Parameter(ops.zeros((1, num_patches, hidden_size), dtype=self.dtype), requires_grad=False)
        self.blocks = nn.CellList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dtype=self.dtype, **block_kwargs)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=self.dtype)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        ops.Assign()(self.pos_embed, Tensor(pos_embed, ms.float32).unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        ops.Assign()(w, init.initializer(init.XavierUniform(), w.shape, ms.float32).init_data())
        b = self.x_embedder.proj.bias
        ops.Assign()(b, init.initializer(init.Zero(), b.shape, ms.float32).init_data())

        # Initialize label embedding table:
        emb_table = self.y_embedder.embedding_table.embedding_table
        ops.Assign()(emb_table, init.initializer(init.Normal(sigma=0.02), emb_table.shape, ms.float32).init_data())

        # Initialize timestep embedding MLP:
        weight = self.t_embedder.mlp[0].weight
        ops.Assign()(weight, init.initializer(init.Normal(sigma=0.02), weight.shape, ms.float32).init_data())
        weight = self.t_embedder.mlp[2].weight
        ops.Assign()(weight, init.initializer(init.Normal(sigma=0.02), weight.shape, ms.float32).init_data())

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            w = block.adaLN_modulation[-1].weight
            b = block.adaLN_modulation[-1].bias
            ops.Assign()(w, init.initializer(init.Zero(), w.shape, ms.float32).init_data())
            ops.Assign()(b, init.initializer(init.Zero(), b.shape, ms.float32).init_data())

        # Zero-out output layers:
        for layer in [self.final_layer.adaLN_modulation[-1], self.final_layer.linear]:
            w = layer.weight
            b = layer.bias
            ops.Assign()(w, init.initializer(init.Zero(), w.shape, ms.float32).init_data())
            ops.Assign()(b, init.initializer(init.Zero(), b.shape, ms.float32).init_data())

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

    def construct(self, x, t, y):
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

    def construct_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)


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
