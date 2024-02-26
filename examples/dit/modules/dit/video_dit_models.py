import logging

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common import initializer as init

logger = logging.getLogger(__name__)
from .layers import (
    DiTBlock,
    FinalLayer,
    LabelEmbedder,
    PatchEmbed,
    TimestepEmbedder,
    get_1d_sincos_temp_embed,
    get_2d_sincos_pos_embed,
)


class DiT_Factorised_Encoder(nn.Cell):
    """
    Diffusion model with a Transformer backbon for video generation based on DiT
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=56,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        dtype=ms.float32,
        block_kwargs={},
        condition="class",
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.dtype = dtype
        if condition is not None:
            assert isinstance(condition, str), f"Expect that the condition type is a string, but got {type(condition)}"
            self.condition = condition.lower()
        else:
            self.condition = condition

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True, dtype=self.dtype)
        self.t_embedder = TimestepEmbedder(hidden_size, dtype=self.dtype)
        assert self.condition in [None, "text", "class"], f"Unsupported condition type! {self.condition}"
        if self.condition == "class":
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob, dtype=self.dtype)
        elif self.condition == "text":
            self.text_embedder = nn.SequentialCell(
                nn.SiLU().to_float(self.dtype),
                nn.Dense(77 * 768, hidden_size, has_bias=True).to_float(self.dtype),
            )
        num_patches = self.x_embedder.num_patches
        # TODO: Text Embedding Projection
        # Will use fixed sin-cos embedding:
        self.pos_embed = ms.Parameter(ops.zeros((1, num_patches, hidden_size), dtype=ms.float32), requires_grad=False)
        self.temp_embed = ms.Parameter(ops.zeros((1, num_frames, hidden_size), dtype=ms.float32), requires_grad=False)

        self.blocks = nn.CellList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dtype=self.dtype, **block_kwargs)
                for _ in range(depth // 2)
            ]
        )
        self.temp_blocks = nn.CellList(
            [
                DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, dtype=self.dtype, **block_kwargs)
                for _ in range(depth // 2)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels, dtype=self.dtype)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        ops.Assign()(self.pos_embed, Tensor(pos_embed, ms.float32).unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        ops.Assign()(self.temp_embed, Tensor(temp_embed, ms.float32).unsqueeze(0))

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

        # Zero-out adaLN modulation layers in DiT blocks (spatial):
        for block in self.blocks:
            w = block.adaLN_modulation[-1].weight
            b = block.adaLN_modulation[-1].bias
            ops.Assign()(w, init.initializer(init.Zero(), w.shape, ms.float32).init_data())
            ops.Assign()(b, init.initializer(init.Zero(), b.shape, ms.float32).init_data())
        # Zero-out adaLN modulation layers in DiT blocks (temporal):
        for block in self.temp_blocks:
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
        Forward pass of DiT.
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

        if y is not None:
            assert (
                self.condition == "class"
            ), f"When input y is not None, expect that the condition type equals to `class`, but got {self.condition}"
            y = self.y_embedder(y, self.training)  # (N, D)
            # (N, D) -> (N*num_frames, D)
            y_spatial = y.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
            # (N, D) -> (N*T, D)
            y_temp = y.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)
        else:
            y_spatial, y_temp = None, None

        # text embedding
        if text_embed is not None:
            assert (
                self.condition == "text"
            ), f"When input y is not None, expect that the condition type equals to `class`, but got {self.condition}"
            text_embed = self.text_embedder(text_embed.reshape(bs, -1))  # (N, L*D)
            # (N, D) -> (N*num_frames, D)
            text_embed_spatial = text_embed.repeat_interleave(repeats=self.temp_embed.shape[1], dim=0)
            # (N, D) -> (N*T, D)
            text_embed_temp = text_embed.repeat_interleave(repeats=self.pos_embed.shape[1], dim=0)
        else:
            text_embed_spatial, text_embed_temp = None, None

        # spatial encoder
        c = self.get_condition_embed(t_spatial, y_spatial, text_embed_spatial)
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)

        # add time embed
        _, num_patches, channels = x.shape
        x = (
            x.reshape((bs, num_frames, num_patches, channels))
            .permute((0, 2, 1, 3))
            .reshape((bs * num_patches, num_frames, channels))
        )
        x = x + self.temp_embed

        # temporal encoder
        c = self.get_condition_embed(t_temp, y_temp, text_embed_temp)
        for block in self.temp_blocks:
            x = block(x, c)  # (B*T, F, D)
        _, num_frames, channels = x.shape
        x = (
            x.reshape((bs, num_patches, num_frames, channels))
            .permute((0, 2, 1, 3))
            .reshape((bs * num_frames, num_patches, channels))
        )

        c = self.get_condition_embed(t_spatial, y_spatial, None)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
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
        combined = ops.cat([half, half], axis=0)
        model_out = self.construct(combined, t, y=y, text_embed=text_embed)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :, :3], model_out[:, :, 3:]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=2)

    def load_params_from_dit_ckpt(self, dit_ckpt):
        logger.info(f"Loading {dit_ckpt} params into VideoDiT model...")
        param_dict = ms.load_checkpoint(dit_ckpt)
        param_not_load, ckpt_not_load = ms.load_param_into_net(
            self,
            param_dict,
        )
        assert len(ckpt_not_load) == 0, f"ckpt params should be all loaded, but found {ckpt_not_load} are not loaded."


def VideoDiT_XL_2(**kwargs):
    return DiT_Factorised_Encoder(depth=56, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


VideoDiT_models = {
    "DiT-XL/2": VideoDiT_XL_2,
}
