from typing import Any

from tqdm import tqdm

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Constant, Normal, XavierUniform, initializer

from mindone.diffusers._peft.utils import ModulesToSaveWrapper, _get_submodules
from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models import ModelMixin
from mindone.transformers.integrations import PeftAdapterMixin

from .attn_layers import Attention, CrossAttention, FlashCrossMHAModified, FlashSelfMHAModified
from .embedders import PatchEmbed, TimestepEmbedder, timestep_embedding
from .layers import Mlp
from .norm_layers import LayerNorm, RMSNorm
from .poolers import AttentionPool


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class FP32_Layernorm(LayerNorm):
    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        origin_dtype = inputs.dtype
        x = self.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps)
        return x.to(origin_dtype)


class FP32_SiLU(nn.SiLU):
    def construct(self, inputs: ms.Tensor) -> ms.Tensor:
        return ops.silu(inputs.float()).to(inputs.dtype)


class HunYuanDiTBlock(nn.Cell):
    """
    A HunYuanDiT block with `add` conditioning.
    """

    def __init__(
        self,
        hidden_size,
        c_emb_size,
        num_heads,
        mlp_ratio=4.0,
        text_states_dim=1024,
        use_flash_attn=False,
        qk_norm=False,
        norm_type="layer",
        skip=False,
    ):
        super().__init__()
        self.use_flash_attn = use_flash_attn
        use_ele_affine = True

        if norm_type == "layer":
            norm_layer = FP32_Layernorm
        elif norm_type == "rms":
            norm_layer = RMSNorm
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        # ========================= Self-Attention =========================
        self.norm1 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        if use_flash_attn:
            self.attn1 = FlashSelfMHAModified(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)
        else:
            self.attn1 = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm)

        # ========================= FFN =========================
        self.norm2 = norm_layer(hidden_size, elementwise_affine=use_ele_affine, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # TODO
        approx_gelu = mint.nn.functional.gelu
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # ========================= Add =========================
        # Simply use add like SDXL.
        self.default_modulation = nn.SequentialCell(FP32_SiLU(), nn.Dense(c_emb_size, hidden_size, has_bias=True))

        # ========================= Cross-Attention =========================
        if use_flash_attn:
            self.attn2 = FlashCrossMHAModified(
                hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm
            )
        else:
            self.attn2 = CrossAttention(
                hidden_size, text_states_dim, num_heads=num_heads, qkv_bias=True, qk_norm=qk_norm
            )
        self.norm3 = norm_layer(hidden_size, elementwise_affine=True, eps=1e-6)

        # ========================= Skip Connection =========================
        if skip:
            self.skip_norm = norm_layer(2 * hidden_size, elementwise_affine=True, eps=1e-6)
            self.skip_linear = nn.Dense(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None

        self.gradient_checkpointing = False

    def _forward(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        # Long Skip Connection
        if self.skip_linear is not None:
            cat = ops.cat([x, skip], axis=-1)
            cat = self.skip_norm(cat)
            x = self.skip_linear(cat)

        # Self-Attention
        shift_msa = self.default_modulation(c).unsqueeze(dim=1)
        x = (
            x
            + self.attn1(
                self.norm1(x) + shift_msa,
                freq_cis_img,
            )[0]
        )

        # Cross-Attention
        x = x + self.attn2(self.norm3(x), text_states, freq_cis_img)[0]

        # FFN Layer
        mlp_inputs = self.norm2(x)
        x = x + self.mlp(mlp_inputs)

        return x

    def construct(self, x, c=None, text_states=None, freq_cis_img=None, skip=None):
        # if self.gradient_checkpointing and self.training:
        #     return checkpoint.checkpoint(self._forward, x, c, text_states, freq_cis_img, skip)
        return self._forward(x, c, text_states, freq_cis_img, skip)


class FinalLayer(nn.Cell):
    """
    The final layer of HunYuanDiT.
    """

    def __init__(self, final_hidden_size, c_emb_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = LayerNorm(final_hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Dense(final_hidden_size, patch_size * patch_size * out_channels, has_bias=True)
        self.adaLN_modulation = nn.SequentialCell(
            FP32_SiLU(), nn.Dense(c_emb_size, 2 * final_hidden_size, has_bias=True)
        )

    def construct(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, axis=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class HunYuanDiT(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    HunYuanDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    input_size: tuple
        The size of the input image.
    patch_size: int
        The size of the patch.
    in_channels: int
        The number of input channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    depth: int
        The number of transformer blocks.
    num_heads: int
        The number of attention heads.
    mlp_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    log_fn: callable
        The logging function.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        input_size: tuple = (32, 32),
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        log_fn: callable = print,
    ):
        super().__init__()
        self.args = args
        self.log_fn = log_fn
        self.depth = depth
        self.learn_sigma = args.learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if args.learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.text_states_dim = args.text_states_dim
        self.text_states_dim_t5 = args.text_states_dim_t5
        self.text_len = args.text_len
        self.text_len_t5 = args.text_len_t5
        self.norm = args.norm

        use_flash_attn = args.infer_mode == "fa" or args.use_flash_attn
        if use_flash_attn:
            log_fn("    Enable Flash Attention.")
        qk_norm = args.qk_norm  # See http://arxiv.org/abs/2302.05442 for details.

        self.mlp_t5 = nn.SequentialCell(
            nn.Dense(self.text_states_dim_t5, self.text_states_dim_t5 * 4, has_bias=True),
            FP32_SiLU(),
            nn.Dense(self.text_states_dim_t5 * 4, self.text_states_dim, has_bias=True),
        )
        # learnable replace
        self.text_embedding_padding = ms.Parameter(
            ops.randn(self.text_len + self.text_len_t5, self.text_states_dim, dtype=ms.float32),
            name="text_embedding_padding",
        )

        # Attention pooling
        pooler_out_dim = 1024
        self.pooler = AttentionPool(self.text_len_t5, self.text_states_dim_t5, num_heads=8, output_dim=pooler_out_dim)

        # Dimension of the extra input vectors
        self.extra_in_dim = pooler_out_dim

        if args.size_cond:
            # Image size and crop size conditions
            self.extra_in_dim += 6 * 256

        if args.use_style_cond:
            # Here we use a default learned embedder layer for future extension.
            self.style_embedder = nn.Embedding(1, hidden_size)
            self.extra_in_dim += hidden_size
        else:
            self.style_embedder = None

        # Text embedding for `add`
        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.extra_embedder = nn.SequentialCell(
            nn.Dense(self.extra_in_dim, hidden_size * 4),
            FP32_SiLU(),
            nn.Dense(hidden_size * 4, hidden_size, has_bias=True),
        )

        # Image embedding
        num_patches = self.x_embedder.num_patches
        log_fn(f"    Number of tokens: {num_patches}")

        # HUnYuanDiT Blocks
        self.blocks = nn.CellList(
            [
                HunYuanDiTBlock(
                    hidden_size=hidden_size,
                    c_emb_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    text_states_dim=self.text_states_dim,
                    use_flash_attn=use_flash_attn,
                    qk_norm=qk_norm,
                    norm_type=self.norm,
                    skip=layer > depth // 2,
                )
                for layer in range(depth)
            ]
        )

        self.final_layer = FinalLayer(hidden_size, hidden_size, patch_size, self.out_channels)
        self.unpatchify_channels = self.out_channels

        self.initialize_weights()

    def check_condition_validation(self, image_meta_size, style):
        if self.args.size_cond is None and image_meta_size is not None:
            raise ValueError(
                f"When `size_cond` is None, `image_meta_size` should be None, but got " f"{type(image_meta_size)}. "
            )
        if self.args.size_cond is not None and image_meta_size is None:
            raise ValueError("When `size_cond` is not None, `image_meta_size` should not be None. ")
        if not self.args.use_style_cond and style is not None:
            raise ValueError(f"When `use_style_cond` is False, `style` should be None, but got {type(style)}. ")
        if self.args.use_style_cond and style is None:
            raise ValueError("When `use_style_cond` is True, `style` should be not None.")

    def enable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = True

    def disable_gradient_checkpointing(self):
        for block in self.blocks:
            block.gradient_checkpointing = False

    def construct(
        self,
        x,
        t,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        cos_cis_img=None,
        sin_cis_img=None,
        return_dict=False,
        controls=None,
    ):
        """
        Forward pass of the encoder.

        Parameters
        ----------
        x: ms.Tensor
            (B, D, H, W)
        t: ms.Tensor
            (B)
        encoder_hidden_states: ms.Tensor
            CLIP text embedding, (B, L_clip, D)
        text_embedding_mask: ms.Tensor
            CLIP text embedding mask, (B, L_clip)
        encoder_hidden_states_t5: ms.Tensor
            T5 text embedding, (B, L_t5, D)
        text_embedding_mask_t5: ms.Tensor
            T5 text embedding mask, (B, L_t5)
        image_meta_size: ms.Tensor
            (B, 6)
        style: ms.Tensor
            (B)
        cos_cis_img: ms.Tensor
        sin_cis_img: ms.Tensor
        return_dict: bool
            Whether to return a dictionary.
        """
        text_states = encoder_hidden_states  # 2,77,1024
        text_states_t5 = encoder_hidden_states_t5  # 2,256,2048
        text_states_mask = text_embedding_mask.bool()  # 2,77
        text_states_t5_mask = text_embedding_mask_t5.bool()  # 2,256
        b_t5, l_t5, c_t5 = text_states_t5.shape
        text_states_t5 = self.mlp_t5(text_states_t5.view(-1, c_t5))
        text_states = ops.cat([text_states, text_states_t5.view(b_t5, l_t5, -1)], axis=1)  # 2,205，1024
        clip_t5_mask = ops.cat([text_states_mask, text_states_t5_mask], axis=-1)

        text_states = ops.where(
            clip_t5_mask.unsqueeze(2), text_states, self.text_embedding_padding.unsqueeze(0).to(text_states.dtype)
        )

        _, _, oh, ow = x.shape
        th, tw = oh // self.patch_size, ow // self.patch_size

        # ========================= Build time and image embedding =========================
        t = self.t_embedder(t)
        x = self.x_embedder(x)

        # Get image RoPE embedding according to `reso`lution.
        freqs_cis_img = (cos_cis_img, sin_cis_img)

        # ========================= Concatenate all extra vectors =========================
        # Build text tokens with pooling
        extra_vec = self.pooler(encoder_hidden_states_t5)

        if self.args.size_cond is None:
            image_meta_size = None
        # self.check_condition_validation(image_meta_size, style)
        # Build image meta size tokens if applicable
        if image_meta_size is not None:
            image_meta_size = timestep_embedding(image_meta_size.view(-1), 256)  # [B * 6, 256]
            if self.args.use_fp16:
                image_meta_size = image_meta_size.half()
            image_meta_size = image_meta_size.view(-1, 6 * 256)
            extra_vec = ops.cat([extra_vec, image_meta_size], axis=1)  # [B, D + 6 * 256]

        # Build style tokens
        if self.args.use_style_cond and style is not None:
            style_embedding = self.style_embedder(style)
            extra_vec = ops.cat([extra_vec, style_embedding], axis=1)

        # Concatenate all extra vectors
        c = t + self.extra_embedder(extra_vec)  # [B, D]

        # ========================= Forward pass through HunYuanDiT blocks =========================
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.depth // 2:
                if controls is not None:
                    skip = skips.pop() + controls.pop()
                else:
                    skip = skips.pop()
                x = block(x, c, text_states, freqs_cis_img, skip)  # (N, L, D)
            else:
                x = block(x, c, text_states, freqs_cis_img)  # (N, L, D)

            if layer < (self.depth // 2 - 1):
                skips.append(x)

        # ========================= Final layer =========================
        x = self.final_layer(x, c)  # (N, L, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, th, tw)  # (N, out_channels, H, W)

        if return_dict:
            return {"x": x}
        return (x,)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                module.weight.set_data(
                    initializer(XavierUniform(), shape=module.weight.shape, dtype=module.weight.dtype)
                )
                if module.bias is not None:
                    module.bias.set_data(initializer(Constant(0), shape=module.bias.shape, dtype=module.bias.dtype))

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Dense (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        w.set_data(initializer(XavierUniform(), shape=w.shape, dtype=w.dtype))
        w = w.view(w.shape[0], -1)
        self.x_embedder.proj.bias.set_data(
            initializer(Constant(0), shape=self.x_embedder.proj.bias.shape, dtype=self.x_embedder.proj.bias.dtype)
        )

        # Initialize label embedding table:
        self.extra_embedder[0].weight.set_data(
            initializer(
                Normal(sigma=0.02), shape=self.extra_embedder[0].weight.shape, dtype=self.extra_embedder[0].weight.dtype
            )
        )
        self.extra_embedder[2].weight.set_data(
            initializer(
                Normal(sigma=0.02), shape=self.extra_embedder[2].weight.shape, dtype=self.extra_embedder[2].weight.dtype
            )
        )

        # Initialize timestep embedding MLP:
        self.t_embedder.mlp[0].weight.set_data(
            initializer(
                Normal(sigma=0.02), shape=self.t_embedder.mlp[0].weight.shape, dtype=self.t_embedder.mlp[0].weight.dtype
            )
        )
        self.t_embedder.mlp[2].weight.set_data(
            initializer(
                Normal(sigma=0.02), shape=self.t_embedder.mlp[2].weight.shape, dtype=self.t_embedder.mlp[2].weight.dtype
            )
        )

        # Zero-out adaLN modulation layers in HunYuanDiT blocks:
        for block in self.blocks:
            block.default_modulation[-1].weight.set_data(
                initializer(
                    Constant(0),
                    shape=block.default_modulation[-1].weight.shape,
                    dtype=block.default_modulation[-1].weight.dtype,
                )
            )
            block.default_modulation[-1].bias.set_data(
                initializer(
                    Constant(0),
                    shape=block.default_modulation[-1].bias.shape,
                    dtype=block.default_modulation[-1].bias.dtype,
                )
            )

        # Zero-out output layers:
        self.final_layer.adaLN_modulation[-1].weight.set_data(
            initializer(
                Constant(0),
                shape=self.final_layer.adaLN_modulation[-1].weight.shape,
                dtype=self.final_layer.adaLN_modulation[-1].weight.dtype,
            )
        )
        self.final_layer.adaLN_modulation[-1].bias.set_data(
            initializer(
                Constant(0),
                shape=self.final_layer.adaLN_modulation[-1].bias.shape,
                dtype=self.final_layer.adaLN_modulation[-1].bias.dtype,
            )
        )
        self.final_layer.linear.weight.set_data(
            initializer(
                Constant(0), shape=self.final_layer.linear.weight.shape, dtype=self.final_layer.linear.weight.dtype
            )
        )
        self.final_layer.linear.bias.set_data(
            initializer(Constant(0), shape=self.final_layer.linear.bias.shape, dtype=self.final_layer.linear.bias.dtype)
        )

    def unpatchify(self, x, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        p = self.x_embedder.patch_size[0]
        # h = w = int(x.shape[1] ** 0.5)

        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4)
        imgs = x.reshape(x.shape[0], c, h * p, w * p)
        return imgs

    def _replace_module(self, parent, child_name, new_module, child) -> None:
        setattr(parent, child_name, new_module)
        # It's not necessary to set requires_grad here, as that is handled by
        # _mark_only_adapters_as_trainable

        # child layer wraps the original module, unpack it
        if hasattr(child, "base_layer"):
            child = child.get_base_layer()
        elif hasattr(child, "quant_linear_module"):
            # TODO maybe not necessary to have special treatment?
            child = child.quant_linear_module

        if not hasattr(new_module, "base_layer"):
            new_module.weight = child.weight
            if hasattr(child, "bias"):
                new_module.bias = child.bias

        if getattr(child, "state", None) is not None:
            if hasattr(new_module, "base_layer"):
                new_module.base_layer.state = child.state
            else:
                new_module.state = child.state

    def merge_and_unload(
        self,
        merge=True,
        progressbar: bool = False,
        safe_merge: bool = False,
        adapter_names=None,
    ):
        if merge:
            if getattr(self, "quantization_method", None) == "gptq":
                raise ValueError("Cannot merge layers when the model is gptq quantized")

        def merge_recursively(module):
            # helper function to recursively merge the base_layer of the target
            path = []
            layer = module
            while hasattr(layer, "base_layer"):
                path.append(layer)
                layer = layer.base_layer
            for layer_before, layer_after in zip(path[:-1], path[1:]):
                layer_after.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                layer_before.base_layer = layer_after.base_layer
            module.merge(safe_merge=safe_merge, adapter_names=adapter_names)

        key_list = [key for key, _ in self.cells_and_names()]
        desc = "Unloading " + ("and merging " if merge else "") + "model"

        for key in tqdm(key_list, disable=not progressbar, desc=desc):
            try:
                parent, target, target_name = _get_submodules(self, key)
            except AttributeError:
                continue

            if hasattr(target, "base_layer"):
                if merge:
                    merge_recursively(target)
                self._replace_module(parent, target_name, target.get_base_layer(), target)
            elif isinstance(target, ModulesToSaveWrapper):
                # save any additional trainable modules part of `modules_to_save`
                new_module = target.modules_to_save[target.active_adapter]
                if hasattr(new_module, "base_layer"):
                    # check if the module is itself a tuner layer
                    if merge:
                        new_module.merge(safe_merge=safe_merge, adapter_names=adapter_names)
                    new_module = new_module.get_base_layer()
                setattr(parent, target_name, new_module)


#################################################################################
#                            HunYuanDiT Configs                                 #
#################################################################################

HUNYUAN_DIT_CONFIG = {
    "DiT-g/2": {"depth": 40, "hidden_size": 1408, "patch_size": 2, "num_heads": 16, "mlp_ratio": 4.3637},
    "DiT-XL/2": {"depth": 28, "hidden_size": 1152, "patch_size": 2, "num_heads": 16},
}


def DiT_g_2(args, **kwargs):
    return HunYuanDiT(args, depth=40, hidden_size=1408, patch_size=2, num_heads=16, mlp_ratio=4.3637, **kwargs)


def DiT_XL_2(args, **kwargs):
    return HunYuanDiT(args, depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


HUNYUAN_DIT_MODELS = {
    "DiT-g/2": DiT_g_2,
    "DiT-XL/2": DiT_XL_2,
}
