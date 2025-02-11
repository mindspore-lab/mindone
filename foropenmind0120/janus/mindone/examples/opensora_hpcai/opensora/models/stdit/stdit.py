"""
OpenSora v1.0 STDiT architecture
"""

import os
import re

import numpy as np
from mindcv.models.layers import DropPath
from opensora.models.layers.blocks import (
    CaptionEmbedder,
    LayerNorm,
    LinearPatchEmbed,
    Mlp,
    MultiHeadCrossAttention,
    PatchEmbed,
    PatchEmbed3D,
    SelfAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    t2i_modulate,
)

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import XavierUniform, initializer

from mindone.models.utils import constant_, normal_, xavier_uniform_


class STDiTBlock(nn.Cell):
    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        assert not enable_layernorm_kernel, "Not implemented"
        assert not enable_sequence_parallelism, "Not implemented"

        self.attn_cls = SelfAttention
        self.mha_cls = MultiHeadCrossAttention

        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attention=enable_flashattn,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads, enable_flash_attention=enable_flashattn)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # TODO: check parsing approx_gelu
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = ms.Parameter(ops.randn(6, hidden_size) / hidden_size**0.5)

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attention=enable_flashattn,
        )

    @staticmethod
    def _rearrange_in_S(x, T):
        # x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        B, TS, C = x.shape
        S = TS // T
        x = ops.reshape(x, (B * T, S, C))
        return x

    @staticmethod
    def _rearrange_out_S(x, T):
        # x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        BT, S, C = x.shape
        B = BT // T
        x = ops.reshape(x, (B, T * S, C))
        return x

    @staticmethod
    def _rearrange_in_T(x, T):
        # x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        B, TS, C = x.shape
        S = TS // T
        x = ops.reshape(x, (B, T, S, C))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B * S, T, C))
        return x

    @staticmethod
    def _rearrange_out_T(x, S):
        # x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        BS, T, C = x.shape
        B = BS // S
        x = ops.reshape(x, (B, S, T, C))
        x = ops.transpose(x, (0, 2, 1, 3))
        x = ops.reshape(x, (B, T * S, C))
        return x

    def construct(self, x, y, t, mask=None, tpe=None):
        """
        x: (B N C_x)
        y: (1 B*N_tokens C_y)
        t: (B C_t)
        mask: (B, N_tokens)
        """
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, axis=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = self._rearrange_in_S(x_m, T=self.d_t)
        x_s = self.attn(x_s)

        x_s = self._rearrange_out_S(x_s, T=self.d_t)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        x_t = self._rearrange_in_T(x, T=self.d_t)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = self.attn_temp(x_t)

        x_t = self._rearrange_out_T(x_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x


class STDiT(nn.Cell):
    """
    Spatial-Temporal DiT model

    Note:
        - The arguments `enable_layernorm_kernel` and `enable_sequence_parallelism` are only reserved for compatibility. They are not supported currently.
        - `patchify_conv3d_replace`: If None, use the default conv3d for patch embedding. If `conv2d`, replace 1xhxw conv3d with hxw conv2d equivalently. \
                If `linear`, replace the conv3d with a dense layer equivalently.
    """

    def __init__(
        self,
        input_size=(1, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        dtype=ms.float32,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        use_recompute=False,
        num_recompute_blocks=None,
        patchify_conv3d_replace=None,
        manual_pad=False,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = int(np.prod([input_size[i] // patch_size[i] for i in range(3)]))
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.space_scale = space_scale
        self.time_scale = time_scale

        assert patchify_conv3d_replace in [None, "linear", "conv2d"]
        assert not manual_pad, "manual_pad not supported for STDiT v1"

        pos_embed = self.get_spatial_pos_embed()
        pos_embed_temporal = self.get_temporal_pos_embed()
        self.pos_embed = Tensor(pos_embed, dtype=ms.float32)
        self.pos_embed_temporal = Tensor(pos_embed_temporal, dtype=ms.float32)

        # conv3d replacement. FIXME: after CANN+MS support bf16 and fp32, remove redundancy
        self.patchify_conv3d_replace = patchify_conv3d_replace
        if patchify_conv3d_replace is None:
            self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        elif patchify_conv3d_replace == "linear":
            assert patch_size[0] == 1 and patch_size[1] == patch_size[2]
            assert input_size[1] == input_size[2]
            print("Replace conv3d patchify with linear layer")
            self.x_embedder = LinearPatchEmbed(patch_size[1], in_channels, hidden_size, bias=True)
        elif patchify_conv3d_replace == "conv2d":
            assert patch_size[0] == 1 and patch_size[1] == patch_size[2]
            assert input_size[1] == input_size[2]
            print("Replace conv3d patchify with conv2d layer")
            self.x_embedder = PatchEmbed(patch_size[1], in_channels, hidden_size, bias=True)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size, has_bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        drop_path = np.linspace(0, drop_path, depth)
        self.blocks = nn.CellList(
            [
                STDiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(
            hidden_size, int(np.prod(self.patch_size)), self.out_channels, d_t=self.num_temporal, d_s=self.num_spatial
        )

        # init model
        self.initialize_weights()
        self.initialize_temporal()

        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

        # sequence parallel related configs
        self.enable_sequence_parallelism = enable_sequence_parallelism
        self.sp_rank = None

        if use_recompute:
            if num_recompute_blocks is None:
                num_recompute_blocks = len(self.blocks)
            print("Num recomputed stdit blocks: {}".format(num_recompute_blocks))
            for i, block in enumerate(self.blocks):
                # recompute the first N blocks
                if i < num_recompute_blocks:
                    self.recompute(block)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def construct(self, x, timestep, y, mask=None, **kwargs):
        """
        Args:
            x (ms.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (ms.Tensor): diffusion time steps; of shape [B]
            y (ms.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (ms.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (ms.Tensor): output latent representation; of shape [B, C, T, H, W]
        """

        # embedding
        if self.patchify_conv3d_replace is None:
            x = self.x_embedder(x)  # out: [B, N, C]=[B, thw, C]
        else:
            # (b c t h w) -> (bt c h w)
            _b, _c, _t, _h, _w = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape((_b * _t, _c, _h, _w))
            x = self.x_embedder(x)  # out: [bt, h'w', d]
            # (bt, h'w', d] -> (b , t'h'w', d)
            x = x.reshape((_b, -1, self.hidden_size))

        # x = rearrange(x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial)
        B, TS, C = x.shape
        x = ops.reshape(x, (B, TS // self.num_spatial, self.num_spatial, C))

        x = x + self.pos_embed
        # x = rearrange(x, "B T S C -> B (T S) C")
        x = ops.reshape(x, (B, TS, C))

        t = self.t_embedder(timestep)  # [B, C]
        # why project again on t ?
        t0 = self.t_block(t)  # [B, C]
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        # (b 1 max_tokens d_t) -> (b max_tokens d_t)  -> (1 b*max_tokens d_t)
        y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                tpe = self.pos_embed_temporal
            else:
                tpe = None

            x = block(x, y, t0, mask=mask, tpe=tpe)

        # x.shape: [B, N, C]
        # final process
        x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]

        x = self.unpatchify(x)  # [B, C_out, T, H, W]

        return x

    # @ms.jit
    def construct_with_cfg(self, x, t, y, mask=None, cfg_scale=4.0, cfg_channel=None, **kwargs):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)

        # model_out = self.construct(combined, t, y=y, mask=mask)
        model_out = self(combined, t, y=y, mask=mask)
        if cfg_channel is None:
            cfg_channel = self.in_channels
        # torch only takes the first 3 dimension for eps. but for z=4, out z=8, the first 4 dims are for eps, the rest 4 dim are for variance.
        eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)

    def unpatchify(self, x):
        """
        Args:
            x (ms.Tensor): of shape [B, N, C]

        Return:
            x (ms.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        C_out = self.out_channels

        # "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)"
        B, Nthw, THWC = x.shape
        x = ops.reshape(x, (B, N_t, N_h, N_w, T_p, H_p, W_p, C_out))
        x = ops.transpose(x, (0, 7, 1, 4, 2, 5, 3, 6))
        x = ops.reshape(x, (B, C_out, N_t * T_p, N_h * H_p, N_w * W_p))

        return x

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        # pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        pos_embed = np.expand_dims(pos_embed, axis=0)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        # pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        pos_embed = np.expand_dims(pos_embed, axis=0)
        return pos_embed

    def freeze_not_temporal(self):
        for param in self.get_parameters():
            if "attn_temp" not in param.name:
                param.requires_grad = False

    def freeze_text(self):
        for param in self.get_parameters():
            if "cross_attn" in param.name:
                param.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            constant_(block.attn_temp.proj.weight, 0)
            constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        # xavier_uniform_(w.view([w.shape[0], -1]))
        w = self.x_embedder.proj.weight
        w_flatted = w.reshape(w.shape[0], -1)
        # FIXME: incompatible in optim parallel mode
        # FIXME: impl in torch can be incorrect. can be reshape order mismatch
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))

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

    def load_from_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found. No checkpoint loaded!!")
        else:
            sd = ms.load_checkpoint(ckpt_path)

            regex = re.compile(r"^network\.|\._backbone")
            sd = {regex.sub("", k): v for k, v in sd.items()}

            # load conv3d weight from pretrained conv2d or dense layer
            key_3d = "x_embedder.proj.weight"
            if self.patchify_conv3d_replace == "linear":
                if len(sd[key_3d].shape) == 5:
                    conv3d_weight = sd.pop(key_3d)  # c_out, c_in, 1, 2, 2
                    assert conv3d_weight.shape[-3] == 1
                    sd[key_3d] = ms.Parameter(conv3d_weight.reshape(conv3d_weight.shape[0], -1), name=key_3d)
            elif self.patchify_conv3d_replace == "conv2d":
                if len(sd[key_3d].shape) == 5:
                    conv3d_weight = sd.pop(key_3d)  # c_out, c_in, 1, 2, 2
                    assert conv3d_weight.shape[-3] == 1
                    sd[key_3d] = ms.Parameter(conv3d_weight.squeeze(axis=-3), name=key_3d)

            m, u = ms.load_param_into_net(self, sd)
            print("net param not load: ", m, len(m))
            print("ckpt param not load: ", u, len(u))


def STDiT_XL_2(from_pretrained=None, **kwargs):
    model = STDiT(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        ms.load_checkpoint(from_pretrained, model)
    return model
