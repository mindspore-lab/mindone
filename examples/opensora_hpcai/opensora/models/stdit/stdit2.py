import os
import re
from typing import Optional, Tuple

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
    PositionEmbedding2D,
    SelfAttention,
    SizeEmbedder,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    t2i_modulate,
)
from opensora.models.layers.rotary_embedding import RotaryEmbedding

import mindspore as ms
from mindspore import Parameter, Tensor, dtype, load_checkpoint, load_param_into_net, nn, ops
from mindspore.common.initializer import XavierUniform, initializer

from mindone.models.utils import constant_, normal_, xavier_uniform_


class STDiT2Block(nn.Cell):
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        rope=None,
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn

        assert not enable_layernorm_kernel, "Not implemented"
        if enable_sequence_parallelism:
            raise NotImplementedError("Sequence parallelism is not supported yet.")
        else:
            self.attn_cls = SelfAttention
            self.mha_cls = MultiHeadCrossAttention

        # spatial branch
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attention=enable_flashattn,
            qk_norm=qk_norm,
        )
        self.scale_shift_table = Parameter(ops.randn(6, hidden_size) / hidden_size**0.5)

        # cross attn
        self.cross_attn = self.mha_cls(hidden_size, num_heads, enable_flash_attention=enable_flashattn)

        # mlp branch
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # temporal branch
        self.norm_temp = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)  # new
        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flash_attention=self.enable_flashattn,
            rope=rope,
            qk_norm=qk_norm,
        )
        self.scale_shift_table_temporal = Parameter(ops.randn(3, hidden_size) / hidden_size**0.5)  # new

    @staticmethod
    def t_mask_select(x_mask: Tensor, x: Tensor, masked_x: Tensor, T: int, S: int) -> Tensor:
        x = x.reshape(x.shape[0], T, S, x.shape[-1])  # B (T S) C -> B T S C
        masked_x = masked_x.reshape(masked_x.shape[0], T, S, masked_x.shape[-1])  # B (T S) C -> B T S C
        x = ops.where(x_mask[:, :, None, None], x, masked_x)  # x_mask: [B, T]
        return x.reshape(x.shape[0], T * S, x.shape[-1])  # B T S C -> B (T S) C

    def construct(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        t_tmp: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        t0: Optional[Tensor] = None,
        t0_tmp: Optional[Tensor] = None,
        T: Optional[int] = None,
        S: Optional[int] = None,
        spatial_mask: Optional[Tensor] = None,
        temporal_pos: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
    ):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, axis=1)
        shift_tmp, scale_tmp, gate_tmp = (self.scale_shift_table_temporal[None] + t_tmp.reshape(B, 3, -1)).chunk(
            3, axis=1
        )

        shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (None,) * 6
        shift_tmp_zero, scale_tmp_zero, gate_tmp_zero = (None,) * 3
        if frames_mask is not None:
            shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = (
                self.scale_shift_table[None] + t0.reshape(B, 6, -1)
            ).chunk(6, axis=1)
            shift_tmp_zero, scale_tmp_zero, gate_tmp_zero = (
                self.scale_shift_table_temporal[None] + t0_tmp.reshape(B, 3, -1)
            ).chunk(3, axis=1)

        # modulate
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        if frames_mask is not None:
            x_m_zero = t2i_modulate(self.norm1(x), shift_msa_zero, scale_msa_zero)
            x_m = self.t_mask_select(frames_mask, x_m, x_m_zero, T, S)

        # spatial branch
        x_s = x_m.reshape(B * T, S, C)  # B (T S) C -> (B T) S C
        if spatial_mask is not None:
            spatial_mask = ops.repeat_interleave(spatial_mask.to(ms.int32), T, axis=0)  # B S -> (B T) S
        x_s = self.attn(x_s, mask=spatial_mask)
        x_s = x_s.reshape(B, T * S, C)  # (B T) S C -> B (T S) C

        if frames_mask is not None:
            x_s_zero = gate_msa_zero * x_s
            x_s = gate_msa * x_s
            x_s = self.t_mask_select(frames_mask, x_s, x_s_zero, T, S)
        else:
            x_s = gate_msa * x_s
        x = x + self.drop_path(x_s)

        # modulate
        x_m = t2i_modulate(self.norm_temp(x), shift_tmp, scale_tmp)
        if frames_mask is not None:
            x_m_zero = t2i_modulate(self.norm_temp(x), shift_tmp_zero, scale_tmp_zero)
            x_m = self.t_mask_select(frames_mask, x_m, x_m_zero, T, S)

        # temporal branch
        x_t = x_m.reshape(B, T, S, C).swapaxes(1, 2).reshape(B * S, T, C)  # B (T S) C -> (B S) T C
        if temporal_mask is not None:
            temporal_mask = ops.repeat_interleave(temporal_mask.to(ms.int32), S, axis=0)  # B T -> (B S) T
        x_t = self.attn_temp(x_t, mask=temporal_mask, freqs_cis=temporal_pos)
        x_t = x_t.reshape(B, S, T, C).swapaxes(1, 2).reshape(B, T * S, C)  # (B S) T C -> B (T S) C

        if frames_mask is not None:
            x_t_zero = gate_tmp_zero * x_t
            x_t = gate_tmp * x_t
            x_t = self.t_mask_select(frames_mask, x_t, x_t_zero, T, S)
        else:
            x_t = gate_tmp * x_t
        x = x + self.drop_path(x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # modulate
        x_m = t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)
        if frames_mask is not None:
            x_m_zero = t2i_modulate(self.norm2(x), shift_mlp_zero, scale_mlp_zero)
            x_m = self.t_mask_select(frames_mask, x_m, x_m_zero, T, S)

        # mlp
        x_mlp = self.mlp(x_m)
        if frames_mask is not None:
            x_mlp_zero = gate_mlp_zero * x_mlp
            x_mlp = gate_mlp * x_mlp
            x_mlp = self.t_mask_select(frames_mask, x_mlp, x_mlp_zero, T, S)
        else:
            x_mlp = gate_mlp * x_mlp
        x = x + self.drop_path(x_mlp)

        return x


class STDiT2(nn.Cell):
    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=32,
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
        dtype=dtype.float32,
        freeze=None,
        qk_norm=False,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        use_recompute=False,
        num_recompute_blocks=None,
        patchify_conv3d_replace=None,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel

        assert patchify_conv3d_replace in [None, "linear", "conv2d"]

        # support dynamic input
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size
        self.pos_embed = PositionEmbedding2D(hidden_size)

        self.patchify_conv3d_replace = patchify_conv3d_replace
        if patchify_conv3d_replace is None:
            self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        elif patchify_conv3d_replace == "linear":
            assert patch_size[0] == 1 and patch_size[1] == patch_size[2]
            print("Replace conv3d patchify with linear layer")
            self.x_embedder = LinearPatchEmbed(patch_size[1], in_channels, hidden_size, bias=True)
        elif patchify_conv3d_replace == "conv2d":
            assert patch_size[0] == 1 and patch_size[1] == patch_size[2]
            print("Replace conv3d patchify with conv2d layer")
            self.x_embedder = PatchEmbed(patch_size[1], in_channels, hidden_size, bias=True)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size))
        self.t_block_temp = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 3 * hidden_size))  # new
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        drop_path = np.linspace(0, drop_path, depth)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)  # new
        self.blocks = nn.CellList(
            [
                STDiT2Block(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    rope=self.rope.rotate_queries_or_keys,
                    qk_norm=qk_norm,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size).item(), self.out_channels)

        # multi_res
        assert self.hidden_size % 3 == 0, "hidden_size must be divisible by 3"
        self.csize_embedder = SizeEmbedder(self.hidden_size // 3)
        self.ar_embedder = SizeEmbedder(self.hidden_size // 3)
        self.fl_embedder = SizeEmbedder(self.hidden_size)  # new
        self.fps_embedder = SizeEmbedder(self.hidden_size)  # new

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

    def get_dynamic_size(self, x: Tensor) -> Tuple[int, int, int]:
        _, _, T, H, W = x.shape
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return T, H, W

    def construct(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        num_frames: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        ar: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        spatial_pos: Optional[Tensor] = None,
        spatial_mask: Optional[Tensor] = None,
        temporal_pos: Optional[Tensor] = None,
        temporal_mask: Optional[Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        B = x.shape[0]

        # === process data info ===
        # 1. get dynamic size
        hw = ops.stack([height, width], axis=1)
        rs = (height[0] * width[0]) ** 0.5
        csize = self.csize_embedder(hw, B)

        # 2. get aspect ratio
        ar = ar.unsqueeze(1)
        ar = self.ar_embedder(ar, B)
        data_info = ops.cat([csize, ar], axis=1)

        # 3. get number of frames
        fl = num_frames.unsqueeze(1)
        fps = fps.unsqueeze(1)
        fl = self.fl_embedder(fl, B)
        fl = fl + self.fps_embedder(fps, B)

        # === get dynamic shape size ===
        _, _, Tx, Hx, Wx = x.shape
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        scale = rs / self.input_sq_size
        base_size = round(S**0.5)
        # BUG MS2.3rc1: ops.meshgrid() bprop is not supported

        if spatial_pos is None:
            pos_emb = ops.stop_gradient(self.pos_embed(H, W, scale=scale, base_size=base_size))
        else:
            pos_emb = spatial_pos

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

        x = x.reshape(B, T, S, x.shape[-1])  # B (T S) C -> B T S C
        x = x + pos_emb.to(x.dtype)
        x = x.reshape(B, T * S, x.shape[-1])  # B T S C -> B (T S) C

        # prepare adaIN
        t = self.t_embedder(timestep)  # [B, C]
        t_spc = t + data_info  # [B, C]
        t_tmp = t + fl  # [B, C]
        t_spc_mlp = self.t_block(t_spc)  # [B, 6*C]
        t_tmp_mlp = self.t_block_temp(t_tmp)  # [B, 3*C]

        t0_spc, t0_spc_mlp, t0_tmp_mlp = None, None, None
        if frames_mask is not None:
            t0_timestep = ops.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep)
            t0_spc = t0 + data_info
            t0_tmp = t0 + fl
            t0_spc_mlp = self.t_block(t0_spc)
            t0_tmp_mlp = self.t_block_temp(t0_tmp)

        # prepare y
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for block in self.blocks:
            x = block(
                x,
                y,
                t_spc_mlp,
                t_tmp_mlp,
                mask=mask,
                frames_mask=frames_mask,
                t0=t0_spc_mlp,
                t0_tmp=t0_tmp_mlp,
                T=T,
                S=S,
                spatial_mask=spatial_mask,
                temporal_pos=temporal_pos,
                temporal_mask=temporal_mask,
            )

        # x.shape: [B, N, C]
        # final process
        x = self.final_layer(x, t, frames_mask, t0_spc, T, S)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        return x.astype(dtype.float32)

    def construct_with_cfg(self, x, timestep, y, cfg_scale, cfg_channel=None, **kwargs):
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        if "frames_mask" in kwargs and kwargs["frames_mask"] is not None:
            if len(kwargs["frames_mask"]) != len(x):
                kwargs["frames_mask"] = ops.cat([kwargs["frames_mask"], kwargs["frames_mask"]], axis=0)
        model_out = self(combined, timestep, y, **kwargs)
        model_out = model_out["x"] if isinstance(model_out, dict) else model_out
        if cfg_channel is None:
            cfg_channel = model_out.shape[1] // 2
        eps, rest = model_out[:, :cfg_channel], model_out[:, cfg_channel:]
        cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = ops.cat([half_eps, half_eps], axis=0)
        return ops.cat([eps, rest], axis=1)

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        C_out = self.out_channels

        # B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)
        B, Nthw, THWC = x.shape
        x = ops.reshape(x, (B, N_t, N_h, N_w, T_p, H_p, W_p, C_out))
        x = ops.transpose(x, (0, 7, 1, 4, 2, 5, 3, 6))
        x = ops.reshape(x, (B, C_out, N_t * T_p, N_h * H_p, N_w * W_p))
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

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

        # Initialize patch_embed like nn.Dense (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight
        w_flatted = w.reshape(w.shape[0], -1)
        # FIXME: incompatible in optim parallel mode
        # FIXME: impl in torch can be incorrect. can be reshape order mismatch
        w.set_data(initializer(XavierUniform(), w_flatted.shape, w_flatted.dtype).reshape(w.shape))

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)
        normal_(self.t_block[1].weight, std=0.02)
        normal_(self.t_block_temp[1].weight, std=0.02)

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
            sd = load_checkpoint(ckpt_path)

            regex = re.compile(r"^network\.|\._backbone")
            sd = {regex.sub("", k): v for k, v in sd.items()}

            key_3d = "x_embedder.proj.weight"
            # load conv3d weight from pretrained conv2d or dense layer
            if self.patchify_conv3d_replace == "linear":
                if len(sd[key_3d].shape) == 5:
                    conv3d_weight = sd.pop(key_3d)  # c_out, c_in, 1, 2, 2
                    assert conv3d_weight.shape[-3] == 1
                    sd[key_3d] = Parameter(conv3d_weight.reshape(conv3d_weight.shape[0], -1), name=key_3d)
            elif self.patchify_conv3d_replace == "conv2d":
                if len(sd[key_3d].shape) == 5:
                    conv3d_weight = sd.pop(key_3d)  # c_out, c_in, 1, 2, 2
                    assert conv3d_weight.shape[-3] == 1
                    sd[key_3d] = Parameter(conv3d_weight.squeeze(axis=-3), name=key_3d)

            # Loading PixArt weights (T5's sequence length is 120 vs. 200 in STDiT2).
            if self.y_embedder.y_embedding.shape != sd["y_embedder.y_embedding"].shape:
                print("WARNING: T5's sequence length doesn't match STDiT2. Padding with default values.")
                param = sd["y_embedder.y_embedding"].value()
                sd["y_embedder.y_embedding"] = Parameter(
                    ops.concat((param, self.y_embedder.y_embedding.value()[param.shape[0] :]), axis=0),
                    name=self.y_embedder.y_embedding.name,
                    requires_grad=self.y_embedder.y_embedding.requires_grad,
                )

            m, u = load_param_into_net(self, sd)
            print("net param not load: ", m, len(m))
            print("ckpt param not load: ", u, len(u))


def STDiT2_XL_2(from_pretrained=None, **kwargs):
    model = STDiT2(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(from_pretrained, model)
    return model
