"""
OpenSora v1.2 STDiT architecture, using DSP parallism (https://arxiv.org/abs/2403.10266)
Reference: https://github.com/NUS-HPC-AI-Lab/VideoSys/blob/master/videosys/models/transformers/open_sora_transformer_3d.py
"""

import logging
import os
import re
from typing import Optional, Tuple, Union

import numpy as np
from mindcv.models.layers import DropPath
from opensora.acceleration.communications import AlltoAll, GatherFowardSplitBackward, SplitFowardGatherBackward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
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
    t_mask_select,
)
from opensora.models.layers.operation_selector import check_dynamic_mode, get_chunk_op
from opensora.models.layers.rotary_embedding import RotaryEmbedding

import mindspore as ms
from mindspore import Parameter, Tensor, load_checkpoint, load_param_into_net, mint, nn, ops
from mindspore.communication import get_group_size

from mindone.models.utils import constant_, normal_, xavier_uniform_

logger = logging.getLogger(__name__)


class STDiT3DSPBlock(nn.Cell):
    @ms.lazy_inline(policy="front")
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        drop_path=0.0,
        rope=None,
        qk_norm=False,
        temporal=False,
        enable_flash_attention=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
    ):
        super().__init__()
        self.temporal = temporal
        self.hidden_size = hidden_size
        self.enable_sequence_parallelism = enable_sequence_parallelism

        assert not enable_layernorm_kernel, "Not implemented"
        attn_cls = SelfAttention
        mha_cls = MultiHeadCrossAttention

        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            rope=rope,
            enable_flash_attention=enable_flash_attention,
        )
        self.cross_attn = mha_cls(hidden_size, num_heads, enable_flash_attention=enable_flash_attention)
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = Parameter(np.random.randn(6, hidden_size).astype(np.float32) / hidden_size**0.5)
        self.chunk = get_chunk_op()

        if self.enable_sequence_parallelism:
            sp_group = get_sequence_parallel_group()
            self.all2all_to_spatial_shard = AlltoAll(2, 1, group=sp_group)
            self.all2all_to_temporal_shard = AlltoAll(1, 2, group=sp_group)

    def construct(
        self,
        x: Tensor,
        y: Tensor,
        t: Tensor,
        mask: Optional[Tensor] = None,  # text mask
        frames_mask: Optional[Tensor] = None,  # temporal mask
        t0: Optional[Tensor] = None,  # t with timestamp=0
        T: Optional[int] = None,  # number of frames
        S: Optional[int] = None,  # number of pixel patches
        spatial_pad: Union[int, Tensor] = 0,
        temporal_pad: Union[int, Tensor] = 0,
    ) -> Tensor:
        # prepare modulate parameters
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.chunk(
            self.scale_shift_table[None] + t.reshape(B, 6, -1), 6, 1
        )

        # frames mask branch
        shift_msa_zero, scale_msa_zero, gate_msa_zero, shift_mlp_zero, scale_mlp_zero, gate_mlp_zero = self.chunk(
            self.scale_shift_table[None] + t0.reshape(B, 6, -1), 6, 1
        )

        # modulate (attention)
        norm1 = self.norm1(x)
        x_m = t2i_modulate(norm1, shift_msa, scale_msa)
        # frames mask branch
        x_m_zero = t2i_modulate(norm1, shift_msa_zero, scale_msa_zero)
        x_m = t_mask_select(frames_mask, x_m, x_m_zero, T, S)

        # attention
        if self.temporal:
            if self.enable_sequence_parallelism:
                x_m, S, T = self.dynamic_switch(
                    x_m, S, T, to_spatial_shard=True, spatial_pad=spatial_pad, temporal_pad=temporal_pad
                )
            x_m = x_m.reshape(B, T, S, C).swapaxes(1, 2).reshape(B * S, T, C)  # B (T S) C -> (B S) T C
            x_m = self.attn(x_m)
            x_m = x_m.reshape(B, S, T, C).swapaxes(1, 2).reshape(B, T * S, C)  # (B S) T C -> B (T S) C
            if self.enable_sequence_parallelism:
                x_m, S, T = self.dynamic_switch(
                    x_m, S, T, to_spatial_shard=False, spatial_pad=spatial_pad, temporal_pad=temporal_pad
                )
        else:
            x_m = x_m.reshape(B * T, S, C)  # B (T S) C -> (B T) S C
            x_m = self.attn(x_m)
            x_m = x_m.reshape(B, T * S, C)  # (B T) S C -> B (T S) C

        # modulate (attention)
        x_m_s = gate_msa * x_m
        # frames mask branch
        x_m_s_zero = gate_msa_zero * x_m
        x_m_s = t_mask_select(frames_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        # cross attention
        x = x + self.cross_attn(x, y, mask)

        # modulate (MLP)
        norm2 = self.norm2(x)
        x_m = t2i_modulate(norm2, shift_mlp, scale_mlp)
        # frames mask branch
        x_m_zero = t2i_modulate(norm2, shift_mlp_zero, scale_mlp_zero)
        x_m = t_mask_select(frames_mask, x_m, x_m_zero, T, S)

        # MLP
        x_m = self.mlp(x_m)

        # modulate (MLP)
        x_m_s = gate_mlp * x_m
        # frames mask branch
        x_m_s_zero = gate_mlp_zero * x_m
        x_m_s = t_mask_select(frames_mask, x_m_s, x_m_s_zero, T, S)

        # residual
        x = x + self.drop_path(x_m_s)

        return x

    def dynamic_switch(
        self,
        x: Tensor,
        s: int,
        t: int,
        to_spatial_shard: bool,
        spatial_pad: Union[int, Tensor],
        temporal_pad: Union[int, Tensor],
    ) -> Tuple[Tensor, int, int]:
        b, _, d = x.shape
        x = ops.reshape(x, (b, t, s, d))
        if to_spatial_shard:
            x = self.all2all_to_spatial_shard(x, spatial_pad, temporal_pad)
        else:
            x = self.all2all_to_temporal_shard(x, temporal_pad, spatial_pad)
        new_s, new_t = x.shape[2], x.shape[1]
        x = ops.reshape(x, (b, -1, d))
        return x, new_s, new_t


class STDiT3_DSP(nn.Cell):
    def __init__(
        self,
        input_size=(None, None, None),
        input_sq_size=512,
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        caption_channels=4096,
        model_max_length=300,
        qk_norm=True,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        only_train_temporal=False,
        freeze_y_embedder=False,
        skip_y_embedder=False,
        use_recompute=False,
        num_recompute_blocks=None,
        patchify_conv3d_replace=None,
        manual_pad=False,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels

        # model size related
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # computation related
        self.drop_path = drop_path
        self.enable_flash_attn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.enable_sequence_parallelism = enable_sequence_parallelism

        # input size related
        self.patch_size = patch_size
        self.input_sq_size = input_sq_size
        self.pos_embed = PositionEmbedding2D(hidden_size)
        self.rope = RotaryEmbedding(dim=self.hidden_size // self.num_heads)

        # embedding
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
            self.x_embedder = PatchEmbed(patch_size[1], in_channels, hidden_size, bias=True, manual_pad=manual_pad)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fps_embedder = SizeEmbedder(self.hidden_size)
        self.t_block = nn.SequentialCell(nn.SiLU(), nn.Dense(hidden_size, 6 * hidden_size))
        self.skip_y_embedder = skip_y_embedder
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        # spatial blocks
        drop_path = np.linspace(0, self.drop_path, depth)
        self.spatial_blocks = nn.CellList(
            [
                STDiT3DSPBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i].item(),
                    qk_norm=qk_norm,
                    enable_flash_attention=enable_flashattn,
                    enable_layernorm_kernel=enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                )
                for i in range(depth)
            ]
        )

        # temporal blocks
        drop_path = np.linspace(0, self.drop_path, depth)
        self.temporal_blocks = nn.CellList(
            [
                STDiT3DSPBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop_path=drop_path[i].item(),
                    qk_norm=qk_norm,
                    enable_flash_attention=enable_flashattn,
                    enable_layernorm_kernel=enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    # temporal
                    temporal=True,
                    rope=self.rope.rotate_queries_or_keys,
                )
                for i in range(depth)
            ]
        )

        # final layer
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size).item(), self.out_channels)

        self.initialize_weights()
        if only_train_temporal:
            for param in self.get_parameters():
                param.requires_grad = False
            for block in self.temporal_blocks:
                for param in block.get_parameters():
                    param.requires_grad = True

        self._freeze_y_embedder = False
        if freeze_y_embedder:
            self._freeze_y_embedder = True
            self.y_embedder.set_grad(False)  # Pynative: disable memory allocation for gradients
            for param in self.y_embedder.get_parameters():  # turn off explicitly for correct count of trainable params
                param.requires_grad = False

        if use_recompute:
            for blocks in [self.spatial_blocks, self.temporal_blocks]:
                for block in blocks:
                    self.recompute(block)

        if self.enable_sequence_parallelism:
            sp_group = get_sequence_parallel_group()
            logger.info(f"Initialize STDIT-v3 model with dynamic sequence parallel group `{sp_group}`.")
            self.sp_size = get_group_size(sp_group)
            self.split_forward_gather_backward = SplitFowardGatherBackward(dim=1, grad_scale="down", group=sp_group)
            self.gather_forward_split_backward = GatherFowardSplitBackward(dim=1, grad_scale="up", group=sp_group)

        self.is_dynamic_shape = check_dynamic_mode()
        self.chunk = get_chunk_op()

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Dense):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize fps_embedder
        normal_(self.fps_embedder.mlp[0].weight, std=0.02)
        constant_(self.fps_embedder.mlp[0].bias, 0)
        constant_(self.fps_embedder.mlp[2].weight, 0)
        constant_(self.fps_embedder.mlp[2].bias, 0)

        # Initialize temporal blocks
        for block in self.temporal_blocks:
            constant_(block.attn.proj.weight, 0)
            constant_(block.cross_attn.proj.weight, 0)
            constant_(block.mlp.fc2.weight, 0)

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

    def get_pad_num(self, dim_size: int) -> int:
        pad = (self.sp_size - (dim_size % self.sp_size)) % self.sp_size
        return pad

    def construct(
        self,
        x: Tensor,
        timestep: Tensor,
        y: Tensor,
        mask: Optional[Tensor] = None,
        frames_mask: Optional[Tensor] = None,
        fps: Optional[Tensor] = None,
        height: Optional[Tensor] = None,
        width: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        B = x.shape[0]

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.shape
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        if self.is_dynamic_shape:
            # tricky adaptation for dynamic shape in graph mode. Though it also works for static shape, it degrades performance by 50 ms per step.
            base_size = int(round(S ** Tensor(0.5)))
        else:
            base_size = round(S**0.5)

        resolution_sq = (height[0] * width[0]) ** 0.5
        scale = resolution_sq / self.input_sq_size
        # Position embedding doesn't need gradient
        pos_emb = ops.stop_gradient(self.pos_embed(H, W, scale=scale, base_size=base_size))

        # === get timestep embed ===
        t = self.t_embedder(timestep)
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)

        # frames mask branch
        t0_timestep = ops.zeros_like(timestep)
        t0 = self.t_embedder(t0_timestep)
        t0 = t0 + fps
        t0_mlp = self.t_block(t0)

        # === get y embed ===
        if not self.skip_y_embedder:
            y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
            if self._freeze_y_embedder:
                y = ops.stop_gradient(y)
            y = y.squeeze(1).view(1, -1, self.hidden_size)

        # === get x embed ===
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

        # shard over the sequence dim if sp is enabled
        temporal_pad, spatial_pad = None, None
        if self.enable_sequence_parallelism:
            temporal_pad = self.get_pad_num(T)
            spatial_pad = self.get_pad_num(S)

            if temporal_pad > 0:
                x = mint.nn.functional.pad(x, (0, 0, 0, 0, 0, temporal_pad))
                frames_mask = mint.nn.functional.pad(frames_mask, (0, temporal_pad))

            x = self.split_forward_gather_backward(x)
            T = x.shape[1]
            frames_mask = ops.stop_gradient(self.split_forward_gather_backward(frames_mask))

        x = x.reshape(B, T * S, x.shape[-1])  # B T S C -> B (T S) C

        # === blocks ===
        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = spatial_block(x, y, t_mlp, mask, frames_mask, t0_mlp, T, S, spatial_pad, temporal_pad)
            x = temporal_block(x, y, t_mlp, mask, frames_mask, t0_mlp, T, S, spatial_pad, temporal_pad)

        # === final layer ===
        x = self.final_layer(x, t, frames_mask, t0, T, S)

        if self.enable_sequence_parallelism:
            x = x.reshape(B, T, S, -1)
            x = self.gather_forward_split_backward(x)

            if temporal_pad > 0:
                x = x.narrow(1, 0, x.shape[1] - temporal_pad)

            T, S = x.shape[1], x.shape[2]

        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        return x.astype(ms.float32)

    def construct_with_cfg(self, x, timestep, y, cfg_scale, **kwargs):
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = ops.cat([half, half], axis=0)
        if "frames_mask" in kwargs and kwargs["frames_mask"] is not None:
            if len(kwargs["frames_mask"]) != len(x):
                kwargs["frames_mask"] = ops.cat([kwargs["frames_mask"], kwargs["frames_mask"]], axis=0)
        model_out = self(combined, timestep, y, **kwargs)
        model_out = model_out["x"] if isinstance(model_out, dict) else model_out
        pred = self.chunk(model_out, 2, 1)[0]
        pred_cond, pred_uncond = self.chunk(pred, 2, 0)
        v_pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        return ops.cat([v_pred, v_pred], axis=0)

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

    def load_from_checkpoint(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            print(f"WARNING: {ckpt_path} not found. No checkpoint loaded!!")
        else:
            sd = load_checkpoint(ckpt_path)

            regex = re.compile(r"^network\.|\._backbone")
            sd = {regex.sub("", k): v for k, v in sd.items()}

            # PixArt-Σ: rename 'blocks' to 'spatial_blocks'
            regex = re.compile(r"^blocks")
            sd = {regex.sub("spatial_blocks", k): v for k, v in sd.items()}

            # load conv3d weight from pretrained conv2d or dense layer
            key_3d = "x_embedder.proj.weight"
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

            m, u = load_param_into_net(self, sd)
            print("net param not load: ", m, len(m))
            print("ckpt param not load: ", u, len(u))


def STDiT3_XL_2_DSP(from_pretrained=None, **kwargs):
    # DEBUG only
    # model = STDiT3_DSP(depth=1, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    model = STDiT3_DSP(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(from_pretrained, model)
    return model


def STDiT3_3B_2_DSP(from_pretrained=None, **kwargs):
    model = STDiT3_DSP(depth=28, hidden_size=1872, patch_size=(1, 2, 2), num_heads=26, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(from_pretrained, model)
    return model
