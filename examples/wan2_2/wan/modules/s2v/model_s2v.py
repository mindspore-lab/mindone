# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from copy import deepcopy
from typing import Any, List, Optional, Tuple

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn
from mindspore.common.initializer import XavierUniform, initializer

from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.layers_compat import unflatten
from mindone.diffusers.models.modeling_utils import ModelMixin
from mindone.models.utils import normal_, xavier_uniform_, zeros_

from ...distributed.sequence_parallel import distributed_attention, gather_forward, get_rank, get_world_size
from ...utils.amp import autocast
from ..model import (
    Head,
    WanAttentionBlock,
    WanLayerNorm,
    WanSelfAttention,
    complex_mult,
    flash_attention,
    rope_params,
    sinusoidal_embedding_1d,
)
from .audio_utils import AudioInjector_WAN, CausalAudioEncoder
from .motioner import FramePackMotioner, MotionerTransformers
from .s2v_utils import rope_precompute


def zero_module(module: nn.Cell) -> None:
    """
    Zero out the parameters of a module.
    """
    for _, p in module.parameters_and_names():
        zeros_(p)


def mindspore_dfs(model: nn.Cell, parent_name: str = "root") -> Tuple[List[nn.Cell], List[str]]:
    module_names, modules = [], []
    current_name = parent_name if parent_name else "root"
    module_names.append(current_name)
    modules.append(model)

    for name, child in model.name_cells().items():
        if parent_name:
            child_name = f"{parent_name}.{name}"
        else:
            child_name = name
        child_modules, child_names = mindspore_dfs(child, child_name)
        module_names += child_names
        modules += child_modules
    return modules, module_names


def rope_apply(x: ms.Tensor, grid_sizes: ms.Tensor, freqs: ms.Tensor, start: Optional[Any] = None) -> ms.Tensor:
    dtype = x.dtype
    x = x.to(ms.float32)
    n, _ = x.shape[2], x.shape[3] // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.shape[1]
        x_i = x[i, :s].to(ms.float32).reshape(s, n, -1, 2)
        freqs_i = freqs[i, :s]
        # apply rotary embedding
        x_i = complex_mult(x_i, freqs_i).flatten(2)
        x_i = mint.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return mint.stack(output).to(dtype)


def rope_apply_usp(x: ms.Tensor, grid_sizes: Any, freqs: ms.Tensor) -> ms.Tensor:
    dtype = x.dtype
    x = x.to(ms.float32)
    s, n, _ = x.shape[1], x.shape[2], x.shape[3] // 2
    # loop over samples
    output = []
    for i, _ in enumerate(x):
        s = x.shape[1]
        # precompute multipliers
        x_i = x[i, :s].to(ms.float32).reshape(s, n, -1, 2)
        freqs_i = freqs[i]
        freqs_i_rank = freqs_i
        x_i = complex_mult(x_i, freqs_i_rank).flatten(2)
        x_i = mint.cat([x_i, x[i, s:]])
        # append to collection
        output.append(x_i)
    return mint.stack(output).to(dtype)


def sp_attn_forward_s2v(
    self: WanSelfAttention,
    x: ms.Tensor,
    seq_lens: ms.Tensor,
    grid_sizes: ms.Tensor,
    freqs: ms.Tensor,
    dtype: Any = ms.bfloat16,
) -> ms.Tensor:
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    half_dtypes = (ms.float16, ms.bfloat16)

    def half(x: ms.Tensor) -> ms.Tensor:
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    q, k, v = qkv_fn(x)
    q = rope_apply_usp(q, grid_sizes, freqs)
    k = rope_apply_usp(k, grid_sizes, freqs)

    x = distributed_attention(
        half(q),
        half(k),
        half(v),
        seq_lens,
        window_size=self.window_size,
    ).to(q.dtype)

    # output
    x = x.flatten(2)
    x = self.o(x)
    return x


class Head_S2V(Head):
    def construct(self, x: ms.Tensor, e: ms.Tensor) -> ms.Tensor:
        """
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == ms.float32
        with autocast(dtype=ms.float32):
            e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
            x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x


class WanS2VSelfAttention(WanSelfAttention):
    def construct(self, x: ms.Tensor, seq_lens: ms.Tensor, grid_sizes: ms.Tensor, freqs: ms.Tensor) -> ms.Tensor:
        """
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor, ms.Tensor]:
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanS2VAttentionBlock(WanAttentionBlock):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        dtype: Any = ms.float32,
    ):
        super().__init__(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, dtype=dtype)
        self.self_attn = WanS2VSelfAttention(dim, num_heads, window_size, qk_norm, eps, dtype=dtype)

    def construct(
        self,
        x: ms.Tensor,
        e: ms.Tensor,
        seq_lens: ms.Tensor,
        grid_sizes: ms.Tensor,
        freqs: ms.Tensor,
        context: ms.Tensor,
        context_lens: Optional[ms.Tensor],
    ) -> ms.Tensor:
        assert e[0].dtype == ms.float32
        seg_idx = e[1].item()
        seg_idx = min(max(0, seg_idx), x.shape[1])
        seg_idx = [0, seg_idx, x.shape[1]]
        e = e[0]
        modulation = self.modulation.unsqueeze(2)
        with autocast(dtype=ms.float32):
            e = (modulation + e).chunk(6, dim=1)
        assert e[0].dtype == ms.float32

        e = [element.squeeze(1) for element in e]
        norm_x = self.norm1(x).float()
        parts = []
        for i in range(2):
            parts.append(norm_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e[1][:, i : i + 1]) + e[0][:, i : i + 1])
        norm_x = mint.cat(parts, dim=1)
        # self-attention
        dtype = x.dtype
        y = self.self_attn(norm_x.to(dtype), seq_lens, grid_sizes, freqs)
        with autocast(dtype=ms.float32):
            z = []
            for i in range(2):
                z.append(y[:, seg_idx[i] : seg_idx[i + 1]] * e[2][:, i : i + 1])
            y = mint.cat(z, dim=1)
            x = x + y
        x = x.to(dtype)

        # cross-attention & ffn function
        def cross_attn_ffn(
            x: ms.Tensor, context: ms.Tensor, context_lens: Optional[ms.Tensor], e: ms.Tensor
        ) -> ms.Tensor:
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            norm2_x = self.norm2(x).float()
            parts = []
            for i in range(2):
                parts.append(norm2_x[:, seg_idx[i] : seg_idx[i + 1]] * (1 + e[4][:, i : i + 1]) + e[3][:, i : i + 1])
            norm2_x = mint.cat(parts, dim=1)
            y = self.ffn(norm2_x.to(dtype))
            with autocast(dtype=ms.float32):
                z = []
                for i in range(2):
                    z.append(y[:, seg_idx[i] : seg_idx[i + 1]] * e[5][:, i : i + 1])
                y = mint.cat(z, dim=1)
                x = x + y
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x.to(dtype)


class WanModel_S2V(ModelMixin, ConfigMixin):
    ignore_for_config = ["args", "kwargs", "patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanS2VAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        cond_dim: int = 0,
        audio_dim: int = 5120,
        num_audio_token: int = 4,
        enable_adain: bool = False,
        adain_mode: str = "attn_norm",
        audio_inject_layers: List[int] = [0, 4, 8, 12, 16, 20, 24, 27],
        zero_init: bool = False,
        zero_timestep: bool = False,
        enable_motioner: bool = True,
        add_last_motion: bool = True,
        enable_tsm: bool = False,
        trainable_token_pos_emb: bool = False,
        motion_token_num: int = 1024,
        enable_framepack: bool = False,
        framepack_drop_mode: str = "drop",
        model_type: str = "s2v",
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_len: int = 512,
        in_dim: int = 16,
        dim: int = 2048,
        ffn_dim: int = 8192,
        freq_dim: int = 256,
        text_dim: int = 4096,
        out_dim: int = 16,
        num_heads: int = 16,
        num_layers: int = 32,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        cross_attn_norm: bool = True,
        eps: float = 1e-6,
        dtype: Any = ms.float32,
    ):
        super().__init__()

        assert model_type == "s2v"
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = mint.nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size, dtype=dtype)
        self.text_embedding = nn.SequentialCell(
            mint.nn.Linear(text_dim, dim, dtype=dtype),
            mint.nn.GELU(approximate="tanh"),
            mint.nn.Linear(dim, dim, dtype=dtype),
        )

        self.time_embedding = nn.SequentialCell(
            mint.nn.Linear(freq_dim, dim, dtype=dtype), mint.nn.SiLU(), mint.nn.Linear(dim, dim, dtype=dtype)
        )
        self.time_projection = nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(dim, dim * 6, dtype=dtype))

        # blocks
        self.blocks = nn.CellList(
            [
                WanS2VAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm, cross_attn_norm, eps, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head_S2V(dim, out_dim, patch_size, eps, dtype=dtype)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = mint.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

        # initialize weights
        self.init_weights()

        self.use_context_parallel = False  # will modify in _configure_model func

        if cond_dim > 0:
            self.cond_encoder = mint.nn.Conv3d(
                cond_dim, self.dim, kernel_size=self.patch_size, stride=self.patch_size, dtype=dtype
            )
        self.enbale_adain = enable_adain
        self.casual_audio_encoder = CausalAudioEncoder(
            dim=audio_dim, out_dim=self.dim, num_token=num_audio_token, need_global=enable_adain, dtype=dtype
        )
        all_modules, all_modules_names = mindspore_dfs(self.blocks, parent_name="root.transformer_blocks")
        self.audio_injector = AudioInjector_WAN(
            all_modules,
            all_modules_names,
            dim=self.dim,
            num_heads=self.num_heads,
            inject_layer=audio_inject_layers,
            root_net=self,
            enable_adain=enable_adain,
            adain_dim=self.dim,
            need_adain_ont=adain_mode != "attn_norm",
            dtype=dtype,
        )
        self.adain_mode = adain_mode

        self.trainable_cond_mask = mint.nn.Embedding(3, self.dim, dtype=dtype)

        if zero_init:
            self.zero_init_weights()

        self.zero_timestep = zero_timestep  # Whether to assign 0 value timestep to ref/motion

        # init motioner
        if enable_motioner and enable_framepack:
            raise ValueError(
                "enable_motioner and enable_framepack are mutually exclusive, please set one of them to False"
            )
        self.enable_motioner = enable_motioner
        self.add_last_motion = add_last_motion
        if enable_motioner:
            motioner_dim = 2048
            self.motioner = MotionerTransformers(
                patch_size=(2, 4, 4),
                dim=motioner_dim,
                ffn_dim=motioner_dim,
                freq_dim=256,
                out_dim=16,
                num_heads=16,
                num_layers=13,
                window_size=(-1, -1),
                qk_norm=True,
                cross_attn_norm=False,
                eps=1e-6,
                motion_token_num=motion_token_num,
                enable_tsm=enable_tsm,
                motion_stride=4,
                expand_ratio=2,
                trainable_token_pos_emb=trainable_token_pos_emb,
                dtype=dtype,
            )
            self.zip_motion_out = nn.SequentialCell(
                WanLayerNorm(motioner_dim, dtype=dtype),
                mint.nn.Linear(motioner_dim, self.dim, dtype=dtype),
            )
            zero_module(self.zip_motion_out[1])

            self.trainable_token_pos_emb = trainable_token_pos_emb
            if trainable_token_pos_emb:
                d = self.dim // self.num_heads
                x = mint.zeros([1, motion_token_num, self.num_heads, d])
                x[..., ::2] = 1

                gride_sizes = [
                    [
                        ms.tensor([0, 0, 0]).unsqueeze(0).repeat(1, 1),
                        ms.tensor([1, self.motioner.motion_side_len, self.motioner.motion_side_len])
                        .unsqueeze(0)
                        .repeat(1, 1),
                        ms.tensor([1, self.motioner.motion_side_len, self.motioner.motion_side_len])
                        .unsqueeze(0)
                        .repeat(1, 1),
                    ]
                ]
                token_freqs = rope_apply(x, gride_sizes, self.freqs)
                token_freqs = token_freqs[0, :, 0].reshape(motion_token_num, -1, 2)
                token_freqs = token_freqs * 0.01
                self.token_freqs = ms.Parameter(token_freqs)

        self.enable_framepack = enable_framepack
        if enable_framepack:
            self.frame_packer = FramePackMotioner(
                inner_dim=self.dim,
                num_heads=self.num_heads,
                zip_frame_buckets=[1, 2, 16],
                drop_mode=framepack_drop_mode,
                dtype=dtype,
            )

    def zero_init_weights(self) -> None:
        zero_module(self.trainable_cond_mask)
        if hasattr(self, "cond_encoder"):
            zero_module(self.cond_encoder)

        for i in range(len(self.audio_injector.injector)):
            zero_module(self.audio_injector.injector[i].o)
            if self.enbale_adain:
                zero_module(self.audio_injector.injector_adain_layers[i].linear)

    def process_motion(
        self, motion_latents: List[ms.Tensor], drop_motion_frames: bool = False
    ) -> Tuple[List[Any], List[Any]]:
        if drop_motion_frames or motion_latents[0].shape[1] == 0:
            return [], []
        self.lat_motion_frames = motion_latents[0].shape[1]
        mot = [self.patch_embedding(m.unsqueeze(0)) for m in motion_latents]
        batch_size = len(mot)

        mot_remb = []
        flattern_mot = []
        for bs in range(batch_size):
            height, width = mot[bs].shape[3], mot[bs].shape[4]
            flat_mot = mot[bs].flatten(2).transpose(1, 2).contiguous()
            motion_grid_sizes = [
                [
                    ms.tensor([-self.lat_motion_frames, 0, 0]).unsqueeze(0).repeat(1, 1),
                    ms.tensor([0, height, width]).unsqueeze(0).repeat(1, 1),
                    ms.tensor([self.lat_motion_frames, height, width]).unsqueeze(0).repeat(1, 1),
                ]
            ]
            motion_rope_emb = rope_precompute(
                flat_mot.view(1, flat_mot.shape[1], self.num_heads, self.dim // self.num_heads),
                motion_grid_sizes,
                self.freqs,
                start=None,
            )
            mot_remb.append(motion_rope_emb)
            flattern_mot.append(flat_mot)
        return flattern_mot, mot_remb

    def process_motion_frame_pack(
        self, motion_latents: Any, drop_motion_frames: bool = False, add_last_motion: int = 2
    ) -> Tuple[List[Any], List[Any]]:
        flattern_mot, mot_remb = self.frame_packer(motion_latents, add_last_motion)
        if drop_motion_frames:
            return [m[:, :0] for m in flattern_mot], [m[:, :0] for m in mot_remb]
        else:
            return flattern_mot, mot_remb

    def process_motion_transformer_motioner(
        self, motion_latents: List[ms.Tensor], drop_motion_frames: bool = False, add_last_motion: bool = True
    ) -> Tuple[List[Any], List[Any]]:
        batch_size, height, width = (
            len(motion_latents),
            motion_latents[0].shape[2] // self.patch_size[1],
            motion_latents[0].shape[3] // self.patch_size[2],
        )

        freqs = self.freqs
        if self.trainable_token_pos_emb:
            with autocast(dtype=ms.float32):
                token_freqs = self.token_freqs.to(ms.float32)
                token_freqs = token_freqs / token_freqs.norm(dim=-1, keepdim=True)
                freqs = [freqs, token_freqs]

        if not drop_motion_frames and add_last_motion:
            last_motion_latent = [u[:, -1:] for u in motion_latents]
            last_mot = [self.patch_embedding(m.unsqueeze(0)) for m in last_motion_latent]
            last_mot = [m.flatten(2).transpose(1, 2) for m in last_mot]
            last_mot = mint.cat(last_mot)
            gride_sizes = [
                [
                    ms.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                    ms.tensor([0, height, width]).unsqueeze(0).repeat(batch_size, 1),
                    ms.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
                ]
            ]
        else:
            last_mot = mint.zeros([batch_size, 0, self.dim], dtype=motion_latents[0].dtype)
            gride_sizes = []

        zip_motion = self.motioner(motion_latents)
        zip_motion = self.zip_motion_out(zip_motion)
        if drop_motion_frames:
            zip_motion = zip_motion * 0.0
        zip_motion_grid_sizes = [
            [
                ms.tensor([-1, 0, 0]).unsqueeze(0).repeat(batch_size, 1),
                ms.tensor([0, self.motioner.motion_side_len, self.motioner.motion_side_len])
                .unsqueeze(0)
                .repeat(batch_size, 1),
                ms.tensor([1 if not self.trainable_token_pos_emb else -1, height, width])
                .unsqueeze(0)
                .repeat(batch_size, 1),
            ]
        ]

        mot = mint.cat([last_mot, zip_motion], dim=1)
        gride_sizes = gride_sizes + zip_motion_grid_sizes

        motion_rope_emb = rope_precompute(
            mot.view(batch_size, mot.shape[1], self.num_heads, self.dim // self.num_heads),
            gride_sizes,
            freqs,
            start=None,
        )
        return [m.unsqueeze(0) for m in mot], [r.unsqueeze(0) for r in motion_rope_emb]

    def inject_motion(
        self,
        x: List[ms.Tensor],
        seq_lens: ms.Tensor,
        rope_embs: List[Any],
        mask_input: List[ms.Tensor],
        motion_latents: List[ms.Tensor],
        drop_motion_frames: bool = False,
        add_last_motion: bool = True,
    ) -> Tuple[List[ms.Tensor], ms.Tensor, List[Any], ms.Tensor]:
        # inject the motion frames token to the hidden states
        if self.enable_motioner:
            mot, mot_remb = self.process_motion_transformer_motioner(
                motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion
            )
        elif self.enable_framepack:
            mot, mot_remb = self.process_motion_frame_pack(
                motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=add_last_motion
            )
        else:
            mot, mot_remb = self.process_motion(motion_latents, drop_motion_frames=drop_motion_frames)

        if len(mot) > 0:
            x = [mint.cat([u, m.to(u.dtype)], dim=1) for u, m in zip(x, mot)]
            seq_lens = seq_lens + ms.tensor([r.shape[1] for r in mot], dtype=ms.int64)
            rope_embs = [mint.cat([u, m], dim=1) for u, m in zip(rope_embs, mot_remb)]
            mask_input = [
                mint.cat([m, 2 * mint.ones([1, u.shape[1] - m.shape[1]], dtype=m.dtype)], dim=1)
                for m, u in zip(mask_input, x)
            ]
        return x, seq_lens, rope_embs, mask_input

    def after_transformer_block(self, block_idx: int, hidden_states: ms.Tensor) -> ms.Tensor:
        if block_idx in self.audio_injector.injected_block_id.keys():
            audio_attn_id = self.audio_injector.injected_block_id[block_idx]
            audio_emb = self.merged_audio_emb  # b f n c
            num_frames = audio_emb.shape[1]

            if self.use_context_parallel:
                hidden_states = gather_forward(hidden_states, dim=1)

            input_hidden_states = hidden_states[:, : self.original_seq_len].clone()  # b (f h w) c
            # b (t n) c -> (b t) n c
            input_hidden_states = input_hidden_states.reshape(
                input_hidden_states.shape[0], num_frames, -1, input_hidden_states.shape[2]
            )
            input_hidden_states = input_hidden_states.reshape(-1, *input_hidden_states.shape[2:])

            if self.enbale_adain and self.adain_mode == "attn_norm":
                audio_emb_global = self.audio_emb_global
                # b t n c -> (b t) n c
                audio_emb_global = audio_emb_global.reshape(-1, *audio_emb_global.shape[2:])
                adain_hidden_states = self.audio_injector.injector_adain_layers[audio_attn_id](
                    input_hidden_states, temb=audio_emb_global[:, 0]
                )
                attn_hidden_states = adain_hidden_states
            else:
                attn_hidden_states = self.audio_injector.injector_pre_norm_feat[audio_attn_id](input_hidden_states)
            # "b t n c -> (b t) n c"
            audio_emb = audio_emb.reshape(-1, *audio_emb.shape[2:])
            attn_audio_emb = audio_emb
            residual_out = self.audio_injector.injector[audio_attn_id](
                x=attn_hidden_states,
                context=attn_audio_emb,
                context_lens=mint.ones(attn_hidden_states.shape[0], dtype=ms.int64) * attn_audio_emb.shape[1],
            )
            # (b t) n c -> b (t n) c
            residual_out = residual_out.reshape(-1, num_frames, *residual_out.shape[1:])
            residual_out = residual_out.reshape(residual_out.shape[0], -1, residual_out.shape[3])
            hidden_states[:, : self.original_seq_len] = hidden_states[:, : self.original_seq_len] + residual_out

            if self.use_context_parallel:
                hidden_states = mint.chunk(hidden_states, get_world_size(), dim=1)[get_rank()]

        return hidden_states

    def construct(
        self,
        x: List[ms.Tensor],
        t: ms.Tensor,
        context: List[ms.Tensor],
        seq_len: Any,
        ref_latents: List[ms.Tensor],
        motion_latents: List[ms.Tensor],
        cond_states: List[ms.Tensor],
        audio_input: Optional[ms.Tensor] = None,
        motion_frames: List[int] = [17, 5],
        add_last_motion: int = 2,
        drop_motion_frames: bool = False,
        *extra_args,
        **extra_kwargs,
    ) -> List[ms.Tensor]:
        """
        x:                  A list of videos each with shape [C, T, H, W].
        t:                  [B].
        context:            A list of text embeddings each with shape [L, C].
        seq_len:            A list of video token lens, no need for this model.
        ref_latents         A list of reference image for each video with shape [C, 1, H, W].
        motion_latents      A list of  motion frames for each video with shape [C, T_m, H, W].
        cond_states         A list of condition frames (i.e. pose) each with shape [C, T, H, W].
        audio_input         The input audio embedding [B, num_wav2vec_layer, C_a, T_a].
        motion_frames       The number of motion frames and motion latents frames encoded by vae, i.e. [17, 5]
        add_last_motion     For the motioner, if add_last_motion > 0, it means that the most recent frame (i.e., the last frame) will be added.
                            For frame packing, the behavior depends on the value of add_last_motion:
                            add_last_motion = 0: Only the farthest part of the latent (i.e., clean_latents_4x) is included.
                            add_last_motion = 1: Both clean_latents_2x and clean_latents_4x are included.
                            add_last_motion = 2: All motion-related latents are used.
        drop_motion_frames  Bool, whether drop the motion frames info
        """
        add_last_motion = self.add_last_motion * add_last_motion
        audio_input = mint.cat([audio_input[..., 0:1].repeat(1, 1, 1, motion_frames[0]), audio_input], dim=-1)
        audio_emb_res = self.casual_audio_encoder(audio_input)
        if self.enbale_adain:
            audio_emb_global, audio_emb = audio_emb_res
            self.audio_emb_global = audio_emb_global[:, motion_frames[1] :].clone()
        else:
            audio_emb = audio_emb_res
        self.merged_audio_emb = audio_emb[:, motion_frames[1] :, :]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        # cond states
        cond = [self.cond_encoder(c.unsqueeze(0)) for c in cond_states]
        x = [x_ + pose for x_, pose in zip(x, cond)]

        grid_sizes = mint.stack([ms.tensor(u.shape[2:], dtype=ms.int64) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = ms.tensor([u.shape[1] for u in x], dtype=ms.int64)

        original_grid_sizes = deepcopy(grid_sizes)
        grid_sizes = [[mint.zeros_like(grid_sizes), grid_sizes, grid_sizes]]

        # ref and motion
        self.lat_motion_frames = motion_latents[0].shape[1]

        ref = [self.patch_embedding(r.unsqueeze(0)) for r in ref_latents]
        batch_size = len(ref)
        height, width = ref[0].shape[3], ref[0].shape[4]
        ref_grid_sizes = [
            [
                ms.tensor([30, 0, 0]).unsqueeze(0).repeat(batch_size, 1),  # the start index
                ms.tensor([31, height, width]).unsqueeze(0).repeat(batch_size, 1),  # the end index
                ms.tensor([1, height, width]).unsqueeze(0).repeat(batch_size, 1),
            ]  # the range
        ]

        ref = [r.flatten(2).transpose(1, 2) for r in ref]  # r: 1 c f h w
        self.original_seq_len = seq_lens[0]

        seq_lens = seq_lens + ms.tensor([r.shape[1] for r in ref], dtype=ms.int64)

        grid_sizes = grid_sizes + ref_grid_sizes

        x = [mint.cat([u, r.to(u.dtype)], dim=1) for u, r in zip(x, ref)]

        # Initialize masks to indicate noisy latent, ref latent, and motion latent.
        # However, at this point, only the first two (noisy and ref latents) are marked;
        # the marking of motion latent will be implemented inside `inject_motion`.
        mask_input = [mint.zeros([1, u.shape[1]], dtype=ms.int64) for u in x]
        for i in range(len(mask_input)):
            mask_input[i][:, self.original_seq_len :] = 1

        # compute the rope embeddings for the input
        x = mint.cat(x)
        b, s, n, d = x.shape[0], x.shape[1], self.num_heads, self.dim // self.num_heads
        self.pre_compute_freqs = rope_precompute(x.view(b, s, n, d), grid_sizes, self.freqs, start=None)

        x = [u.unsqueeze(0) for u in x]
        self.pre_compute_freqs = [u.unsqueeze(0) for u in self.pre_compute_freqs]

        x, seq_lens, self.pre_compute_freqs, mask_input = self.inject_motion(
            x,
            seq_lens,
            self.pre_compute_freqs,
            mask_input,
            motion_latents,
            drop_motion_frames=drop_motion_frames,
            add_last_motion=add_last_motion,
        )

        x = mint.cat(x, dim=0)
        self.pre_compute_freqs = mint.cat(self.pre_compute_freqs, dim=0)
        mask_input = mint.cat(mask_input, dim=0)

        x = x + self.trainable_cond_mask(mask_input).to(x.dtype)

        # time embeddings
        if self.zero_timestep:
            t = mint.cat([t, mint.zeros([1], dtype=t.dtype)])
        with autocast(dtype=ms.float32):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
            e0 = unflatten(self.time_projection(e), 1, (6, self.dim))
            assert e.dtype == ms.float32 and e0.dtype == ms.float32

        if self.zero_timestep:
            e = e[:-1]
            zero_e0 = e0[-1:]
            e0 = e0[:-1]
            e0 = mint.cat([e0.unsqueeze(2), zero_e0.unsqueeze(2).repeat(e0.shape[0], 1, 1, 1)], dim=2)
            e0 = [e0, self.original_seq_len]
        else:
            e0 = e0.unsqueeze(2).repeat(1, 1, 2, 1)
            e0 = [e0, 0]

        # context
        context_lens = None
        context = [u.to(self.dtype) for u in context]
        context = self.text_embedding(
            mint.stack([mint.cat([u, u.new_zeros((self.text_len - u.shape[0], u.shape[1]))]) for u in context])
        )

        if self.use_context_parallel:
            # sharded tensors for long context attn
            sp_rank = get_rank()
            x = mint.chunk(x, get_world_size(), dim=1)
            sq_size = [u.shape[1] for u in x]
            sq_start_size = sum(sq_size[:sp_rank])
            x = x[sp_rank]
            # Confirm the application range of the time embedding in e0[0] for each sequence:
            # - For tokens before seg_id: apply e0[0][:, :, 0]
            # - For tokens after seg_id: apply e0[0][:, :, 1]
            seg_idx = e0[1] - sq_start_size
            e0[1] = seg_idx

            self.pre_compute_freqs = mint.chunk(self.pre_compute_freqs, get_world_size(), dim=1)
            self.pre_compute_freqs = self.pre_compute_freqs[sp_rank]

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.pre_compute_freqs,
            context=context,
            context_lens=context_lens,
        )

        x = x.to(self.dtype)
        for idx, block in enumerate(self.blocks):
            x = block(x, **kwargs)
            x = self.after_transformer_block(idx, x)

        # Context Parallel
        if self.use_context_parallel:
            x = gather_forward(x.contiguous(), dim=1)
        # unpatchify
        x = x[:, : self.original_seq_len]
        # head
        x = self.head(x, e)
        x = self.unpatchify(x, original_grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x: List[ms.Tensor], grid_sizes: ms.Tensor) -> List[ms.Tensor]:
        """
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = mint.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self) -> None:
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for _, m in self.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

        # init embeddings
        patch_embedding_shape = self.patch_embedding.weight.shape
        patch_embedding_shape_flatten = (patch_embedding_shape[0], math.prod(patch_embedding_shape[1:]))
        data = initializer(
            XavierUniform(), patch_embedding_shape_flatten, self.patch_embedding.weight.dtype
        ).init_data()
        self.patch_embedding.weight.set_data(data.reshape(patch_embedding_shape))
        for _, m in self.text_embedding.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                normal_(m.weight, std=0.02)
        for _, m in self.time_embedding.cells_and_names():
            if isinstance(m, mint.nn.Linear):
                normal_(m.weight, std=0.02)

        # init output layer
        zeros_(self.head.head.weight)
