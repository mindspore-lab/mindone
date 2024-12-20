import logging
import math
from functools import partial

import numpy as np

import mindspore as ms
from mindspore import Tensor, mint, nn, ops, recompute
from mindspore.common.initializer import Constant, TruncatedNormal, initializer

from mindone.models.modules.flash_attention import MSFlashAttention as FlashAttention

from .helpers import to_2tuple
from .pos_embed import get_3d_sincos_pos_embed, interpolate_pos_embed_internvideo2

logger = logging.getLogger(__name__)


def trunc_normal_(tensor: ms.Parameter, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0):
    tensor.set_data(initializer(TruncatedNormal(std, mean, a, b), tensor.shape, tensor.dtype))


class DropPath(nn.Cell):
    def __init__(
        self,
        drop_prob: float = 0.0,
        scale_by_keep: bool = True,
    ) -> None:
        super().__init__()
        self.keep_prob = 1.0 - drop_prob
        self.scale_by_keep = scale_by_keep
        self.dropout = nn.Dropout(p=drop_prob)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        if self.keep_prob == 1.0 or not self.training:
            return x
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.dropout(ops.ones(shape))
        if not self.scale_by_keep:
            random_tensor = ops.mul(random_tensor, self.keep_prob)
        return x * random_tensor


class LayerNorm(nn.LayerNorm):
    """subclass torch's LayerNorm to handle fp16."""

    def construct(self, x: Tensor):
        orig_type = x.dtype
        ret = super().construct(ops.cast(x, ms.float32))
        return ops.cast(ret, orig_type)


class CrossAttention(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        out_dim=None,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5
        assert all_head_dim == dim

        self.q = nn.Dense(dim, all_head_dim, has_bias=False)
        self.k = nn.Dense(dim, all_head_dim, has_bias=False)
        self.v = nn.Dense(dim, all_head_dim, has_bias=False)

        if qkv_bias:
            self.q_bias = ms.Parameter(ops.zeros(all_head_dim))
            self.k_bias = ms.Parameter(ops.zeros(all_head_dim))
            self.v_bias = ms.Parameter(ops.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def construct(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]

        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias

        q = mint.nn.functional.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)

        k = mint.nn.functional.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        v = mint.nn.functional.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)

        q = q * self.scale
        attn = q @ k.swapaxes(-2, -1)  # (B, N_head, N_q, N_k)

        attn = ops.softmax(attn.astype(ms.float32), axis=-1).astype(attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v).swapaxes(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentiveBlock(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=LayerNorm,
        attn_head_dim=None,
        out_dim=None,
    ):
        super().__init__()

        self.norm1_q = norm_layer((dim,))
        self.norm1_k = norm_layer((dim,))
        self.norm1_v = norm_layer((dim,))
        self.cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            out_dim=out_dim,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def construct(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)

        return x


class AttentionPoolingBlock(AttentiveBlock):
    def construct(self, x):
        x_q = x.mean(1, keep_dims=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().construct(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class RMSNorm(nn.Cell):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = ms.Parameter(np.ones(hidden_size).astype(np.float32))
        self.variance_epsilon = eps

    def construct(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(ms.float32)
        hidden_states = ops.rms_norm(hidden_states, self.weight, self.variance_epsilon)[0]
        return hidden_states.to(input_dtype)


class LayerScale(nn.Cell):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = ms.Parameter(init_values * ops.ones(dim))
        self.force_fp32 = force_fp32

    def construct(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul(self.gamma) if self.inplace else x * self.gamma
            return out


class Attention(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        use_flash_attn=False,
        causal=False,
        norm_layer=LayerNorm,
        qk_normalization=False,
        use_fused_rmsnorm=False,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Dense(dim, dim * 3, has_bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Dense(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)

        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)

        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer((dim,)) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer((dim,)) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def _naive_attn(self, x):
        B, N, C = x.shape
        # print(x.shape, torch.cuda.memory_allocated(), torch.cuda.memory_allocated())
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.swapaxes(1, 2).flatten(start_dim=-2, end_dim=-1)).view(B_, N_, H_, D_).swapaxes(1, 2)
            k = self.k_norm(k.swapaxes(1, 2).flatten(start_dim=-2, end_dim=-1)).view(B_, N_, H_, D_).swapaxes(1, 2)

        attn = (q * self.scale) @ k.swapaxes(-2, -1)
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = ops.softmax(attn.astype(ms.float32), axis=-1).astype(attn.dtype)
        attn = self.attn_drop(attn)
        # print(torch.cuda.memory_allocated(), torch.cuda.memory_allocated())
        x = (attn @ v).swapaxes(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        qkv = self.qkv(x)
        # qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        b, s, _ = qkv.shape
        qkv = qkv.reshape(b, s, 3, self.num_heads, -1)

        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(start_dim=-2, end_dim=-1))[0].view(q.shape)
                k = self.k_norm(k.flatten(start_dim=-2, end_dim=-1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(start_dim=-2, end_dim=-1)).view(q.shape)
                k = self.k_norm(k.flatten(start_dim=-2, end_dim=-1)).view(k.shape)
            qkv = ops.stack([q, k, v], dim=2)

        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        # outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        b_, s_, _, _ = context.shape
        context = context.reshape(b_, s_, -1)
        outs = self.proj(context)
        outs = self.proj_drop(outs)
        return outs

    def construct(self, x):
        x = self._naive_attn(x) if not self.use_flash_attn else self._flash_attn(x)
        return x


class Mlp(nn.Cell):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Dense(in_features, hidden_features, has_bias=bias[0])
        self.act = act_layer(approximate=False)
        self.drop1 = nn.Dropout(p=drop_probs[0])
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=bias[1])
        self.drop2 = nn.Dropout(p=drop_probs[1])

    def construct(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Cell):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=LayerNorm,
        use_flash_attn=False,
        use_fused_mlp=False,
        fused_mlp_heuristic=1,
        with_cp=False,
        qk_normalization=False,
        layerscale_no_force_fp32=False,
        use_fused_rmsnorm=False,
    ):
        super().__init__()

        self.norm1 = norm_layer((dim,))
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            use_flash_attn=use_flash_attn,
            causal=False,
            norm_layer=norm_layer,
            qk_normalization=qk_normalization,
            use_fused_rmsnorm=use_fused_rmsnorm,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values, force_fp32=(not layerscale_no_force_fp32))
            if init_values
            else nn.Identity()
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer((dim,))
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            raise Exception("Sorry, FusedMLP is not supported yet.")
            # self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = (
            LayerScale(dim, init_values=init_values, force_fp32=(not layerscale_no_force_fp32))
            if init_values
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm

    def construct(self, x, residual=None):
        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x

        return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Cell):
    """3D Image to Patch Embedding"""

    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_frames=8, tubelet_size=1, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            num_frames // tubelet_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )  # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.num_img_patches = self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
            stride=(tubelet_size, patch_size[0], patch_size[1]),
            has_bias=True,
        )
        self.norm = norm_layer((embed_dim,)) if norm_layer else nn.Identity()

    def construct(self, x):
        x = self.proj(x)
        x = x.flatten(start_dim=3).permute(0, 2, 3, 1)  # B x C x T x HW => B x T x HW x C
        x = self.norm(x)
        return x


class Linear_Decoder(nn.Cell):
    def __init__(self, in_channels=1408, out_channels=3200, norm_layer=LayerNorm, clip_norm_type="l2"):
        super().__init__()
        self.clip_norm_type = clip_norm_type
        logger.info(f"Normalization Type: {clip_norm_type}")

        self.head = nn.Dense(in_channels, out_channels)
        self.norm = norm_layer((out_channels,))

    def construct(self, x):
        x = self.norm(self.head(x))

        if self.clip_norm_type == "l2":
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.clip_norm_type == "none":
            pass
        else:
            raise NotImplementedError

        return x


class PretrainInternVideo2(nn.Cell):
    def __init__(
        self,
        in_chans: int = 3,
        patch_size: int = 14,
        img_size: int = 224,
        qkv_bias: bool = False,
        drop_path_rate: float = 0.25,
        embed_dim: int = 1408,
        num_heads: int = 16,
        mlp_ratio: float = 48 / 11,
        init_values: float = 1e-5,
        qk_normalization: bool = True,
        depth: int = 40,
        use_flash_attn: bool = True,
        use_fused_rmsnorm: bool = True,
        use_fused_mlp: bool = True,
        fused_mlp_heuristic: int = 1,
        attn_pool_num_heads: int = 16,
        clip_embed_dim: int = 768,
        layerscale_no_force_fp32: bool = False,
        num_frames: int = 8,
        tubelet_size: int = 1,
        sep_pos_embed: bool = False,
        sep_image_video_pos_embed: bool = False,
        use_checkpoint: bool = False,
        checkpoint_num: int = 0,
        # for unmasked teacher
        clip_teacher_embed_dim: int = 3200,
        clip_teacher_final_dim: int = 768,  # if 0, not distill final features
        clip_norm_type: str = "l2",
        clip_return_layer: int = 1,
        clip_student_return_interval: int = 1,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        assert (
            use_flash_attn == use_fused_rmsnorm == use_fused_mlp
        ), "use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent"
        assert use_flash_attn is False, "flash_attn is not currently supported"

        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim

        self.depth = depth
        self.clip_norm_type = clip_norm_type
        self.return_index = []
        for i in range(clip_return_layer):
            self.return_index.append(depth - int(i * clip_student_return_interval) - 1)
        logger.info(f"Normalization Type: {clip_norm_type}")
        logger.info(f"Strudent Return Index: {self.return_index}")

        if use_fused_rmsnorm:
            raise Exception("Sorry, DropoutAddRMSNorm is not supported yet.")
            # norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        else:
            norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        self.norm_layer_for_blocks = norm_layer_for_blocks
        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
        )
        num_patches = self.patch_embed.num_patches
        num_img_patches = self.patch_embed.num_img_patches

        self.cls_token = ms.Parameter(ops.zeros((1, 1, embed_dim)))

        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        self.sep_image_video_pos_embed = sep_image_video_pos_embed
        if sep_pos_embed:
            raise NotImplementedError
        else:
            if sep_image_video_pos_embed:
                logger.info("Use joint position embedding, for image and video we use different pos_embed.")
                self.pos_embed = ms.Parameter(ops.zeros((1, num_patches + 1, embed_dim)))
                self.img_pos_embed = ms.Parameter(ops.zeros((1, num_img_patches + 1, embed_dim)))
                # for CLIP decoder
                self.clip_pos_embed = ms.Parameter(ops.zeros((1, num_patches + 1, embed_dim)))
                self.clip_img_pos_embed = ms.Parameter(ops.zeros((1, num_img_patches + 1, embed_dim)))
            else:
                logger.info("Use joint position embedding, for image and video we use same pos_embed.")
                self.pos_embed = ms.Parameter(ops.zeros((1, num_patches + 1, embed_dim)))
                self.clip_pos_embed = ms.Parameter(ops.zeros((1, num_patches + 1, embed_dim)))
        dpr = [x.item() for x in ops.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        logger.info(f"Droppath rate: {dpr}")
        logger.info(f"Checkpoint list: {with_cp_list}")

        self.blocks = nn.CellList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer_for_blocks,
                    drop_path=dpr[i],
                    init_values=init_values,
                    attn_drop=0.0,
                    use_flash_attn=use_flash_attn,
                    use_fused_mlp=use_fused_mlp,
                    fused_mlp_heuristic=fused_mlp_heuristic,
                    with_cp=with_cp_list[i],
                    qk_normalization=qk_normalization,
                    layerscale_no_force_fp32=layerscale_no_force_fp32,
                    use_fused_rmsnorm=use_fused_rmsnorm,
                )
                for i in range(depth)
            ]
        )
        self.clip_projector = AttentionPoolingBlock(
            dim=embed_dim,
            num_heads=attn_pool_num_heads,
            qkv_bias=True,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            norm_layer=partial(LayerNorm, epsilon=1e-5),
            out_dim=clip_embed_dim,
        )

        # CLIP decoder
        self.clip_decoder = nn.CellList(
            [
                Linear_Decoder(
                    in_channels=embed_dim,
                    out_channels=clip_teacher_embed_dim,
                    norm_layer=partial(LayerNorm, epsilon=1e-5),
                    clip_norm_type=clip_norm_type,
                )
                for _ in range(clip_return_layer)
            ]
        )
        self.final_clip_decoder = nn.Identity()
        if clip_teacher_final_dim > 0:
            self.final_clip_decoder = Linear_Decoder(
                in_channels=clip_embed_dim,
                out_channels=clip_teacher_final_dim,
                norm_layer=partial(LayerNorm, epsilon=1e-5),
                clip_norm_type=clip_norm_type,
            )

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def init_pos_embed(self):
        logger.info("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            # trunc_normal_(self.pos_embed, std=.02)
            # trunc_normal_(self.clip_pos_embed, std=.02)
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.patch_embed.grid_size[1],  # height & weight
                self.patch_embed.grid_size[0],  # t_size
                cls_token=True,
            )
            self.pos_embed.set_data(ms.Tensor(pos_embed).float().unsqueeze(0))
            self.clip_pos_embed.set_data(ms.Tensor(pos_embed).float().unsqueeze(0))

            if self.sep_image_video_pos_embed:
                img_pos_embed = get_3d_sincos_pos_embed(
                    self.pos_embed.shape[-1], self.patch_embed.grid_size[1], 1, cls_token=True  # height & weight
                )
                self.img_pos_embed.set_data(ms.Tensor(img_pos_embed).float().unsqueeze(0))
                self.clip_img_pos_embed.set_data(ms.Tensor(img_pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Dense):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Dense) and m.bias is not None:
                m.bias.set_data(initializer(Constant(0), m.bias.shape, m.bias.dtype))
        elif isinstance(m, nn.LayerNorm):
            m.beta.set_data(initializer(Constant(0), m.beta.shape, m.beta.dtype))
            m.gamma.set_data(initializer(Constant(1.0), m.gamma.shape, m.gamma.dtype))

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param = param.div(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {
            "pos_embed",
            "pos_embed_spatial",
            "pos_embed_temporal",
            "pos_embed_cls",
            "img_pos_embed",
            "cls_token",
            "clip_pos_embed",
            "clip_pos_embed_spatial",
            "clip_pos_embed_temporal",
            "clip_pos_embed_cls",
            "clip_img_pos_embed",
        }

    def construct(self, x, mask=None, use_image=False, x_vis_return_idx=-1, x_vis_only=False):
        x = self.patch_embed(x.type(self.dtype))
        # print(f"x.shape: {x.shape} x.dtype: {x.dtype}, model.dtype: {self.dtype}")
        B, T, L, C = x.shape  # T: temporal; L: spatial
        x = x.reshape([B, T * L, C])

        # append cls token
        cls_tokens = self.cls_token.repeat_interleave(B, 0)
        x = mint.cat((cls_tokens, x), dim=1)

        # add pos_embed
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            if use_image:
                if self.sep_image_video_pos_embed:
                    pos_embed = self.img_pos_embed
                else:
                    # (1, num_img_patches + 1, embed_dim)
                    # print('origin pos_embed.shape:', self.pos_embed.shape)
                    cls_pos_embed = self.pos_embed[:, 0:1, :]
                    # print('cls_pos_embed.shape:', cls_pos_embed.shape)

                    img_pos_embed = (
                        self.pos_embed[:, 1:, :]
                        .view(1, self.num_frames, self.patch_embed.num_patches // self.num_frames, self.embed_dim)
                        .mean(dim=1)
                    )
                    # print('img_pos_embed.shape:', img_pos_embed.shape)

                    pos_embed = mint.cat([cls_pos_embed, img_pos_embed], dim=1)
                    # print('final img_pos_embed.shape:', pos_embed.shape)
            else:
                pos_embed = self.pos_embed
        x = x + pos_embed

        # mask tokens, ~mask means visible
        if mask is not None:
            x = x[~mask].reshape(B, -1, C)
        else:
            x = x.reshape(B, -1, C)

        residual = None
        x_clip = []
        for idx, blk in enumerate(self.blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            # print(f"\033[31m这是{idx}, {x.shape}\033[0m")
            if blk.with_cp:
                x = recompute(blk, x, residual=residual)
            else:
                x = blk(x, residual=residual)
            # return intermediate features
            if idx in self.return_index:
                if isinstance(x, tuple) and len(x) == 2:
                    tmp_x, tmp_residual = x
                    if residual is not None:
                        x_clip.append(tmp_x + tmp_residual)
                else:
                    x_clip.append(x)
            if idx == (self.depth + x_vis_return_idx):
                # print(f'idx = {idx} len(self.blocks)={len(self.blocks)}')
                break

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        x_vis = x
        if x_vis_only:
            return x_vis

        x_pool_vis = self.clip_projector(x_vis)
        x_align = self.final_clip_decoder(x_pool_vis)

        # align CLIP
        x_clip = ops.stack(x_clip)
        K, B, _, C_CLIP = x_clip.shape
        # add pos_embed
        if self.sep_pos_embed:
            raise NotImplementedError
        else:
            if use_image:
                if self.sep_image_video_pos_embed:
                    clip_pos_embed = self.clip_img_pos_embed
                else:
                    # (1, num_img_patches + 1, embed_dim)
                    # print('origin pos_embed.shape:', self.pos_embed.shape)
                    clip_cls_pos_embed = self.clip_pos_embed[:, 0:1, :]
                    # print('cls_pos_embed.shape:', cls_pos_embed.shape)

                    clip_img_pos_embed = (
                        self.clip_pos_embed[:, 1:, :]
                        .view(1, self.num_frames, self.patch_embed.num_patches // self.num_frames, self.embed_dim)
                        .mean(dim=1)
                    )
                    # print('img_pos_embed.shape:', img_pos_embed.shape)

                    clip_pos_embed = mint.cat([clip_cls_pos_embed, clip_img_pos_embed], dim=1)
                    # print('final img_pos_embed.shape:', pos_embed.shape)

            else:
                clip_pos_embed = self.clip_pos_embed

        clip_pos_embed = clip_pos_embed.repeat_interleave(B, 0)
        if mask is not None:
            x_clip = x_clip + clip_pos_embed[~mask].view(B, -1, C_CLIP).unsqueeze(0).repeat_interleave(K, 0)
        else:
            x_clip = x_clip + clip_pos_embed.view(B, -1, C_CLIP).unsqueeze(0).repeat_interleave(K, 0)

        # CLIP decoder
        x_clip_align = []
        for idx, clip_decoder in enumerate(self.clip_decoder):
            x_clip_align.append(clip_decoder(x_clip[idx]))
        x_clip_align = ops.stack(x_clip_align)

        return x_vis, x_pool_vis, x_clip_align, x_align


def pretrain_internvideo2_1b_patch14_224(config):
    model = PretrainInternVideo2(
        in_chans=3,
        img_size=224,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        mlp_ratio=48 / 11,
        clip_embed_dim=config.vision_encoder.clip_embed_dim,
        attn_pool_num_heads=16,
        qkv_bias=False,
        drop_path_rate=0.25,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=config.vision_encoder.get("use_flash_attn", True),
        use_fused_rmsnorm=config.vision_encoder.get("use_fused_rmsnorm", True),
        use_fused_mlp=config.vision_encoder.get("use_fused_mlp", True),
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=False,
        num_frames=config.vision_encoder.num_frames,
        tubelet_size=config.vision_encoder.tubelet_size,
        sep_pos_embed=False,
        sep_image_video_pos_embed=config.vision_encoder.sep_image_video_pos_embed,
        use_checkpoint=config.vision_encoder.use_checkpoint,
        checkpoint_num=config.vision_encoder.checkpoint_num,
        clip_teacher_embed_dim=config.vision_encoder.clip_teacher_embed_dim,
        clip_teacher_final_dim=config.vision_encoder.clip_teacher_final_dim,
        clip_norm_type=config.vision_encoder.clip_norm_type,
        clip_return_layer=config.vision_encoder.clip_return_layer,
        clip_student_return_interval=config.vision_encoder.clip_student_return_interval,
    )

    if config.vision_encoder.pretrained is not None:
        logger.info(f"Loading pretrained weights from {config.vision_encoder.pretrained}")
        state_dict = ms.load_checkpoint(config.vision_encoder.pretrained)

        state_dict_new = {}
        for k, v in state_dict.items():
            if k.startswith("vision_encoder"):
                state_dict_new[k[len("vision_encoder.") :]] = v
        state_dict = state_dict_new

        interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=4)
        ms.load_param_into_net(model, state_dict)
    else:
        logger.info("No pretrained weights!!!")
    return model


def pretrain_internvideo2_6b_patch14_224(config):
    model = PretrainInternVideo2(
        in_chans=3,
        img_size=224,
        patch_size=14,
        embed_dim=3200,
        depth=48,
        num_heads=25,
        mlp_ratio=4,
        clip_embed_dim=config.vision_encoder.clip_embed_dim,
        attn_pool_num_heads=16,
        qkv_bias=False,
        drop_path_rate=0.3,
        init_values=0.00001,
        qk_normalization=True,
        use_flash_attn=config.vision_encoder.get("use_flash_attn", True),
        use_fused_rmsnorm=config.vision_encoder.get("use_fused_rmsnorm", True),
        use_fused_mlp=config.vision_encoder.get("use_fused_mlp", True),
        fused_mlp_heuristic=1,
        layerscale_no_force_fp32=False,
        num_frames=config.vision_encoder.num_frames,
        tubelet_size=config.vision_encoder.tubelet_size,
        sep_pos_embed=False,
        sep_image_video_pos_embed=config.vision_encoder.sep_image_video_pos_embed,
        use_checkpoint=config.vision_encoder.use_checkpoint,
        checkpoint_num=config.vision_encoder.checkpoint_num,
        clip_teacher_embed_dim=config.vision_encoder.clip_teacher_embed_dim,
        clip_teacher_final_dim=config.vision_encoder.clip_teacher_final_dim,
        clip_norm_type=config.vision_encoder.clip_norm_type,
        clip_return_layer=config.vision_encoder.clip_return_layer,
        clip_student_return_interval=config.vision_encoder.clip_student_return_interval,
    )

    if config.vision_encoder.pretrained is not None:
        logger.info(f"Loading pretrained weights from {config.vision_encoder.pretrained}")
        state_dict = ms.load_checkpoint(config.vision_encoder.pretrained, map_location="cpu")
        interpolate_pos_embed_internvideo2(state_dict, model, orig_t_size=8)
        ms.load_param_into_net(model, state_dict, strict_load=False)
        logger.info("Pretrained weights loaded.")
    else:
        logger.info("No pretrained weights!!!")
    return model


if __name__ == "__main__":
    seed = 4217
    np.random.seed(seed)
    num_frames = 8
    img_size = 224

    model = pretrain_internvideo2_1b_patch14_224(clip_return_layer=6)
    # print(model)

    # flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda().half())
    # s = time.time()
    # print(flop_count_table(flops, max_depth=1))
    # print(time.time()-s)

    mask = ops.cat(
        [
            ops.ones((1, 8 * int(16 * 16 * 0.75))),
            ops.zeros((1, 8 * int(16 * 16 * 0.25))),
            ops.zeros((1, 1)),
        ],
        dim=-1,
    ).to(ms.bool)

    output = model(ops.rand((4, 3, num_frames, img_size, img_size)), mask.repeat(4, 1))
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)
    print(output[3].shape)
