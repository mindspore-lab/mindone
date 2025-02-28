# Modified from ``https://github.com/openai/CLIP'' and ``https://github.com/mlfoundations/open_clip''
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import logging
import math
from typing import List, Tuple

import numpy as np
import mindspore as ms
import mindspore.mint as mint
import mindspore.mint.nn.functional as F
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, Tensor
from mindspore.nn.utils import no_init_parameters
from mindspore.dataset.transforms import Compose
import mindspore.dataset.vision as vision

from ..utils.utils import load_pth
from .tokenizers import HuggingfaceTokenizer
from .xlm_roberta import XLMRoberta

__all__ = [
    "XLMRobertaCLIP",
    "clip_xlm_roberta_vit_h_14",
    "CLIPModel",
]


def pos_interpolate(pos: Tensor, seq_len: int) -> Tensor:
    if pos.shape[1] == seq_len:
        return pos
    else:
        src_grid = int(math.sqrt(pos.shape[1]))
        tar_grid = int(math.sqrt(seq_len))
        n = pos.shape[1] - src_grid * src_grid
        return mint.cat(
            [
                pos[:, :n],
                F.interpolate(
                    pos[:, n:].float().reshape(1, src_grid, src_grid, -1).permute(0, 3, 1, 2),
                    size=(tar_grid, tar_grid),
                    mode="bicubic",
                    align_corners=False,
                )
                .flatten(2)
                .transpose(1, 2),
            ],
            dim=1,
        )


class QuickGELU(nn.Cell):
    def construct(self, x: Tensor) -> Tensor:
        return x * mint.sigmoid(1.702 * x)


class LayerNorm(mint.nn.LayerNorm):
    # TODO: to float32
    def construct(self, x: Tensor) -> Tensor:
        return super().construct(x).type_as(x)


class SelfAttention(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        causal: bool = False,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        dtype: ms.Type = ms.float32,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.causal = causal
        self.attn_dropout = attn_dropout
        self.proj_dropout = proj_dropout

        # layers
        self.to_qkv = mint.nn.Linear(dim, dim * 3, dtype=dtype)
        self.proj = mint.nn.Linear(dim, dim, dtype=dtype)

    def construct(self, x: Tensor) -> Tensor:
        """
        x:   [B, L, C].
        """
        b, s, c, n, d = *x.shape, self.num_heads, self.head_dim

        # compute query, key, value
        q, k, v = self.to_qkv(x).view(b, s, 3, n, d).unbind(2)

        # compute attention
        p = self.attn_dropout if self.training else 0.0
        x = ops.flash_attention_score(
            query=q,
            key=k,
            value=v,
            head_num=self.num_heads,
            keep_prob=1.0 - p,
            input_layout="BSND",
        )
        x = x.reshape(b, s, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)
        return x


class SwiGLU(nn.Cell):
    def __init__(self, dim: int, mid_dim: int, dtype: ms.Type = ms.float32) -> None:
        super().__init__()
        self.dim = dim
        self.mid_dim = mid_dim

        # layers
        self.fc1 = mint.nn.Linear(dim, mid_dim, dtype=dtype)
        self.fc2 = mint.nn.Linear(dim, mid_dim, dtype=dtype)
        self.fc3 = mint.nn.Linear(mid_dim, dim, dtype=dtype)

    def construct(self, x: Tensor) -> Tensor:
        x = F.silu(self.fc1(x)) * self.fc2(x)
        x = self.fc3(x)
        return x


class AttentionBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        num_heads: int,
        post_norm: bool = False,
        causal: bool = False,
        activation: str = "quick_gelu",
        attn_dropout: str = 0.0,
        proj_dropout: str = 0.0,
        norm_eps: str = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        assert activation in ["quick_gelu", "gelu", "swi_glu"]
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.post_norm = post_norm
        self.causal = causal
        self.norm_eps = norm_eps

        # layers
        self.norm1 = LayerNorm(dim, eps=norm_eps, dtype=dtype)
        self.attn = SelfAttention(dim, num_heads, causal, attn_dropout, proj_dropout, dtype=dtype)
        self.norm2 = LayerNorm(dim, eps=norm_eps, dtype=dtype)
        if activation == "swi_glu":
            self.mlp = SwiGLU(dim, int(dim * mlp_ratio), dtype=dtype)
        else:
            self.mlp = nn.SequentialCell(
                mint.nn.Linear(dim, int(dim * mlp_ratio), dtype=dtype),
                QuickGELU() if activation == "quick_gelu" else mint.nn.GELU(),
                mint.nn.Linear(int(dim * mlp_ratio), dim, dtype=dtype),
                mint.nn.Dropout(proj_dropout),
            )

    def construct(self, x: Tensor) -> Tensor:
        if self.post_norm:
            x = x + self.norm1(self.attn(x))
            x = x + self.norm2(self.mlp(x))
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.mlp(self.norm2(x))
        return x


class AttentionPool(nn.Cell):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        num_heads: int,
        activation: str = "gelu",
        proj_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.proj_dropout = proj_dropout
        self.norm_eps = norm_eps

        # layers
        gain = 1.0 / math.sqrt(dim)
        self.cls_embedding = Parameter(Tensor(gain * np.random.randn(1, 1, dim), dtype=dtype))
        self.to_q = mint.nn.Linear(dim, dim, dtype=dtype)
        self.to_kv = mint.nn.Linear(dim, dim * 2, dtype=dtype)
        self.proj = mint.nn.Linear(dim, dim, dtype=dtype)
        self.norm = LayerNorm(dim, eps=norm_eps, dtype=dtype)
        self.mlp = nn.SequentialCell(
            mint.nn.Linear(dim, int(dim * mlp_ratio), dtype=dtype),
            QuickGELU() if activation == "quick_gelu" else nn.GELU(),
            mint.nn.Linear(int(dim * mlp_ratio), dim, dtype=dtype),
            mint.nn.Dropout(proj_dropout),
        )

    def construct(self, x: Tensor) -> Tensor:
        """
        x:  [B, L, C].
        """
        b, s, c, n, d = *x.shape, self.num_heads, self.head_dim

        # compute query, key, value
        q = self.to_q(self.cls_embedding).view(1, 1, n, d).expand((b, -1, -1, -1))
        k, v = self.to_kv(x).view(b, s, 2, n, d).unbind(2)

        # compute attention
        x = ops.flash_attention_score(
            query=q,
            key=k,
            value=v,
            head_num=self.num_heads,
            input_layout="BSND",
        )
        x = x.reshape(b, 1, c)

        # output
        x = self.proj(x)
        x = F.dropout(x, self.proj_dropout, self.training)

        # mlp
        x = x + self.mlp(self.norm(x))
        return x[:, 0]


class VisionTransformer(nn.Cell):
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        dim: int = 768,
        mlp_ratio: int = 4,
        out_dim: int = 512,
        num_heads: int = 12,
        num_layers: int = 12,
        pool_type: str = "token",
        pre_norm: bool = True,
        post_norm: bool = False,
        activation: str = "quick_gelu",
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        if image_size % patch_size != 0:
            print("[WARNING] image_size is not divisible by patch_size", flush=True)
        assert pool_type in ("token", "token_fc", "attn_pool")
        out_dim = out_dim or dim
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pool_type = pool_type
        self.post_norm = post_norm
        self.norm_eps = norm_eps

        # embeddings
        gain = 1.0 / math.sqrt(dim)
        self.patch_embedding = mint.nn.Conv2d(
            3, dim, kernel_size=patch_size, stride=patch_size, bias=not pre_norm, dtype=dtype
        )
        if pool_type in ("token", "token_fc"):
            self.cls_embedding = Parameter(Tensor(gain * np.random.randn(1, 1, dim), dtype=dtype))
        self.pos_embedding = Parameter(
            Tensor(
                gain * np.random.randn(1, self.num_patches + (1 if pool_type in ("token", "token_fc") else 0), dim),
                dtype=dtype,
            )
        )
        self.dropout = mint.nn.Dropout(embedding_dropout)

        # transformer
        self.pre_norm = LayerNorm(dim, eps=norm_eps, dtype=dtype) if pre_norm else None
        self.transformer = nn.SequentialCell(
            *[
                AttentionBlock(
                    dim,
                    mlp_ratio,
                    num_heads,
                    post_norm,
                    False,
                    activation,
                    attn_dropout,
                    proj_dropout,
                    norm_eps,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.post_norm = LayerNorm(dim, eps=norm_eps, dtype=dtype)

        # head
        if pool_type == "token":
            self.head = Parameter(Tensor(gain * np.random.randn(dim, out_dim), dtype=dtype))
        elif pool_type == "token_fc":
            self.head = mint.nn.Linear(dim, out_dim, dtype=dtype)
        elif pool_type == "attn_pool":
            self.head = AttentionPool(dim, mlp_ratio, num_heads, activation, proj_dropout, norm_eps, dtype=dtype)

    def construct(self, x: Tensor, interpolation: bool = False, use_31_block: bool = False) -> Tensor:
        b = x.shape[0]

        # embeddings
        x = self.patch_embedding(x).flatten(2).permute(0, 2, 1)
        if self.pool_type in ("token", "token_fc"):
            x = mint.cat([self.cls_embedding.expand((b, -1, -1)), x], dim=1)
        if interpolation:
            e = pos_interpolate(self.pos_embedding, x.shape[1])
        else:
            e = self.pos_embedding
        x = self.dropout(x + e)
        if self.pre_norm is not None:
            x = self.pre_norm(x)

        # transformer
        if use_31_block:
            x = self.transformer[:-1](x)
            return x
        else:
            x = self.transformer(x)
            return x


class XLMRobertaWithHead(XLMRoberta):
    def __init__(self, dtype: ms.Type = ms.float32, **kwargs) -> None:
        self.out_dim = kwargs.pop("out_dim")
        super().__init__(dtype=dtype, **kwargs)

        # head
        mid_dim = (self.dim + self.out_dim) // 2
        self.head = nn.SequentialCell(
            mint.nn.Linear(self.dim, mid_dim, bias=False, dtype=dtype),
            mint.nn.GELU(),
            mint.nn.Linear(mid_dim, self.out_dim, bias=False, dtype=dtype),
        )

    def construct(self, ids: Tensor) -> Tensor:
        # xlm-roberta
        x = super().construct(ids)

        # average pooling
        mask = ids.ne(self.pad_id).unsqueeze(-1).to(x.dtype)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1)

        # head
        x = self.head(x)
        return x


class XLMRobertaCLIP(nn.Cell):
    def __init__(
        self,
        embed_dim: int = 1024,
        image_size: int = 224,
        patch_size: int = 14,
        vision_dim: int = 1280,
        vision_mlp_ratio: float = 4,
        vision_heads: int = 16,
        vision_layers: int = 32,
        vision_pool: str = "token",
        vision_pre_norm: bool = True,
        vision_post_norm: bool = False,
        activation="gelu",
        vocab_size: int = 250002,
        max_text_len: int = 514,
        type_size: int = 1,
        pad_id: int = 1,
        text_dim: int = 1024,
        text_heads: int = 16,
        text_layers: int = 24,
        text_post_norm: bool = True,
        text_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        embedding_dropout: float = 0.0,
        norm_eps: float = 1e-5,
        dtype: ms.Type = ms.float32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.vision_mlp_ratio = vision_mlp_ratio
        self.vision_heads = vision_heads
        self.vision_layers = vision_layers
        self.vision_pre_norm = vision_pre_norm
        self.vision_post_norm = vision_post_norm
        self.activation = activation
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        self.type_size = type_size
        self.pad_id = pad_id
        self.text_dim = text_dim
        self.text_heads = text_heads
        self.text_layers = text_layers
        self.text_post_norm = text_post_norm
        self.norm_eps = norm_eps
        self.dtype = dtype

        # models
        self.visual = VisionTransformer(
            image_size=image_size,
            patch_size=patch_size,
            dim=vision_dim,
            mlp_ratio=vision_mlp_ratio,
            out_dim=embed_dim,
            num_heads=vision_heads,
            num_layers=vision_layers,
            pool_type=vision_pool,
            pre_norm=vision_pre_norm,
            post_norm=vision_post_norm,
            activation=activation,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            embedding_dropout=embedding_dropout,
            norm_eps=norm_eps,
            dtype=dtype,
        )
        self.textual = XLMRobertaWithHead(
            vocab_size=vocab_size,
            max_seq_len=max_text_len,
            type_size=type_size,
            pad_id=pad_id,
            dim=text_dim,
            out_dim=embed_dim,
            num_heads=text_heads,
            num_layers=text_layers,
            post_norm=text_post_norm,
            dropout=text_dropout,
            dtype=dtype,
        )
        self.log_scale = Parameter(Tensor(math.log(1 / 0.07) * np.ones([]), dtype=dtype))

    def construct(self, imgs: Tensor, txt_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """
        imgs:       [B, 3, H, W] of ms.float32.
        - mean:     [0.48145466, 0.4578275, 0.40821073]
        - std:      [0.26862954, 0.26130258, 0.27577711]
        txt_ids:    [B, L] of ms.int32.
                    Encoded by data.CLIPTokenizer.
        """
        xi = self.visual(imgs)
        xt = self.textual(txt_ids)
        return xi, xt


def _clip(
    pretrained=False,
    pretrained_name=None,
    model_cls=XLMRobertaCLIP,
    return_transforms=False,
    return_tokenizer=False,
    tokenizer_padding="eos",
    dtype: ms.Type = ms.float32,
    **kwargs,
):
    # init model
    model = model_cls(**kwargs, dtype=dtype)
    output = (model,)

    # init transforms
    if return_transforms:
        # mean and std
        if "siglip" in pretrained_name.lower():
            mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        else:
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]

        # transforms
        transforms = Compose(
            [
                vision.Resize((model.image_size, model.image_size), interpolation=vision.Inter.BICUBIC),
                vision.ToTensor(),
                vision.Normalize(mean=mean, std=std, is_hwc=False),
            ]
        )
        output += (transforms,)
    return output[0] if len(output) == 1 else output


def clip_xlm_roberta_vit_h_14(
    pretrained=False, pretrained_name="open-clip-xlm-roberta-large-vit-huge-14", dtype: ms.Type = ms.float32, **kwargs
):
    cfg = dict(
        embed_dim=1024,
        image_size=224,
        patch_size=14,
        vision_dim=1280,
        vision_mlp_ratio=4,
        vision_heads=16,
        vision_layers=32,
        vision_pool="token",
        activation="gelu",
        vocab_size=250002,
        max_text_len=514,
        type_size=1,
        pad_id=1,
        text_dim=1024,
        text_heads=16,
        text_layers=24,
        text_post_norm=True,
        text_dropout=0.1,
        attn_dropout=0.0,
        proj_dropout=0.0,
        embedding_dropout=0.0,
    )
    cfg.update(**kwargs)
    return _clip(pretrained, pretrained_name, XLMRobertaCLIP, dtype=dtype, **cfg)


class CLIPModel:
    def __init__(self, dtype, checkpoint_path, tokenizer_path):
        self.dtype = dtype
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        with no_init_parameters():
            model, self.transforms = clip_xlm_roberta_vit_h_14(
                pretrained=False,
                return_transforms=True,
                return_tokenizer=False,
                dtype=dtype,
            )
        model.set_train(False)
        for param in model.trainable_params():
            param.requires_grad = False

        if checkpoint_path is not None:
            logging.info(f"loading {checkpoint_path}")
            if checkpoint_path.endswith(".pth"):
                param_dict = load_pth(checkpoint_path, dtype=model.dtype)
                ms.load_param_into_net(model, param_dict)
            else:
                ms.load_checkpoint(checkpoint_path, model)
        model.init_parameters_data()

        self.model = model

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=self.model.max_text_len - 2, clean="whitespace"
        )

    def visual(self, videos: List[Tensor]) -> Tensor:
        # preprocess
        size = (self.model.image_size,) * 2
        videos = mint.cat(
            [F.interpolate(u.transpose(0, 1), size=size, mode="bicubic", align_corners=False) for u in videos]
        )
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5).asnumpy())

        out = self.model.visual(Tensor(videos, dtype=self.model.dtype), use_31_block=True)
        return out
