# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
from typing import Any, Dict, List, Optional

import mindspore as ms
import mindspore.mint as mint
import mindspore.nn as nn

from mindone.diffusers.models.attention import AdaLayerNorm

from ...utils.amp import autocast
from ..model import WanAttentionBlock, WanCrossAttention
from .auxi_blocks import MotionEncoder_tc


class CausalAudioEncoder(nn.Cell):
    def __init__(
        self,
        dim: int = 5120,
        num_layers: int = 25,
        out_dim: int = 2048,
        video_rate: int = 8,
        num_token: int = 4,
        need_global: bool = False,
        dtype: Any = ms.float32,
    ):
        super().__init__()
        self.encoder = MotionEncoder_tc(
            in_dim=dim, hidden_dim=out_dim, num_heads=num_token, need_global=need_global, dtype=dtype
        )
        weight = mint.ones((1, num_layers, 1, 1), dtype=dtype) * 0.01

        self.weights = ms.Parameter(weight)
        self.act = mint.nn.SiLU()

    def construct(self, features: ms.Tensor) -> ms.Tensor:
        with autocast(dtype=ms.float32):
            # features B * num_layers * dim * video_length
            weights = self.act(self.weights)
            weights_sum = weights.sum(dim=1, keepdims=True)
            weighted_feat = ((features * weights) / weights_sum).sum(dim=1)  # b dim f
            weighted_feat = weighted_feat.permute(0, 2, 1)  # b f dim
            res = self.encoder(weighted_feat)  # b f n dim

        return res  # b f n dim


class AudioCrossAttention(WanCrossAttention):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


class AudioInjector_WAN(nn.Cell):
    def __init__(
        self,
        all_modules: List[nn.Cell],
        all_modules_names: List[str],
        dim: int = 2048,
        num_heads: int = 32,
        inject_layer: List[int] = [0, 27],
        root_net: Optional[nn.Cell] = None,
        enable_adain: bool = False,
        adain_dim: int = 2048,
        need_adain_ont: bool = False,
        dtype: Any = ms.float32,
    ):
        super().__init__()
        self.injected_block_id: Dict[int, int] = {}
        audio_injector_id: int = 0
        for mod_name, mod in zip(all_modules_names, all_modules):
            if isinstance(mod, WanAttentionBlock):
                for inject_id in inject_layer:
                    if f"transformer_blocks.{inject_id}" in mod_name:
                        self.injected_block_id[inject_id] = audio_injector_id
                        audio_injector_id += 1

        self.injector = nn.CellList(
            [
                AudioCrossAttention(dim=dim, num_heads=num_heads, qk_norm=True, dtype=dtype)
                for _ in range(audio_injector_id)
            ]
        )
        self.injector_pre_norm_feat = nn.CellList(
            [mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype) for _ in range(audio_injector_id)]
        )
        self.injector_pre_norm_vec = nn.CellList(
            [mint.nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6, dtype=dtype) for _ in range(audio_injector_id)]
        )
        if enable_adain:
            self.injector_adain_layers = nn.CellList(
                [
                    AdaLayerNorm(output_dim=dim * 2, embedding_dim=adain_dim, chunk_dim=1)
                    for _ in range(audio_injector_id)
                ]
            )
            if need_adain_ont:
                self.injector_adain_output_layers = nn.CellList(
                    [mint.nn.Linear(dim, dim, dtype=dtype) for _ in range(audio_injector_id)]
                )
