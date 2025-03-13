# Modified from Flux
#
# Copyright 2024 Black Forest Labs

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import load_param_into_net, mint, nn

from mindone.models.utils import zeros_

from ...utils.model_utils import load_state_dict
from .layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    LigerEmbedND,
    MLPEmbedder,
    SingleStreamBlock,
    SinusoidalEmbedding,
)

_logger = logging.getLogger(__name__)


@dataclass
class MMDiTConfig:
    model_type = "MMDiT"
    from_pretrained: str
    cache_dir: str
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool
    cond_embed: bool = False
    fused_qkv: bool = True
    grad_ckpt_settings: Optional[tuple[int, int]] = None
    use_liger_rope: bool = False
    patch_size: int = 2

    def get(self, attribute_name, default=None):
        return getattr(self, attribute_name, default)

    def __contains__(self, attribute_name):
        return hasattr(self, attribute_name)


class MMDiTModel(nn.Cell):
    config_class = MMDiTConfig

    def __init__(self, config: MMDiTConfig, dtype: mstype.Type = mstype.bfloat16):
        super().__init__()

        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = self.in_channels
        self.patch_size = config.patch_size
        self.dtype = dtype

        if config.hidden_size % config.num_heads != 0:
            raise ValueError(f"Hidden size {config.hidden_size} must be divisible by num_heads {config.num_heads}")

        pe_dim = config.hidden_size // config.num_heads
        if sum(config.axes_dim) != pe_dim:
            raise ValueError(f"Got {config.axes_dim} but expected positional dim {pe_dim}")

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        pe_embedder_cls = LigerEmbedND if config.use_liger_rope else EmbedND
        self.pe_embedder = pe_embedder_cls(dim=pe_dim, theta=config.theta, axes_dim=config.axes_dim)

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True, dtype=dtype)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size, dtype=dtype)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size, dtype=dtype)
            if config.guidance_embed
            else nn.Identity()
        )
        self.cond_in = (
            nn.Linear(self.in_channels + self.patch_size**2, self.hidden_size, bias=True, dtype=dtype)
            if config.cond_embed
            else nn.Identity()
        )
        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size, dtype=dtype)

        self.double_blocks = nn.CellList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    qkv_bias=config.qkv_bias,
                    fused_qkv=config.fused_qkv,
                    use_liger_rope=config.use_liger_rope,
                    dtype=dtype,
                )
                for _ in range(config.depth)
            ]
        )

        self.single_blocks = nn.CellList(
            [
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    fused_qkv=config.fused_qkv,
                    use_liger_rope=config.use_liger_rope,
                    dtype=dtype,
                )
                for _ in range(config.depth_single_blocks)
            ]
        )

        self.timestep_embedding = SinusoidalEmbedding(256)
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels, dtype=dtype)
        self.initialize_weights()

        # TODO: add recompute

        self._input_requires_grad = False

    def initialize_weights(self):
        if self.config.cond_embed:
            zeros_(self.cond_in.weight)
            zeros_(self.cond_in.bias)

    def prepare_block_inputs(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,  # t5 encoded vec
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,  # clip encoded vec
        cond: Tensor = None,
        guidance: Tensor | None = None,
    ):
        """
        obtain the processed:
            img: projected noisy img latent,
            txt: text context (from t5),
            vec: clip encoded vector,
            pe: the positional embeddings for concatenated img and txt
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # running on sequences img
        img = self.img_in(img)
        if self.config.cond_embed:
            if cond is None:
                raise ValueError("Didn't get conditional input for conditional model.")
            img = img + self.cond_in(cond)

        vec = self.time_in(self.timestep_embedding(timesteps))
        if self.config.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(self.timestep_embedding(guidance))
        vec = vec + self.vector_in(y_vec)

        txt = self.txt_in(txt)

        # concat: 4096 + t*h*2/4
        ids = mint.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        if self._input_requires_grad:
            # we only apply lora to double/single blocks, thus we only need to enable grad for these inputs
            img.requires_grad_()
            txt.requires_grad_()

        return img, txt, vec, pe

    def enable_input_require_grads(self):
        """Fit peft lora. This method should not be called manually."""
        self._input_requires_grad = True

    def construct(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y_vec: Tensor,
        cond: Tensor = None,
        guidance: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        img, txt, vec, pe = self.prepare_block_inputs(img, img_ids, txt, txt_ids, timesteps, y_vec, cond, guidance)

        for block in self.double_blocks:
            img, txt = block(img, txt, vec, pe)

        img = mint.cat((txt, img), 1)
        for block in self.single_blocks:
            img = block(img, vec, pe)
        img = img[:, txt.shape[1] :, ...]

        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)
        return img


def Flux(
    cache_dir: str = None,
    from_pretrained: str = None,
    dtype: Literal["fp32", "fp16", "bf16"] = "bf16",
    **kwargs,
) -> MMDiTModel:
    dtype = {"fp32": mstype.float32, "fp16": mstype.float16, "bf16": mstype.bfloat16}[dtype]
    config = MMDiTConfig(
        from_pretrained=from_pretrained,
        cache_dir=cache_dir,
        **kwargs,
    )
    with nn.no_init_parameters():
        model = MMDiTModel(config, dtype=dtype)
    if from_pretrained:
        sd, ckpt_path = load_state_dict(from_pretrained)
        m, u = load_param_into_net(model, sd)
        if m or u:
            _logger.info(f"net param not load {len(m)}: {m}")
            _logger.info(f"ckpt param not load {len(u)}: {u}")
        _logger.info(f"Loaded ckpt {ckpt_path} into MMDiT.")
    model.init_parameters_data()
    return model
