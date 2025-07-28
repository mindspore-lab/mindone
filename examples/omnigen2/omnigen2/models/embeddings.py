# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Union

from mindspore import Tensor, mint, nn, ops

from mindone.diffusers.models.activations import get_activation
from mindone.models.utils import normal_, zeros_


def view_as_complex(x: Tensor) -> Tensor:
    real_part, imag_part = mint.chunk(x, 2, dim=-1)
    return ops.Complex()(real_part, imag_part).squeeze(axis=-1)


def view_as_real(x: Tensor) -> Tensor:
    # Stack real and imaginary parts along a new last dimension
    return mint.stack((ops.Real()(x), ops.Imag()(x)), dim=-1)


class TimestepEmbedding(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

        self.initialize_weights()

    def initialize_weights(self):
        normal_(self.linear_1.weight, std=0.02)
        zeros_(self.linear_1.bias)
        normal_(self.linear_2.weight, std=0.02)
        zeros_(self.linear_2.bias)

    def construct(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


def apply_rotary_emb(
    x: Tensor, freqs_cis: Union[Tensor, tuple[Tensor]], use_real: bool = True, use_real_unbind_dim: int = -1
) -> tuple[Tensor, Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (Tensor): Key tensor to apply
        freqs_cis (`tuple[Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        tuple[Tensor, Tensor]: tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = mint.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen and CogView4
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = mint.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        # x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = view_as_complex(x.float().reshape(*x.shape[:-1], x.shape[-1] // 2, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)
