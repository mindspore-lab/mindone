# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import mindspore as ms
from mindspore import nn


class ModLN(nn.Cell):
    """
    Modulation with adaLN.

    References:
    DiT: https://github.com/facebookresearch/DiT/blob/main/models.py#L101
    """

    def __init__(self, inner_dim: int, mod_dim: int, epsilon: float):
        super().__init__()
        self.norm = nn.LayerNorm((inner_dim,), epsilon=epsilon)
        self.mlp = nn.SequentialCell(
            nn.SiLU(),
            nn.Dense(mod_dim, inner_dim * 2),
        )

    @staticmethod
    def modulate(x, shift, scale):
        # x: [N, L, D]
        # shift, scale: [N, D]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def construct(self, x: ms.Tensor, mod: ms.Tensor) -> ms.Tensor:
        shift, scale = self.mlp(mod).chunk(2, axis=-1)  # [N, D]
        return self.modulate(self.norm(x), shift, scale)  # [N, L, D]
