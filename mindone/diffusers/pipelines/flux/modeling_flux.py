# Copyright 2024 Black Forest Labs and The HuggingFace Team. All rights reserved.
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


from dataclasses import dataclass
from typing import Optional, Union

import mindspore as ms
from mindspore import mint, nn

from ...configuration_utils import ConfigMixin, register_to_config
from ...models.modeling_utils import ModelMixin
from ...utils import BaseOutput


@dataclass
class ReduxImageEncoderOutput(BaseOutput):
    image_embeds: Optional[ms.Tensor] = None


class ReduxImageEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        redux_dim: int = 1152,
        txt_in_features: int = 4096,
    ) -> None:
        super().__init__()

        self.redux_up = nn.Dense(redux_dim, txt_in_features * 3)
        self.redux_down = nn.Dense(txt_in_features * 3, txt_in_features)

    def construct(self, x: ms.Tensor, return_dict: Optional[bool] = False) -> Union[tuple, ReduxImageEncoderOutput]:
        projected_x = self.redux_down(mint.nn.functional.silu(self.redux_up(x)))

        if not return_dict:
            return (projected_x,)

        return ReduxImageEncoderOutput(image_embeds=projected_x)
