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

from typing import TYPE_CHECKING

from ..utils import _LazyModule

_import_structure = {
    "adapter": ["MultiAdapter", "T2IAdapter"],
    "autoencoders.autoencoder_kl": ["AutoencoderKL"],
    "controlnet": ["ControlNetModel"],
    "embeddings": ["ImageProjection"],
    "modeling_utils": ["ModelMixin"],
    "transformers.transformer_2d": ["Transformer2DModel"],
    "transformers.transformer_sd3": ["SD3Transformer2DModel"],
    "unets.unet_2d": ["UNet2DModel"],
    "unets.unet_2d_condition": ["UNet2DConditionModel"],
}

if TYPE_CHECKING:
    from .adapter import MultiAdapter, T2IAdapter
    from .autoencoders import AutoencoderKL
    from .controlnet import ControlNetModel
    from .embeddings import ImageProjection
    from .modeling_utils import ModelMixin
    from .transformers import SD3Transformer2DModel, Transformer2DModel
    from .unets import UNet2DConditionModel, UNet2DModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
