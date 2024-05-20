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
    "autoencoders.autoencoder_asym_kl": ["AsymmetricAutoencoderKL"],
    "autoencoders.autoencoder_kl": ["AutoencoderKL"],
    "autoencoders.autoencoder_kl_temporal_decoder": ["AutoencoderKLTemporalDecoder"],
    "autoencoders.autoencoder_tiny": ["AutoencoderTiny"],
    "autoencoders.consistency_decoder_vae": ["ConsistencyDecoderVAE"],
    "controlnet": ["ControlNetModel"],
    "dual_transformer_2d": ["DualTransformer2DModel"],
    "embeddings": ["ImageProjection"],
    "modeling_utils": ["ModelMixin"],
    "transformers.prior_transformer": ["PriorTransformer"],
    "transformers.t5_film_transformer": ["T5FilmDecoder"],
    "transformers.transformer_2d": ["Transformer2DModel"],
    "transformers.transformer_temporal": ["TransformerTemporalModel"],
    "unets.unet_2d": ["UNet2DModel"],
    "unets.unet_2d_condition": ["UNet2DConditionModel"],
    "unets.unet_3d_condition": ["UNet3DConditionModel"],
    "unets.unet_i2vgen_xl": ["I2VGenXLUNet"],
    "unets.unet_kandinsky3": ["Kandinsky3UNet"],
    "unets.unet_motion_model": ["MotionAdapter", "UNetMotionModel"],
    "unets.unet_spatio_temporal_condition": ["UNetSpatioTemporalConditionModel"],
    "unets.uvit_2d": ["UVit2DModel"],
    "vq_model": ["VQModel"],
}

if TYPE_CHECKING:
    from .adapter import MultiAdapter, T2IAdapter
    from .autoencoders import (
        AsymmetricAutoencoderKL,
        AutoencoderKL,
        AutoencoderKLTemporalDecoder,
        AutoencoderTiny,
        ConsistencyDecoderVAE,
    )
    from .controlnet import ControlNetModel
    from .embeddings import ImageProjection
    from .modeling_utils import ModelMixin
    from .transformers import (
        DualTransformer2DModel,
        PriorTransformer,
        T5FilmDecoder,
        Transformer2DModel,
        TransformerTemporalModel,
    )
    from .unets import (
        I2VGenXLUNet,
        Kandinsky3UNet,
        MotionAdapter,
        UNet2DConditionModel,
        UNet2DModel,
        UNet3DConditionModel,
        UNetMotionModel,
        UNetSpatioTemporalConditionModel,
        UVit2DModel,
    )
    from .vq_model import VQModel

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
