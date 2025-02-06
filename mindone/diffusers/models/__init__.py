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
    "autoencoders.autoencoder_kl_cogvideox": ["AutoencoderKLCogVideoX"],
    "autoencoders.autoencoder_kl_temporal_decoder": ["AutoencoderKLTemporalDecoder"],
    "autoencoders.autoencoder_tiny": ["AutoencoderTiny"],
    "autoencoders.consistency_decoder_vae": ["ConsistencyDecoderVAE"],
    "autoencoders.vq_model": ["VQModel"],
    "controlnet": ["ControlNetModel"],
    "controlnet_hunyuan": ["HunyuanDiT2DControlNetModel", "HunyuanDiT2DMultiControlNetModel"],
    "controlnet_sd3": ["SD3ControlNetModel", "SD3MultiControlNetModel"],
    "controlnet_sparsectrl": ["SparseControlNetModel"],
    "controlnet_xs": ["ControlNetXSAdapter", "UNetControlNetXSModel"],
    "dual_transformer_2d": ["DualTransformer2DModel"],
    "embeddings": ["ImageProjection"],
    "modeling_utils": ["ModelMixin"],
    "transformers.auraflow_transformer_2d": ["AuraFlowTransformer2DModel"],
    "transformers.cogvideox_transformer_3d": ["CogVideoXTransformer3DModel"],
    "transformers.dit_transformer_2d": ["DiTTransformer2DModel"],
    "transformers.dual_transformer_2d": ["DualTransformer2DModel"],
    "transformers.hunyuan_transformer_2d": ["HunyuanDiT2DModel"],
    "transformers.latte_transformer_3d": ["LatteTransformer3DModel"],
    "transformers.lumina_nextdit2d": ["LuminaNextDiT2DModel"],
    "transformers.pixart_transformer_2d": ["PixArtTransformer2DModel"],
    "transformers.prior_transformer": ["PriorTransformer"],
    "transformers.t5_film_transformer": ["T5FilmDecoder"],
    "transformers.transformer_2d": ["Transformer2DModel"],
    "transformers.transformer_flux": ["FluxTransformer2DModel"],
    "transformers.transformer_sd3": ["SD3Transformer2DModel"],
    "transformers.transformer_temporal": ["TransformerTemporalModel"],
    "unets.unet_1d": ["UNet1DModel"],
    "unets.unet_2d": ["UNet2DModel"],
    "unets.unet_2d_condition": ["UNet2DConditionModel"],
    "unets.unet_3d_condition": ["UNet3DConditionModel"],
    "unets.unet_i2vgen_xl": ["I2VGenXLUNet"],
    "unets.unet_kandinsky3": ["Kandinsky3UNet"],
    "unets.unet_motion_model": ["MotionAdapter", "UNetMotionModel"],
    "unets.unet_stable_cascade": ["StableCascadeUNet"],
    "unets.unet_spatio_temporal_condition": ["UNetSpatioTemporalConditionModel"],
    "unets.uvit_2d": ["UVit2DModel"],
}

if TYPE_CHECKING:
    from .adapter import MultiAdapter, T2IAdapter
    from .autoencoders import (
        AsymmetricAutoencoderKL,
        AutoencoderKL,
        AutoencoderKLCogVideoX,
        AutoencoderKLTemporalDecoder,
        AutoencoderTiny,
        ConsistencyDecoderVAE,
        VQModel,
    )
    from .controlnet import ControlNetModel
    from .controlnet_hunyuan import HunyuanDiT2DControlNetModel, HunyuanDiT2DMultiControlNetModel
    from .controlnet_sd3 import SD3ControlNetModel, SD3MultiControlNetModel
    from .controlnet_sparsectrl import SparseControlNetModel
    from .controlnet_xs import ControlNetXSAdapter, UNetControlNetXSModel
    from .embeddings import ImageProjection
    from .modeling_utils import ModelMixin
    from .transformers import (
        AuraFlowTransformer2DModel,
        CogVideoXTransformer3DModel,
        DiTTransformer2DModel,
        DualTransformer2DModel,
        FluxTransformer2DModel,
        HunyuanDiT2DModel,
        LatteTransformer3DModel,
        LuminaNextDiT2DModel,
        PixArtTransformer2DModel,
        PriorTransformer,
        SD3Transformer2DModel,
        T5FilmDecoder,
        Transformer2DModel,
        TransformerTemporalModel,
    )
    from .unets import (
        I2VGenXLUNet,
        Kandinsky3UNet,
        MotionAdapter,
        StableCascadeUNet,
        UNet1DModel,
        UNet2DConditionModel,
        UNet2DModel,
        UNet3DConditionModel,
        UNetMotionModel,
        UNetSpatioTemporalConditionModel,
        UVit2DModel,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
