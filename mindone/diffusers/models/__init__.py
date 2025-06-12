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
    "autoencoders.autoencoder_dc": ["AutoencoderDC"],
    "autoencoders.autoencoder_kl": ["AutoencoderKL"],
    "autoencoders.autoencoder_kl_allegro": ["AutoencoderKLAllegro"],
    "autoencoders.autoencoder_kl_cogvideox": ["AutoencoderKLCogVideoX"],
    "autoencoders.autoencoder_kl_hunyuan_video": ["AutoencoderKLHunyuanVideo"],
    "autoencoders.autoencoder_kl_ltx": ["AutoencoderKLLTXVideo"],
    "autoencoders.autoencoder_kl_magvit": ["AutoencoderKLMagvit"],
    "autoencoders.autoencoder_kl_temporal_decoder": ["AutoencoderKLTemporalDecoder"],
    "autoencoders.autoencoder_kl_wan": ["AutoencoderKLWan"],
    "autoencoders.autoencoder_oobleck": ["AutoencoderOobleck"],
    "autoencoders.autoencoder_tiny": ["AutoencoderTiny"],
    "autoencoders.consistency_decoder_vae": ["ConsistencyDecoderVAE"],
    "autoencoders.vq_model": ["VQModel"],
    "autoencoders.autoencoder_kl_mochi": ["AutoencoderKLMochi"],
    "controlnets.controlnet": ["ControlNetModel"],
    "controlnets.controlnet_flux": ["FluxControlNetModel", "FluxMultiControlNetModel"],
    "controlnets.controlnet_hunyuan": [
        "HunyuanDiT2DControlNetModel",
        "HunyuanDiT2DMultiControlNetModel",
    ],
    "controlnets.controlnet_sd3": ["SD3ControlNetModel", "SD3MultiControlNetModel"],
    "controlnets.controlnet_sparsectrl": ["SparseControlNetModel"],
    "controlnets.controlnet_union": ["ControlNetUnionModel"],
    "controlnets.controlnet_xs": ["ControlNetXSAdapter", "UNetControlNetXSModel"],
    "controlnets.multicontrolnet": ["MultiControlNetModel"],
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
    "transformers.sana_transformer": ["SanaTransformer2DModel"],
    "transformers.stable_audio_transformer": ["StableAudioDiTModel"],
    "transformers.t5_film_transformer": ["T5FilmDecoder"],
    "transformers.transformer_2d": ["Transformer2DModel"],
    "transformers.transformer_allegro": ["AllegroTransformer3DModel"],
    "transformers.transformer_cogview3plus": ["CogView3PlusTransformer2DModel"],
    "transformers.transformer_cogview4": ["CogView4Transformer2DModel"],
    "transformers.transformer_easyanimate": ["EasyAnimateTransformer3DModel"],
    "transformers.transformer_flux": ["FluxTransformer2DModel"],
    "transformers.transformer_hunyuan_video": ["HunyuanVideoTransformer3DModel"],
    "transformers.transformer_ltx": ["LTXVideoTransformer3DModel"],
    "transformers.transformer_lumina2": ["Lumina2Transformer2DModel"],
    "transformers.transformer_sd3": ["SD3Transformer2DModel"],
    "transformers.transformer_temporal": ["TransformerTemporalModel"],
    "transformers.transformer_wan": ["WanTransformer3DModel"],
    "transformers.transformer_mochi": ["MochiTransformer3DModel"],
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
        AutoencoderDC,
        AutoencoderKL,
        AutoencoderKLAllegro,
        AutoencoderKLCogVideoX,
        AutoencoderKLHunyuanVideo,
        AutoencoderKLLTXVideo,
        AutoencoderKLMagvit,
        AutoencoderKLMochi,
        AutoencoderKLTemporalDecoder,
        AutoencoderKLWan,
        AutoencoderOobleck,
        AutoencoderTiny,
        ConsistencyDecoderVAE,
        VQModel,
    )
    from .controlnets import (
        ControlNetModel,
        ControlNetUnionModel,
        ControlNetXSAdapter,
        FluxControlNetModel,
        FluxMultiControlNetModel,
        HunyuanDiT2DControlNetModel,
        HunyuanDiT2DMultiControlNetModel,
        MultiControlNetModel,
        SD3ControlNetModel,
        SD3MultiControlNetModel,
        SparseControlNetModel,
        UNetControlNetXSModel,
    )
    from .embeddings import ImageProjection
    from .modeling_utils import ModelMixin
    from .transformers import (
        AllegroTransformer3DModel,
        AuraFlowTransformer2DModel,
        CogVideoXTransformer3DModel,
        CogView3PlusTransformer2DModel,
        CogView4Transformer2DModel,
        DiTTransformer2DModel,
        DualTransformer2DModel,
        EasyAnimateTransformer3DModel,
        FluxTransformer2DModel,
        HunyuanDiT2DModel,
        HunyuanVideoTransformer3DModel,
        LatteTransformer3DModel,
        LTXVideoTransformer3DModel,
        Lumina2Transformer2DModel,
        LuminaNextDiT2DModel,
        MochiTransformer3DModel,
        PixArtTransformer2DModel,
        PriorTransformer,
        SanaTransformer2DModel,
        SD3Transformer2DModel,
        StableAudioDiTModel,
        T5FilmDecoder,
        Transformer2DModel,
        TransformerTemporalModel,
        WanTransformer3DModel,
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
