from typing import TYPE_CHECKING

from ..utils import _LazyModule, deprecate


def text_encoder_lora_state_dict(text_encoder):
    deprecate(
        "text_encoder_load_state_dict in `models`",
        "0.27.0",
        "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",  # noqa: E501
    )
    state_dict = {}

    for name, module in text_encoder_attn_modules(text_encoder):
        for k, v in module.q_proj.lora_linear_layer.parameters_and_names():
            state_dict[f"{name}.q_proj.lora_linear_layer.{k}"] = v

        for k, v in module.k_proj.lora_linear_layer.parameters_and_names():
            state_dict[f"{name}.k_proj.lora_linear_layer.{k}"] = v

        for k, v in module.v_proj.lora_linear_layer.parameters_and_names():
            state_dict[f"{name}.v_proj.lora_linear_layer.{k}"] = v

        for k, v in module.out_proj.lora_linear_layer.parameters_and_names():
            state_dict[f"{name}.out_proj.lora_linear_layer.{k}"] = v

    return state_dict


def text_encoder_attn_modules(text_encoder):
    deprecate(
        "text_encoder_attn_modules in `models`",
        "0.27.0",
        "`text_encoder_lora_state_dict` is deprecated and will be removed in 0.27.0. Make sure to retrieve the weights using `get_peft_model`. See https://huggingface.co/docs/peft/v0.6.2/en/quicktour#peftmodel for more information.",  # noqa: E501
    )
    from mindone.transformers import CLIPTextModel, CLIPTextModelWithProjection

    attn_modules = []

    if isinstance(text_encoder, (CLIPTextModel, CLIPTextModelWithProjection)):
        for i, layer in enumerate(text_encoder.text_model.encoder.layers):
            name = f"text_model.encoder.layers.{i}.self_attn"
            mod = layer.self_attn
            attn_modules.append((name, mod))
    else:
        raise ValueError(f"do not know how to get attention modules for: {text_encoder.__class__.__name__}")

    return attn_modules


_import_structure = {
    "single_file_model": ["FromOriginalModelMixin"],
    "ip_adapter": ["IPAdapterMixin"],
    "lora_pipeline": [
        "AmusedLoraLoaderMixin",
        "StableDiffusionLoraLoaderMixin",
        "SD3LoraLoaderMixin",
        "StableDiffusionXLLoraLoaderMixin",
        "LoraLoaderMixin",
        "FluxLoraLoaderMixin",
        "CogVideoXLoraLoaderMixin",
    ],
    "peft": ["PeftAdapterMixin"],
    "single_file": ["FromSingleFileMixin"],
    "textual_inversion": ["TextualInversionLoaderMixin"],
    "unet": ["UNet2DConditionLoadersMixin"],
}


if TYPE_CHECKING:
    from .ip_adapter import IPAdapterMixin
    from .lora_pipeline import (
        AmusedLoraLoaderMixin,
        CogVideoXLoraLoaderMixin,
        FluxLoraLoaderMixin,
        LoraLoaderMixin,
        SD3LoraLoaderMixin,
        StableDiffusionLoraLoaderMixin,
        StableDiffusionXLLoraLoaderMixin,
    )
    from .peft import PeftAdapterMixin
    from .single_file import FromSingleFileMixin
    from .single_file_model import FromOriginalModelMixin
    from .textual_inversion import TextualInversionLoaderMixin
    from .unet import UNet2DConditionLoadersMixin
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
