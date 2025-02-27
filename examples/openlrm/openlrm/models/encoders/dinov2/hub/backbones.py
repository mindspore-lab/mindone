# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import os
from enum import Enum
from typing import Union

from openlrm.utils.convert_models import torch_to_ms_weight

from mindone.utils.params import load_checkpoint_to_net, load_param_into_net_with_filter

from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name


class Weights(Enum):
    LVD142M = "LVD142M"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from ..models import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        # load mindspore ckpt first
        model_full_name = _make_dinov2_model_name(arch_name, patch_size, num_register_tokens)
        ckpt_dir = os.path.join("checkpoints", f"{model_full_name}_pretrain.ckpt")
        if os.path.isfile(ckpt_dir):
            load_checkpoint_to_net(
                model, ckpt_dir, ignore_net_params_not_loaded=False, ensure_all_ckpt_params_loaded=True
            )
        else:  # otherwise load torch ckpt, then convert to mindspore ckpt
            import torch

            url = _DINOV2_BASE_URL + f"/{model_base_name}/{model_full_name}_pretrain.pth"
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu")
            # ********** Modified by Zexin He in 2023-2024 **********
            state_dict = {k: v for k, v in state_dict.items() if "mask_token" not in k}  # DDP concern
            if vit_kwargs.get("modulation_dim") is not None:
                state_dict = {
                    k.replace("norm1", "norm1.norm").replace("norm2", "norm2.norm"): v for k, v in state_dict.items()
                }
            # ********************************************************

            state_dict_ms = torch_to_ms_weight(
                source_fp=None, target_fp=ckpt_dir, source_data=state_dict
            )  # convert and save ms ckpt
            load_param_into_net_with_filter(model, state_dict_ms, strict_load=True)

    return model


def dinov2_vits14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_small", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitb14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitl14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs)


def dinov2_vitg14(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(*, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs):
    """
    DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
