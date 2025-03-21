from typing import Callable, Optional

import mindspore as ms

from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel
from mindone.utils.params import load_checkpoint_to_net

from .models.dc_ae import DCAE, DCAEConfig, dc_ae_f32

__all__ = ["create_dc_ae_model_cfg", "DCAE_HF", "DC_AE"]


REGISTERED_DCAE_MODEL: dict[str, tuple[Callable, Optional[str]]] = {
    "dc-ae-f32t4c128": (dc_ae_f32, None),
}


def create_dc_ae_model_cfg(name: str, pretrained_path: Optional[str] = None) -> DCAEConfig:
    assert name in REGISTERED_DCAE_MODEL, f"{name} is not supported"
    dc_ae_cls, default_pt_path = REGISTERED_DCAE_MODEL[name]
    pretrained_path = default_pt_path if pretrained_path is None else pretrained_path
    model_cfg = dc_ae_cls(name, pretrained_path)
    return model_cfg


class DCAE_HF(DCAE, PreTrainedModel):
    def __init__(self, model_name: str):
        cfg = create_dc_ae_model_cfg(model_name)
        DCAE.__init__(self, cfg)


def DC_AE(
    model_name: str,
    ms_dtype: ms.dtype = ms.bfloat16,
    from_scratch: bool = False,
    from_pretrained: Optional[str] = None,
    is_training: bool = False,
    use_spatial_tiling: bool = False,
    use_temporal_tiling: bool = False,
    spatial_tile_size: int = 256,
    temporal_tile_size: int = 32,
    tile_overlap_factor: float = 0.25,
    scaling_factor: float = None,
    disc_off_grad_ckpt: bool = False,
) -> DCAE_HF:
    if not from_scratch:
        model = DCAE_HF.from_pretrained(model_name).to(ms_dtype)
    else:
        model = DCAE_HF(model_name).to(ms_dtype)

    if from_pretrained is not None:
        load_checkpoint_to_net(model, from_pretrained)
        print(f"loaded dc_ae from ckpt path: {from_pretrained}")

    model.cfg.is_training = is_training
    model.use_spatial_tiling = use_spatial_tiling
    model.use_temporal_tiling = use_temporal_tiling
    model.spatial_tile_size = spatial_tile_size
    model.temporal_tile_size = temporal_tile_size
    model.tile_overlap_factor = tile_overlap_factor
    if scaling_factor is not None:
        model.scaling_factor = scaling_factor
    model.decoder.disc_off_grad_ckpt = disc_off_grad_ckpt
    return model
