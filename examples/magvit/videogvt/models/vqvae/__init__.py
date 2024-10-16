import logging

import mindspore as ms

from .discriminator import StyleGANDiscriminator
from .vqvae import VQVAE_2D, VQVAE_3D

logger = logging.getLogger(__name__)


def build_model(model_name, model_config, is_training=True, pretrained=None, dtype=ms.float32):
    if model_name == "vqvae-2d":
        model = VQVAE_2D(
            model_config,
            is_training=is_training,
            dtype=dtype,
        )
    elif model_name == "vqvae-3d":
        model = VQVAE_3D(
            model_config,
            is_training=is_training,
            dtype=dtype,
        )
    else:
        raise NotImplementedError(f"{model_name} is not implemented.")

    if pretrained is not None:
        param_dict = ms.load_checkpoint(pretrained)
        ms.load_param_into_net(model, param_dict)
        logger.info(f"Loading vqvae from {pretrained}.")

    return model
