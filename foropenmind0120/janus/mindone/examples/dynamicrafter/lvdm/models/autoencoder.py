import sys

sys.path.append("../stable_diffusion_v2")

from ldm.models.autoencoder import AutoencoderKL as AutoencoderKL_SD

__all__ = ["AutoencoderKL"]


class AutoencoderKL(AutoencoderKL_SD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
