from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config


class IPAdapterLatentDiffusion(LatentDiffusion):
    def __init__(
        self,
        embedder_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedder = instantiate_from_config(embedder_config)
