from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor


class IPAdapterLatentDiffusion(LatentDiffusion):
    def __init__(
        self,
        embedder_config,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.embedder = instantiate_from_config(embedder_config)

    def get_input(self, x, c, c_adm):
        z, c = super().get_input(x, c)
        x = self.transpose(x, (0, 3, 1, 2))
        c = ops.stop_gradient(self.get_learned_conditioning_fortrain(c))
        c_adm = self.embedder(c_adm)
        c = ops.concat([c, c_adm], axis=1)
        return z, c

    def construct(self, x, c, c_adm):
        t = self.uniform_int((x.shape[0],), Tensor(0, dtype=ms.int32), Tensor(self.num_timesteps, dtype=ms.int32))
        x, c = self.get_input(x, c, c_adm)
        return self.p_losses(x, c, t)
