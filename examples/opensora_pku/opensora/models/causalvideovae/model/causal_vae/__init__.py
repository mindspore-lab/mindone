import logging

from mindspore import nn

from .modeling_causalvae import CausalVAEModel

logger = logging.getLogger(__name__)


class CausalVAEModelWrapper(nn.Cell):
    def __init__(self, model_path, subfolder=None, cache_dir=None, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)

    def encode(self, x):  # b c t h w
        # x = self.vae.encode(x)
        x = self.vae.encode(x) * 0.18215
        return x

    def decode(self, x):
        # x = self.vae.decode(x)
        x = self.vae.decode(x / 0.18215)
        # b c t h w -> b t c h w
        x = x.permute(0, 2, 1, 3, 4)
        return x

    def dtype(self):
        return self.vae.dtype
