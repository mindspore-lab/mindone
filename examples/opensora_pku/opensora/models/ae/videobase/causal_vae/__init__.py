import os

from omegaconf import OmegaConf

from mindspore import nn

from mindone.utils.config import instantiate_from_config

from .. import videobase_ae_yaml


class CausalVAEModelWrapper(nn.Cell):
    def __init__(self, model_name="CausalVAEModel_4x8x8"):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        # self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir)
        model_config = videobase_ae_yaml[model_name]
        model_config = os.path.join(os.path.abspath(__file__), model_config)
        self.vae = config = OmegaConf.load(model_config)
        self.vae = instantiate_from_config(config.generator)

    def encode(self, x):  # b c t h w
        # x = self.vae.encode(x).sample()
        x = self.vae.encode(x).sample() * 0.18215
        return x

    def decode(self, x):
        # x = self.vae.decode(x)
        x = self.vae.decode(x / 0.18215)
        # b c t h w -> b t c h w
        x = x.permute(0, 2, 1, 3, 4)
        return x

    def dtype(self):
        return self.vae.dtype
