import logging
import os

from omegaconf import OmegaConf

from mindspore import nn

from mindone.utils.config import instantiate_from_config

logger = logging.getLogger(__name__)


class CausalVAEModelWrapper(nn.Cell):
    def __init__(self, model_config="causal_vae_488.yaml", model_path=None):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        # self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir)
        model_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_config)
        if isinstance(model_config, str) and model_config.endswith(".yaml"):
            model_config = OmegaConf.load(model_config)
        else:
            assert isinstance(model_config, OmegaConf), "Expect to have model_config as a OmegaConf input"

        vae = instantiate_from_config(model_config.generator)
        if model_path is not None:
            if os.path.exists(model_path):
                vae.init_from_ckpt(model_path)
            else:
                logger.info(f"Model path {model_path} not exists")
        else:
            logging.info("Using random initialization for causal vae")
        self.vae = vae

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
