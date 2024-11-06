import logging
import os

import mindspore as ms
from mindspore import nn

from .model.vae import CausalVAEModel, WFVAEModel

logger = logging.getLogger(__name__)


class CausalVAEModelWrapper(nn.Cell):
    def __init__(self, model_path, subfolder=None, cache_dir=None, use_ema=False, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        self.vae, loading_info = CausalVAEModel.from_pretrained(
            model_path, subfolder=subfolder, cache_dir=cache_dir, output_loading_info=True, **kwargs
        )
        logger.info(loading_info)
        if use_ema:
            self.vae.init_from_ema(model_path)
            self.vae = self.vae.ema

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


class WFVAEModelWrapper(nn.Cell):
    def __init__(self, model_path=None, dtype=ms.float32, subfolder=None, cache_dir=None, vae=None, **kwargs):
        super(WFVAEModelWrapper, self).__init__()
        assert model_path is not None or vae is not None, "At least oen of [`model_path`, `vae`] should be provided."

        if vae is not None:
            self.vae = vae
        else:
            assert model_path is not None, "When `vae` is not None, expect to get `model_path`!"
            assert os.path.exists(model_path), f"`model_path` does not exist!: {model_path}"
            self.vae = WFVAEModel.from_pretrained(
                model_path, subfolder=subfolder, cache_dir=cache_dir, dtype=dtype, **kwargs
            )
        self.shift = ms.Tensor(self.vae.config.shift)[None, :, None, None, None]
        self.scale = ms.Tensor(self.vae.config.scale)[None, :, None, None, None]

    def encode(self, x):
        x = (self.vae.encode(x) - self.shift.to(dtype=x.dtype)) * self.scale.to(dtype=x.dtype)
        return x

    def decode(self, x):
        x = x / self.scale.to(dtype=x.dtype) + self.shift.to(dtype=x.dtype)
        x = self.vae.decode(x)
        # b c t h w -> b t c h w
        x = x.transpose(0, 2, 1, 3, 4)
        return x

    def dtype(self):
        return self.vae.dtype


ae_wrapper = {
    "CausalVAEModel_D4_2x8x8": CausalVAEModelWrapper,
    "CausalVAEModel_D8_2x8x8": CausalVAEModelWrapper,
    "CausalVAEModel_D4_4x8x8": CausalVAEModelWrapper,
    "CausalVAEModel_D8_4x8x8": CausalVAEModelWrapper,
    "WFVAEModel_D8_4x8x8": WFVAEModelWrapper,
    "WFVAEModel_D16_4x8x8": WFVAEModelWrapper,
    "WFVAEModel_D32_4x8x8": WFVAEModelWrapper,
    "WFVAEModel_D32_8x8x8": WFVAEModelWrapper,
}

ae_stride_config = {
    "CausalVAEModel_D4_2x8x8": [2, 8, 8],
    "CausalVAEModel_D8_2x8x8": [2, 8, 8],
    "CausalVAEModel_D4_4x8x8": [4, 8, 8],
    "CausalVAEModel_D8_4x8x8": [4, 8, 8],
    "WFVAEModel_D8_4x8x8": [4, 8, 8],
    "WFVAEModel_D16_4x8x8": [4, 8, 8],
    "WFVAEModel_D32_4x8x8": [4, 8, 8],
    "WFVAEModel_D32_8x8x8": [8, 8, 8],
}

ae_channel_config = {
    "CausalVAEModel_D4_2x8x8": 4,
    "CausalVAEModel_D8_2x8x8": 8,
    "CausalVAEModel_D4_4x8x8": 4,
    "CausalVAEModel_D8_4x8x8": 8,
    "WFVAEModel_D8_4x8x8": 8,
    "WFVAEModel_D16_4x8x8": 16,
    "WFVAEModel_D32_4x8x8": 32,
    "WFVAEModel_D32_8x8x8": 32,
}

ae_denorm = {
    "CausalVAEModel_D4_2x8x8": lambda x: (x + 1.0) / 2.0,
    "CausalVAEModel_D8_2x8x8": lambda x: (x + 1.0) / 2.0,
    "CausalVAEModel_D4_4x8x8": lambda x: (x + 1.0) / 2.0,
    "CausalVAEModel_D8_4x8x8": lambda x: (x + 1.0) / 2.0,
    "WFVAEModel_D8_4x8x8": lambda x: (x + 1.0) / 2.0,
    "WFVAEModel_D16_4x8x8": lambda x: (x + 1.0) / 2.0,
    "WFVAEModel_D32_4x8x8": lambda x: (x + 1.0) / 2.0,
    "WFVAEModel_D32_8x8x8": lambda x: (x + 1.0) / 2.0,
}

ae_norm = {
    "CausalVAEModel_D4_2x8x8": lambda x: 2.0 * x - 1.0,
    "CausalVAEModel_D8_2x8x8": lambda x: 2.0 * x - 1.0,
    "CausalVAEModel_D4_4x8x8": lambda x: 2.0 * x - 1.0,
    "CausalVAEModel_D8_4x8x8": lambda x: 2.0 * x - 1.0,
    "WFVAEModel_D8_4x8x8": lambda x: 2.0 * x - 1.0,
    "WFVAEModel_D16_4x8x8": lambda x: 2.0 * x - 1.0,
    "WFVAEModel_D32_4x8x8": lambda x: 2.0 * x - 1.0,
    "WFVAEModel_D32_8x8x8": lambda x: 2.0 * x - 1.0,
}
