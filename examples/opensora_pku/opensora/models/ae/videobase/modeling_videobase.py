import mindspore as ms

from mindone.diffusers import ModelMixin
from mindone.diffusers.configuration_utils import ConfigMixin


class VideoBaseAE(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        pass

    def encode(self, x: ms.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: ms.Tensor, *args, **kwargs):
        pass
