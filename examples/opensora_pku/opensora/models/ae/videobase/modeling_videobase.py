import mindspore as ms

from mindone.diffusers import ModelMixin
from mindone.diffusers.configuration_utils import ConfigMixin


class VideoBaseAE(ModelMixin, ConfigMixin):
    config_name = "config.json"
    _supports_gradient_checkpointing = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # @classmethod
    # def load_from_checkpoint(cls, model_path):
    #     with open(os.path.join(model_path, "config.json"), "r") as file:
    #         config = json.load(file)
    #     state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location="cpu")
    #     if 'state_dict' in state_dict:
    #         state_dict = state_dict['state_dict']
    #     model = cls(config=cls.CONFIGURATION_CLS(**config))
    #     model.load_state_dict(state_dict)
    #     return model

    @classmethod
    def download_and_load_model(cls, model_name, cache_dir=None):
        pass

    def encode(self, x: ms.Tensor, *args, **kwargs):
        pass

    def decode(self, encoding: ms.Tensor, *args, **kwargs):
        pass
