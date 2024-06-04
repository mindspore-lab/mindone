import logging

from mindspore import nn

USE_V2 = True
if USE_V2:
    from .modeling_causalvae_v2 import CausalVAEModel_V2 as CausalVAEModel
else:
    from .modeling_causalvae import CausalVAEModel


logger = logging.getLogger(__name__)


class CausalVAEModelWrapper(nn.Cell):
    def __init__(self, model_path, subfolder=None, cache_dir=None, **kwargs):
        super(CausalVAEModelWrapper, self).__init__()
        # if os.path.exists(ckpt):
        # self.vae = CausalVAEModel.load_from_checkpoint(ckpt)
        self.vae = CausalVAEModel.from_pretrained(model_path, subfolder=subfolder, cache_dir=cache_dir, **kwargs)
        # if isinstance(model_config, str) and model_config.endswith(".yaml"):
        #     model_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), model_config)
        #     model_config = OmegaConf.load(model_config)
        # else:
        #     assert isinstance(
        #         model_config, omegaconf.DictConfig
        #     ), "Expect to have model_config as a omegaconf.DictConfi input"

        # vae = instantiate_from_config(model_config.generator)
        # if model_path is not None:
        #     model_path = os.path.join(model_path, subfolder) if subfolder else model_path
        #     if os.path.exists(model_path):
        #         if not model_path.endswith(".ckpt"):
        #             assert os.path.isdir(
        #                 model_path
        #             ), f"Expect model_path to be a directory or a checkpoint file, but got {model_path}"
        #             model_path = glob.glob(os.path.join(model_path, "*.ckpt"))
        #             assert (
        #                 len(model_path) == 1
        #             ), f"Expect one checkpoint file in {model_path}, but got {len(model_path)}"
        #             model_path = model_path[0]
        #         vae.init_from_ckpt(model_path)
        #     else:
        #         logger.info(f"Model path {model_path} not exists")
        # else:
        #     logging.info("Using random initialization for causal vae")
        # self.vae = vae

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
