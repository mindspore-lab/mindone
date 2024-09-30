import os, sys
import mindspore as ms
import numpy as np
from mindspore import context

# GRAPH_MODE 0
# PYNATIVE_MODE 1

context.set_context(mode=1, device_target="CPU", device_id=1)

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))


from utils.lora_handler import LoraHandler
from utils.common_utils import load_model_checkpoint
from utils.utils import instantiate_from_config
from omegaconf import OmegaConf


MODEL_URL = "https://weights.replicate.delivery/default/Ji4chenLi/t2v-turbo.tar"
MODEL_CACHE = "checkpoints"


base_model_dir = os.path.join(MODEL_CACHE, "VideoCrafter2_model.ckpt")
unet_dir = os.path.join(MODEL_CACHE, "lora_np.pkl")

config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
model_config = config.pop("model", OmegaConf.create())
# pretrained_t2v = instantiate_from_config(model_config)
# pretrained_t2v = load_model_checkpoint(pretrained_t2v, base_model_dir)

unet_config = model_config["params"]["unet_config"]
unet_config["params"]["time_cond_proj_dim"] = 256
unet = instantiate_from_config(unet_config)


use_unet_lora = True
lora_manager = LoraHandler(
    version="cloneofsimo",
    use_unet_lora=use_unet_lora,
    save_for_webui=True,
    unet_replace_modules=["UNetModel"],
)
lora_manager.add_lora_to_model(
    use_unet_lora,
    unet,
    lora_manager.unet_replace_modules,
    lora_path=unet_dir,
    dropout=0.1,
    r=64,
)

