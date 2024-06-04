import logging
import os

from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from utils import model_utils

import mindspore as ms

logger = logging.getLogger("controlnet_model")


# TODO: decide to delete this func
def get_state_dict(d):
    return d.get("state_dict", d)


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    logger.info(f"Loaded model config from [{config_path}]")

    return model


def load_model(model, ckpt_path):
    # TODO: consider add load lora ckpt part

    logger.info(f"Loading model from {ckpt_path}")

    if os.path.exists(ckpt_path):
        param_dict = ms.load_checkpoint(ckpt_path)
        if param_dict:
            param_not_load, ckpt_not_load = model_utils.load_param_into_net_with_filter(model, param_dict, filter=None)
            if len(param_not_load) > 0:
                logger.info("Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")]))
    else:
        logger.warning(f"!!!Warning!!!: {ckpt_path} doesn't exist")

    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False

    return model


def load_state_dict(model, ckpt_path):
    logger.info(f"Loading model from {ckpt_path}")
    state_dict = get_state_dict(ms.load_checkpoint(ckpt_path))
    logger.info(f"Loaded state_dict from [{ckpt_path}]")

    model.set_train(False)
    for param in model.trainable_params():
        param.requires_grad = False
    return state_dict
