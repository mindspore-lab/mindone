import importlib
import logging
import os

from omegaconf import OmegaConf

from mindspore import dtype as mstype
from mindspore import load_checkpoint, load_param_into_net, nn, ops


def build_model_from_config(config, enable_flash_attention=None, args=None):
    config = OmegaConf.load(config).model
    if args is not None:
        if enable_flash_attention is not None:
            config["params"]["unet_config"]["params"]["enable_flash_attention"] = enable_flash_attention
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    # config_params['cond_stage_trainable'] = cond_stage_trainable # TODO: easy config
    return get_obj_from_str(config["target"])(**config_params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_pretrained_model(pretrained_ckpt, net, unet_initialize_random=False):
    logging.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)

        if unet_initialize_random:
            pnames = list(param_dict.keys())
            # pop unet params from pretrained weight
            for pname in pnames:
                if pname.startswith("model.diffusion_model"):
                    param_dict.pop(pname)
            logging.warning("UNet will be initialized randomly")

        param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)
        logging.info("Params not load: {}".format(param_not_load))
    else:
        logging.warning(f"Checkpoint file {pretrained_ckpt} dose not exist!!!")


def replace_unet_conv_in(latent_diffusion_with_loss, use_fp16):
    """
    Replace the first conv_in layer of Unet with 8 channels and halve the weights.
    """
    # copy weights and bias
    _weight = latent_diffusion_with_loss.model.diffusion_model.input_blocks[0][0].conv.weight.copy()  # [320, 4, 3, 3]
    _bias = latent_diffusion_with_loss.model.diffusion_model.input_blocks[0][0].conv.bias.copy()  # [320]
    # double the channel and halve the weights
    _weight = ops.Tile()(_weight, (1, 2, 1, 1))
    _weight = ops.Mul()(_weight, 0.5)
    # create new conv_in layer
    _n_convin_out_channel = latent_diffusion_with_loss.model.diffusion_model.input_blocks[0][0].conv.out_channels
    if use_fp16:
        _new_conv_in = nn.Conv2d(8, _n_convin_out_channel, 3, pad_mode="pad", padding=1, has_bias=True).to_float(
            mstype.float16
        )
    else:
        _new_conv_in = nn.Conv2d(8, _n_convin_out_channel, 3, pad_mode="pad", padding=1, has_bias=True).to_float(
            mstype.float32
        )
    _new_conv_in.weight.set_data(_weight)
    _new_conv_in.bias.set_data(_bias)
    latent_diffusion_with_loss.model.diffusion_model.input_blocks[0][0].conv = _new_conv_in
    print("Unet conv_in layer is replaced")
    # update in_channels in Unet config
    latent_diffusion_with_loss.model.diffusion_model.in_channels = 8
    print("Unet config is updated")
    return
