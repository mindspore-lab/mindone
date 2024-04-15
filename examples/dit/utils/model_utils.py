import argparse
import logging

import mindspore as ms

from mindone.utils.params import load_param_into_net_with_filter
from mindone.utils.version_control import is_old_ms_version

logger = logging.getLogger(__name__)


def remove_pname_prefix(param_dict, prefix="network."):
    # replace the prefix of param dict
    new_param_dict = {}
    for pname in param_dict:
        if pname.startswith(prefix):
            new_pname = pname[len(prefix) :]
        else:
            new_pname = pname
        new_param_dict[new_pname] = param_dict[pname]
    return new_param_dict


def load_dit_ckpt_params(model, ckpt):
    if isinstance(ckpt, str):
        logger.info(f"Loading {ckpt} params into DiT model...")
        param_dict = ms.load_checkpoint(ckpt)
    else:
        param_dict = ckpt

    keys_excluding_pos_embed = [key for key in param_dict.keys() if "pos_embed" != key]

    if is_old_ms_version():
        param_not_load = ms.load_param_into_net(model, param_dict, filter=keys_excluding_pos_embed)
    else:
        param_not_load, ckpt_not_load = load_param_into_net_with_filter(
            model, param_dict, filter=keys_excluding_pos_embed
        )

    logger.info("Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load)))
    logger.info("Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load)))
    return model


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def count_params(model, verbose=False):
    total_params = sum([param.size for param in model.get_parameters()])
    trainable_params = sum([param.size for param in model.get_parameters() if param.requires_grad])

    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params, trainable_params


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")
