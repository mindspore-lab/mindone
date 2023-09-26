# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import copy
import importlib
import logging
import os
from inspect import isfunction

from omegaconf import OmegaConf
from packaging import version
from PIL import Image, ImageDraw, ImageFont

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter, load_checkpoint, load_param_into_net
from mindspore import log as logger
from mindspore.train.serialization import _load_dismatch_prefix_params, _update_param

_logger = logging.getLogger(__name__)


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype("data/DejaVuSans.ttf", size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = ms.numpy.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = ms.numpy.stack(txts)
    txts = ms.Tensor(txts)
    return txts


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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


def instantiate_from_config(config):
    if isinstance(config, str):
        config = OmegaConf.load(config).model
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def extract_into_tensor(a, t, x_shape):
    b = t.shape[0]
    out = ops.GatherD()(a, -1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def is_old_ms_version(last_old_version="1.10.1"):
    # some APIs are changed after ms 1.10.1 version, such as dropout
    return version.parse(ms.__version__) <= version.parse(last_old_version)


def load_pretrained_model(pretrained_ckpt, net):
    _logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)
        param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)
        _logger.info("Params not load: {}".format(param_not_load))
    else:
        _logger.warning("Checkpoint file not exists!!!")


def load_model(model, ckpt_fp, verbose=True, filter=None):
    if os.path.exists(ckpt_fp):
        param_dict = ms.load_checkpoint(ckpt_fp)
        if param_dict:
            param_not_load, ckpt_not_load = load_param_into_net_with_filter(model, param_dict, filter=filter)
            if verbose:
                if len(param_not_load) > 0:
                    logger.info(
                        "Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")])
                    )
    else:
        logger.error(f"!!!Error!!!: {ckpt_fp} doesn't exist")
        raise FileNotFoundError(f"{ckpt_fp} doesn't exist")


def load_param_into_net_with_filter(net, parameter_dict, strict_load=False, filter=None):
    """
    Load parameters into network, return parameter list that are not loaded in the network.

    Args:
        net (Cell): The network where the parameters will be loaded.
        parameter_dict (dict): The dictionary generated by load checkpoint file,
                               it is a dictionary consisting of key: parameters's name, value: parameter.
        strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            into net when parameter name's suffix in checkpoint file is the same as the
                            parameter in the network. When the types are inconsistent perform type conversion
                            on the parameters of the same type, such as float32 to float16. Default: False.
        filter (List): If not None, it will only load the parameters in the given list. Default: None.

    Returns:
        param_not_load (List), the parameter name in model which are not loaded into the network.
        ckpt_not_load (List), the parameter name in checkpoint file which are not loaded into the network.

    Raises:
        TypeError: Argument is not a Cell, or parameter_dict is not a Parameter dictionary.

    Examples:
        >>> import mindspore as ms
        >>>
        >>> net = Net()
        >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
        >>> param_dict = ms.load_checkpoint(ckpt_file_name, filter_prefix="conv1")
        >>> param_not_load, _ = ms.load_param_into_net(net, param_dict)
        >>> print(param_not_load)
        ['conv1.weight']
    """
    if not isinstance(net, nn.Cell):
        logger.critical("Failed to combine the net and the parameters.")
        msg = "For 'load_param_into_net', the argument 'net' should be a Cell, but got {}.".format(type(net))
        raise TypeError(msg)

    if not isinstance(parameter_dict, dict):
        logger.critical("Failed to combine the net and the parameters.")
        msg = "For 'load_param_into_net', the argument 'parameter_dict' should be a dict, " "but got {}.".format(
            type(parameter_dict)
        )
        raise TypeError(msg)
    for key, value in parameter_dict.items():
        if not isinstance(key, str) or not isinstance(value, (Parameter, str)):
            logger.critical("Load parameters into net failed.")
            msg = (
                "For 'parameter_dict', the element in the argument 'parameter_dict' should be a "
                "'str' and 'Parameter' , but got {} and {}.".format(type(key), type(value))
            )
            raise TypeError(msg)

    # TODO: replace by otherway to do check_bool
    # strict_load = Validator.check_bool(strict_load)
    logger.info("Execute the process of loading parameters into net.")
    net.init_parameters_data()
    param_not_load = []
    ckpt_not_load = list(parameter_dict.keys())
    for _, param in net.parameters_and_names():
        if param.name in parameter_dict:
            new_param = copy.deepcopy(parameter_dict[param.name])
            _update_param(param, new_param, strict_load)
            ckpt_not_load.remove(param.name)
        else:
            param_not_load.append(param.name)

    if param_not_load and not strict_load:
        _load_dismatch_prefix_params(net, parameter_dict, param_not_load, strict_load)

    logger.info("Loading parameters into net is finished.")
    if filter:
        param_all_load_flag = len(set(param_not_load).intersection(set(filter))) == 0
        if param_all_load_flag:
            param_not_load.clear()
    if param_not_load:
        logger.warning(
            "For 'load_param_into_net', "
            "{} parameters in the 'net' are not loaded, because they are not in the "
            "'parameter_dict', please check whether the network structure is consistent "
            "when training and loading checkpoint.".format(len(param_not_load))
        )
        for param_name in param_not_load:
            logger.warning("{} is not loaded.".format(param_name))
    return param_not_load, ckpt_not_load
