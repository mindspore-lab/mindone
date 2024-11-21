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
import importlib
import logging
import os
from inspect import isfunction

from omegaconf import OmegaConf
from packaging import version
from PIL import Image, ImageDraw, ImageFont

import mindspore as ms
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net

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


def load_pretrained_model(pretrained_ckpt, net, unet_initialize_random=False):
    _logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)

        if unet_initialize_random:
            pnames = list(param_dict.keys())
            # pop unet params from pretrained weight
            for pname in pnames:
                if pname.startswith("model.diffusion_model"):
                    param_dict.pop(pname)
            print("UNet will be initialized randomly")

        if is_old_ms_version():
            param_not_load = load_param_into_net(net, param_dict)
        else:
            param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)
        _logger.info("Params not load: {}".format(param_not_load))
    else:
        _logger.warning("Checkpoint file not exists!!!")
