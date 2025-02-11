from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

from omegaconf import DictConfig

import mindspore as ms
from mindspore import nn

from mindone.utils.amp import auto_mixed_precision


def create_model(
    config: DictConfig,
    checkpoints: Union[str, List[str]] = "",
    freeze: bool = False,
    load_filter: bool = False,
    param_fp16: bool = False,
    amp_level: Literal["O0", "O1", "O2", "O3"] = "O0",
    load_first_stage_model: bool = True,
    load_conditioner: bool = True,
):
    from models.diffusion import VideoDiffusionEngine

    assert config.model["target"] in [
        "models.diffusion.VideoDiffusionEngine",
    ], f"Not supported for `class {config.model['target']}`"

    # create diffusion engine
    config.model["params"]["load_first_stage_model"] = load_first_stage_model
    config.model["params"]["load_conditioner"] = load_conditioner
    target_map = {"models.diffusion.VideoDiffusionEngine": VideoDiffusionEngine}
    svd = target_map[config.model["target"]](**config.model.get("params", dict()))

    # load pretrained
    svd.load_pretrained(checkpoints)

    # set auto-mix-precision
    svd = auto_mixed_precision(svd, amp_level=amp_level)
    svd.set_train(False)

    # set model parameter/weight dtype to fp16
    if param_fp16:
        print(
            "!!! WARNING: Converted the weight to `fp16`, that may lead to unstable training. You can turn it off by setting `--param_fp16=False`"
        )
        convert_svd_to_fp16(svd)

    if freeze:
        svd.set_train(False)
        svd.set_grad(False)
        for _, p in svd.parameters_and_names():
            p.requires_grad = False

    if load_filter:
        # TODO: Add DeepFloydDataFiltering
        raise NotImplementedError

    return svd, None


def convert_svd_to_fp16(svd):
    vae = svd.first_stage_model
    text_encoders = svd.conditioner
    unet = svd.model

    convert_to_fp16(vae)
    convert_to_fp16(text_encoders)
    convert_to_fp16(unet)


def convert_to_fp16(model, keep_norm_fp32=True):
    if model is not None:
        assert isinstance(model, nn.Cell)

        k_num, c_num = 0, 0
        for _, p in model.parameters_and_names():
            # filter norm/embedding position_ids param
            if keep_norm_fp32 and ("norm" in p.name):
                # print(f"param {p.name} keep {p.dtype}") # disable print
                k_num += 1
            elif "position_ids" in p.name:
                k_num += 1
            else:
                c_num += 1
                p.set_dtype(ms.float16)

        print(f"Convert `{type(model).__name__}` param to fp16, keep/modify num {k_num}/{c_num}.")

    return model
