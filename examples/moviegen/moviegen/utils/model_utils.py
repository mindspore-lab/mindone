import logging
from typing import Dict, Literal, Optional, Union

from jsonargparse.typing import Path_fr
from moviegen import LlamaModel, llama3_1B, llama3_5B, llama3_30B

import mindspore as ms
from mindspore import _no_grad, jit_class, nn

__all__ = ["MODEL_DTYPE", "load_ckpt_params", "no_grad", "init_model"]

logger = logging.getLogger(__name__)

MODEL_SPEC = {"llama-1B": llama3_1B, "llama-5B": llama3_5B, "llama-30B": llama3_30B}

MODEL_DTYPE = {
    "fp32": ms.float32,
    "fp16": ms.float16,
    "bf16": ms.bfloat16,
}


def load_ckpt_params(model: nn.Cell, ckpt: Union[str, Dict]) -> nn.Cell:
    if isinstance(ckpt, str):
        logger.info(f"Loading {ckpt} params into network...")
        param_dict = ms.load_checkpoint(ckpt)
    else:
        param_dict = ckpt

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
    if not (len(param_not_load) == len(ckpt_not_load) == 0):
        logger.warning(
            "Exist ckpt params not loaded: {} (total: {}), or net params not loaded: {} (total: {})".format(
                ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
            )
        )
    return model


@jit_class
class no_grad(_no_grad):
    """
    A context manager that suppresses gradient memory allocation in PyNative mode.
    """

    def __init__(self):
        super().__init__()
        self._pynative = ms.get_context("mode") == ms.PYNATIVE_MODE

    def __enter__(self):
        if self._pynative:
            super().__enter__()

    def __exit__(self, *args):
        if self._pynative:
            super().__exit__(*args)


def init_model(
    name: Literal["llama-1B", "llama-5B", "llama-30B"],
    in_channels: int = 4,
    pretrained_model_path: Optional[Path_fr] = None,
    enable_flash_attention: bool = True,
    recompute: bool = False,
    dtype: Literal["fp32", "fp16", "bf16"] = "fp32",
) -> LlamaModel:
    attn_implementation = "flash_attention" if enable_flash_attention else "eager"
    model = MODEL_SPEC[name](
        in_channels=in_channels,
        attn_implementation=attn_implementation,
        gradient_checkpointing=recompute,
        dtype=MODEL_DTYPE[dtype],
    )
    if pretrained_model_path:
        model = load_ckpt_params(model, pretrained_model_path.absolute)
    else:
        logger.info(f"Initialize {name} model randomly.")
    return model
