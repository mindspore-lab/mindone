import logging
from typing import Dict, Literal, Optional, Tuple, Union

import numpy as np
from jsonargparse.typing import Path_fr
from mg.models import LlamaModel, llama3_1B, llama3_5B, llama3_30B

import mindspore as ms
from mindspore import _no_grad, jit_class, nn

from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.params import load_param_into_net_with_filter

__all__ = ["MODEL_DTYPE", "no_grad", "init_model", "resume_train_net"]

logger = logging.getLogger(__name__)

MODEL_SPEC = {"llama-1B": llama3_1B, "llama-5B": llama3_5B, "llama-30B": llama3_30B}

MODEL_DTYPE = {
    "fp32": ms.float32,
    "fp16": ms.float16,
    "bf16": ms.bfloat16,
}


def load_ckpt_params(model: nn.Cell, ckpt: Union[str, Dict]) -> None:
    if isinstance(ckpt, str):
        logger.info(f"Loading {ckpt} params into network...")
        param_dict = ms.load_checkpoint(ckpt)
        param_dict = {k.replace("network.model.", ""): v for k, v in param_dict.items()}
    else:
        param_dict = ckpt

    # 3.2.2 PE expansion
    for pe in ["pos_embedding_table_t", "pos_embedding_table_h", "pos_embedding_table_w"]:
        if (model_shape := model.__getattr__(pe).__getattr__("embedding_table").shape[0]) != (
            weight_shape := param_dict[pe + ".embedding_table"].shape[0]
        ):
            logger.info(
                f"PE({pe[-1].upper()}): the model shape ({model_shape}) doesn't match the weight shape ({weight_shape})."
                " Expanding linearly."
            )
            # do linear interpolation in FP32 as BF16 is not supported by numpy
            weight = param_dict[pe + ".embedding_table"].to(ms.float32).numpy()
            interp_weight = np.apply_along_axis(
                lambda y: np.interp(
                    np.linspace(0, weight_shape - 1, model_shape), np.linspace(0, weight_shape - 1, weight_shape), y
                ),
                axis=0,
                arr=weight,
            ).astype(weight.dtype)
            param_dict[pe + ".embedding_table"] = ms.Parameter(
                ms.tensor(interp_weight, dtype=model.dtype), name=param_dict[pe + ".embedding_table"].name
            )

    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
    if param_not_load or ckpt_not_load:
        logger.warning(
            f"Exist ckpt params not loaded: {ckpt_not_load} (total: {len(ckpt_not_load)}),\n"
            f"or net params not loaded: {param_not_load} (total: {len(param_not_load)})"
        )


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
    in_channels: int = 16,
    pretrained_model_path: Optional[Path_fr] = None,
    resume: bool = False,
    enable_flash_attention: bool = True,
    recompute_every_nth_block: Optional[int] = None,
    not_recompute_fa: bool = False,
    max_length: Tuple[int, int, int] = (32, 73, 73),
    dtype: Literal["fp32", "fp16", "bf16"] = "fp32",
) -> LlamaModel:
    attn_implementation = "flash_attention" if enable_flash_attention else "eager"
    with nn.no_init_parameters():
        model = MODEL_SPEC[name](
            in_channels=in_channels,
            attn_implementation=attn_implementation,
            recompute_every_nth_block=recompute_every_nth_block,
            not_recompute_fa=not_recompute_fa,
            max_length=max_length,
            dtype=MODEL_DTYPE[dtype],
        )

    if resume:
        logger.info("Resume training checkpoint provided, skipping weight loading.")
    elif pretrained_model_path:
        load_ckpt_params(model, pretrained_model_path.absolute)
    else:
        logger.info(f"Initialize {name} model randomly.")
    # initialize uninitialized parameters, if any
    model.init_parameters_data()
    return model


def resume_train_net(
    train_net: TrainOneStepWrapper, resume_ckpt: Optional[Path_fr] = None
) -> Tuple[Union[int, None], Union[int, None]]:
    if resume_ckpt is None:
        return None, None

    state_dict = ms.load_checkpoint(resume_ckpt)
    if "epoch_num" not in state_dict or "cur_step" not in state_dict or "loss_scale" not in state_dict:
        raise ValueError("Resume training checkpoint is invalid. Please check the checkpoint file.")

    start_epoch = state_dict.pop("epoch_num").item()
    global_step = state_dict.pop("cur_step").item()
    logger.info(f"Resuming training of network from {resume_ckpt} at global step {global_step}")

    # FIXME: `EvalSaveCallback` renames `scale_sense` to `loss_scale` when saving the resume checkpoint
    train_net.scale_sense = ms.Parameter(state_dict.pop("loss_scale"), name="scale_sense")
    param_not_load, ckpt_not_load = load_param_into_net_with_filter(train_net, state_dict, filter=state_dict.keys())
    if param_not_load or ckpt_not_load:
        logger.warning(
            f"Exist ckpt params not loaded: {ckpt_not_load} (total: {len(ckpt_not_load)}),\n"
            f"or net params not loaded: {param_not_load} (total: {len(param_not_load)})"
        )

    return start_epoch, global_step
