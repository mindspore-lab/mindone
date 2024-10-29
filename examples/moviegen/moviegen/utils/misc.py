import argparse
import logging
from typing import Tuple

from moviegen.models import llama3_1B, llama3_5B, llama3_30B

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from mindone.utils.seed import set_random_seed

__all__ = ["MODEL_SPEC", "MODEL_DTYPE", "str2bool", "check_cfgs_in_parser", "init_env"]


logger = logging.getLogger(__name__)


MODEL_SPEC = {"llama-1B": llama3_1B, "llama-5B": llama3_5B, "llama-30B": llama3_30B}

MODEL_DTYPE = {
    "fp32": ms.float32,
    "fp16": ms.float16,
    "bf16": ms.bfloat16,
}


def str2bool(b: str) -> bool:
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser) -> None:
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def init_env(args) -> Tuple[int, int]:
    set_random_seed(args.seed)
    ms.set_context(mode=args.mode, device_target=args.device_target, jit_config=dict(jit_level=args.jit_level))
    if args.use_parallel:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True, device_num=device_num
        )
    else:
        device_num, rank_id = 1, 0

    return device_num, rank_id
