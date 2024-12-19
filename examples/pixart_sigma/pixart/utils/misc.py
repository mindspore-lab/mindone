import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from mindone.utils.seed import set_random_seed

__all__ = ["str2bool", "check_cfgs_in_parser", "init_env", "organize_prompts"]


logger = logging.getLogger(__name__)


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


def organize_prompts(
    prompts: Optional[List[str]] = None,
    negative_prompts: Optional[List[str]] = None,
    prompt_path: Optional[str] = None,
    save_json: bool = True,
    output_dir: str = "./output",
    batch_size: int = 1,
) -> List[Dict[str, List[str]]]:
    if prompt_path is not None:
        if prompts is not None:
            logger.warning("`prompt_path` is given, read prompts from `prompt_path` instead.")

        prompts = list()
        with open(prompt_path, "r") as f:
            for line in f:
                prompts.append(line.strip())

    if isinstance(negative_prompts, list):
        if len(prompts) != len(negative_prompts):
            raise ValueError(
                "prompt's size must be equal to the negative prompt's size, "
                f"but get `{len(prompts)}` and `{len(negative_prompts)}` respectively."
            )

    contents = list()
    for i, prompt in enumerate(prompts):
        negative_prompt = negative_prompts[i] if negative_prompts else ""
        contents.append(dict(id=i, prompt=prompt, negative_prompt=negative_prompt))

    group_contents = list()
    group_prompts, group_nagative_prompts = list(), list()
    for i, record in enumerate(contents, start=1):
        group_prompts.append(record["prompt"])
        group_nagative_prompts.append(record["negative_prompt"])
        if i % batch_size == 0 or i == len(contents):
            group_contents.append(dict(prompt=group_prompts, negative_prompt=group_nagative_prompts))
            group_prompts, group_nagative_prompts = list(), list()

    if save_json:
        with open(os.path.join(output_dir, "prompts.json"), "w") as f:
            json.dump(contents, f, indent=4)
    return group_contents
