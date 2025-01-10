import argparse
import os
import shutil

import numpy as np
from test_accuracy_data import PROMPTS

import mindspore as ms

from mindone.diffusers import StableDiffusion3Pipeline

ms.set_context(deterministic="ON")
ms.set_context(mode=0, jit_config={"jit_level": "O2"})


def generate_imgs(pipe, save_path, use_cache_and_gate=True, use_todo=False):
    if use_cache_and_gate and not use_todo:
        data_dir = "cache_gate"
    elif use_todo and not use_cache_and_gate:
        data_dir = "todo"
    elif use_cache_and_gate and use_todo:
        data_dir = "cache_gate_todo"
    else:
        data_dir = "base"

    save_path = os.path.join(save_path, data_dir)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path)
    for i, prompt in enumerate(PROMPTS):
        generator = np.random.Generator(np.random.PCG64(0))
        img = pipe(
            prompt=prompt,
            use_cache_and_gate=use_cache_and_gate,
            use_todo=use_todo,
            generator=generator,
        )[
            0
        ][0]

        img.save(os.path.join(save_path, f"output_{i}.png"))


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Stable Diffusion 3 checkpoint",
    )
    args = parser.parse_args()
    return args


def main():
    if os.path.exists("./data"):
        shutil.rmtree("./data")
    os.makedirs("./data")
    data_root_path = os.path.abspath("./data")

    args = arg_parse()

    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.dirname(current_path))
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        custom_pipeline=os.path.join(parent_path, "stable_diffusion_3/pipeline_stable_diffusion_3.py"),
        mindspore_dtype=ms.float16,
    )
    generate_imgs(pipe, data_root_path, use_cache_and_gate=False, use_todo=False)
    generate_imgs(pipe, data_root_path, use_cache_and_gate=True, use_todo=False)
    generate_imgs(pipe, data_root_path, use_cache_and_gate=True, use_todo=True)


if __name__ == "__main__":
    main()
