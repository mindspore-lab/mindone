import argparse
import os
import time

import numpy as np

import mindspore as ms

from mindone.diffusers import StableDiffusion3Pipeline


def sd3_performerce_test(
    pipe,
    use_cache_and_gate=False,
    use_todo=False,
    performance_threshold=6.0,
    prompt="A cat holding a sign that says hello world",
    generator=np.random.Generator(np.random.PCG64(0)),
):
    all_spend_time = 0
    steps = 11
    for i in range(steps):
        start = time.time()
        pipe(
            prompt=prompt,
            use_cache_and_gate=use_cache_and_gate,
            use_todo=use_todo,
            generator=generator,
        )
        end = time.time()
        if i > 0:
            all_spend_time += end - start

    avg_time = all_spend_time / (steps - 1)
    print(f">> average spend time: {avg_time:.3f}")

    if use_cache_and_gate or use_todo:
        assert avg_time < performance_threshold


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
    ms.set_context(mode=0, jit_config={"jit_level": "O2"})

    args = arg_parse()
    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.dirname(current_path))
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_path,
        custom_pipeline=os.path.join(parent_path, "stable_diffusion_3/pipeline_stable_diffusion_3.py"),
        mindspore_dtype=ms.float16,
    )

    # prepare test cases
    test_cases = [(pipe, False, False, 6.00), (pipe, True, False, 4.46), (pipe, True, True, 3.80)]

    # run tests
    for case in test_cases:
        sd3_performerce_test(*case)


if __name__ == "__main__":
    main()
