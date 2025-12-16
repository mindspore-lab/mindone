import argparse
import os
import time

import numpy as np
from test_accuracy_data import PROMPTS

import mindspore as ms

from mindone.diffusers import FluxPipeline

ms.set_context(deterministic="ON")
ms.set_context(mode=0, jit_config={"jit_level": "O1"})


def generate_base_imgs(args, parent_path, data_root_path):
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        mindspore_dtype=ms.bfloat16,
    )

    data_root_path = os.path.join(data_root_path, "base")
    for hw in [1024, 2048]:
        save_path = os.path.join(data_root_path, f"{hw}")
        if not os.path.exists(f"{save_path}"):
            os.makedirs(f"{save_path}")
        all_spend_time = 0
        for i, prompt in enumerate(PROMPTS):
            generator = np.random.Generator(np.random.PCG64(0))
            start_time = time.time()
            img = pipe(
                prompt=prompt,
                generator=generator,
                height=hw,
                width=hw,
            )[
                0
            ][0]
            end_time = time.time()
            if i > 0:
                all_spend_time += end_time - start_time

            img.save(os.path.join(save_path, f"output_{i}.png"))
        avg_time = all_spend_time / (i)
        print(f"generate {hw} img spend time: {avg_time:.3f}s")


def generate_cache_imgs(args, parent_path, data_root_path):
    pipe = FluxPipeline.from_pretrained(
        args.model_path,
        mindspore_dtype=ms.bfloat16,
        custom_pipeline=os.path.join(parent_path, "pipeline_flux.py"),
    )

    # 定义hw和threshold的非正交组合，使用字典映射
    hw_threshold_map = {1024: [0.08, 0.12], 2048: [0.06, 0.08]}  # 对1024分辨率使用这两个阈值  # 对2048分辨率使用这两个阈值

    # 添加新变量enable_taylor，也可以根据需要设置不同的组合
    taylorseer_derivative_options = [0, 2]

    # 遍历每个hw及其对应的threshold列表
    for hw, thresholds in hw_threshold_map.items():
        for threshold in thresholds:
            for taylorseer_derivative in taylorseer_derivative_options:
                # 构建保存路径，包含所有参数信息
                save_path = os.path.join(data_root_path, f"fbcache_{threshold}_{taylorseer_derivative}", f"{hw}")
                if not os.path.exists(f"{save_path}"):
                    os.makedirs(f"{save_path}")

                all_spend_time = 0
                for i, prompt in enumerate(PROMPTS):
                    generator = np.random.Generator(np.random.PCG64(0))
                    start_time = time.time()
                    img = pipe(
                        prompt=prompt,
                        generator=generator,
                        height=hw,
                        width=hw,
                        residual_diff_threshold=threshold,
                        taylorseer_derivative=taylorseer_derivative,
                    )[0][0]
                    end_time = time.time()
                    if i > 0:
                        all_spend_time += end_time - start_time

                    img.save(os.path.join(save_path, f"output_{i}.png"))

                avg_time = all_spend_time / i
                print(
                    f"generate {hw} img with threshold={threshold}, taylorseer_derivative={taylorseer_derivative} spend time: {avg_time:.3f}s"
                )


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="ckpt/FLUX.1-dev",
        help="FLUX.1-dev checkpoint",
    )
    args = parser.parse_args()
    return args


def main():
    if not os.path.exists("./flux_output"):
        os.makedirs("./flux_output")
    data_root_path = os.path.abspath("./flux_output")

    args = arg_parse()

    current_path = os.path.abspath(os.path.dirname(__file__))
    parent_path = os.path.abspath(os.path.dirname(current_path))

    generate_base_imgs(args, parent_path, data_root_path)
    generate_cache_imgs(args, parent_path, data_root_path)


if __name__ == "__main__":
    main()
