# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
import os
import sys

sys.path.insert(0, f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}")

import numpy as np
from infer.utils import get_parameter_number, seed_everything, set_parameter_grad_false, timing_decorator

import mindspore as ms
from mindspore import ops

from mindone.diffusers import HunyuanDiTPipeline


class Text2Image:
    def __init__(self, pretrain="Tencent-Hunyuan/HunyuanDiT-Diffusers"):
        self.pipe = HunyuanDiTPipeline.from_pretrained(
            pretrain, mindspore_dtype=ms.float32
        )  # NOTE: CumProd does not support float16
        set_parameter_grad_false(self.pipe.transformer)
        print("text2image transformer model", get_parameter_number(self.pipe.transformer))

        self.neg_txt = (
            "文本,特写,裁剪,出框,最差质量,低质量,JPEG伪影,PGLY,重复,病态,残缺,多余的手指,变异的手,"
            "画得不好的手,画得不好的脸,变异,畸形,模糊,脱水,糟糕的解剖学,糟糕的比例,多余的肢体,克隆的脸,"
            "毁容,恶心的比例,畸形的肢体,缺失的手臂,缺失的腿,额外的手臂,额外的腿,融合的手指,手指太多,长脖子"
        )

    @timing_decorator("text to image")
    def __call__(self, *args, **kwargs):
        res = self.call(*args, **kwargs)
        return res

    def call(self, prompt, seed=0, steps=25):
        """
        args:
            prompr: str
            seed: int
            steps: int
        return:
            rgb: PIL.Image
        """
        print("prompt is:", prompt)
        prompt = prompt + ",白色背景,3D风格,最佳质量"
        if seed is not None:
            seed_everything(seed)
            generator = np.random.Generator(np.random.PCG64(seed=seed))
        else:
            generator = np.random.Generator(np.random.PCG64(0))

        rgb = ops.stop_gradient(
            self.pipe(
                prompt=prompt,
                negative_prompt=self.neg_txt,
                num_inference_steps=steps,
                width=1024,
                height=1024,
                generator=generator,
                return_dict=False,
            )
        )[0][0]

        return rgb


if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--text2image_path", default="Tencent-Hunyuan/HunyuanDiT-Diffusers", type=str)
        parser.add_argument("--text_prompt", default="", type=str)
        parser.add_argument("--output_img_path", default="./outputs/test/img.jpg", type=str)
        parser.add_argument("--device", default="Ascend", type=str)
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--steps", default=25, type=int)
        return parser.parse_args()

    args = get_args()
    ms.set_context(device_target=args.device, mode=1)

    text2image_model = Text2Image()
    rgb_img = text2image_model(args.text_prompt, seed=args.seed, steps=args.steps)
    rgb_img.save(args.output_img_path)
