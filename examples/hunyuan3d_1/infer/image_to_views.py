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
from infer.utils import get_parameter_number, seed_everything, set_parameter_grad_false, str_to_bool, timing_decorator
from mvd.hunyuan3d_mvd_lite_pipeline import Hunyuan3d_MVD_Lite_Pipeline
from mvd.hunyuan3d_mvd_std_pipeline import HunYuan3D_MVD_Std_Pipeline
from PIL import Image

import mindspore as ms
from mindspore import ops


def save_gif(pils, save_path, df=False):
    # save a list of PIL.Image to gif
    spf = 4000 / len(pils)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pils[0].save(save_path, format="GIF", save_all=True, append_images=pils[1:], duration=spf, loop=0)
    return save_path


class Image2Views:
    def __init__(
        self,
        use_lite=False,
        ckpt_path="./weights/mvd_std"
        # std_pretrain='./weights/mvd_std', lite_pretrain='./weights/mvd_lite'
    ):
        if use_lite:
            print("Loading", ckpt_path)
            self.pipe = Hunyuan3d_MVD_Lite_Pipeline.from_pretrained(
                ckpt_path,
                mindspore_dtype=ms.float16,
                use_safetensors=True,
            )
        else:
            print("Loading", ckpt_path)
            self.pipe = HunYuan3D_MVD_Std_Pipeline.from_pretrained(
                ckpt_path,
                mindspore_dtype=ms.float16,
                use_safetensors=True,
            )
        self.order = [0, 1, 2, 3, 4, 5] if use_lite else [0, 2, 4, 5, 3, 1]
        set_parameter_grad_false(self.pipe.unet)
        print("image2views unet model", get_parameter_number(self.pipe.unet))

    @timing_decorator("image to views")
    def __call__(self, *args, **kwargs):
        res = self.call(*args, **kwargs)

        return res

    def call(self, pil_img, seed=0, steps=50, guidance_scale=2.0):
        seed_everything(seed)
        generator = np.random.Generator(np.random.PCG64(seed=int(seed)))
        res_img = ops.stop_gradient(
            self.pipe(pil_img, num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator).images
        )
        #  '(n h) (m w) c -> (n m) h w c', n=3, m=2)
        show_image = np.asarray(res_img[0], dtype=np.uint8)
        # '(n h) (m w) c -> (n m) h w c', n=3, m=2
        nh, mw, c = show_image.shape
        assert (nh % 3 == 0) and (mw % 2 == 0)
        n = 3
        m = 2
        h = nh // n
        w = mw // m
        show_image = show_image.reshape(n, h, m, w, c).transpose(0, 2, 1, 3, 4).reshape(-1, h, w, c)

        pils = [res_img[1]] + [Image.fromarray(show_image[idx]) for idx in self.order]

        return res_img, pils


if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--rgba_path", type=str, required=True)
        parser.add_argument("--output_views_path", type=str, required=True)
        parser.add_argument("--output_cond_path", type=str, required=True)
        parser.add_argument("--mvd_ckt_path", type=str, required=True)
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--steps", default=50, type=int)
        parser.add_argument("--device", default="Ascend", type=str)
        parser.add_argument("--use_lite", default="false", type=str)
        return parser.parse_args()

    args = get_args()
    ms.set_context(device_target=args.device, mode=1)

    args.use_lite = str_to_bool(args.use_lite)

    rgba_pil = Image.open(args.rgba_path)

    assert rgba_pil.mode == "RGBA", "rgba_pil must be RGBA mode"

    model = Image2Views(use_lite=args.use_lite, ckpt_path=args.mvd_ckt_path)

    (views_pil, cond), _ = model(rgba_pil, seed=args.seed, steps=args.steps)

    views_pil.save(args.output_views_path)
    cond.save(args.output_cond_path)
