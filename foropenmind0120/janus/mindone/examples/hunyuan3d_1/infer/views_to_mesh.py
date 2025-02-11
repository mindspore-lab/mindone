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
from PIL import Image, ImageSequence
from svrm.predictor import MV23DPredictor

import mindspore as ms
from mindspore import ops


class Views2Mesh:
    def __init__(self, mv23d_cfg_path, mv23d_ckt_path, use_lite=False):
        """
        mv23d_cfg_path: config yaml file
        mv23d_ckt_path: path to ckpt
        use_lite: lite version
        """
        self.mv23d_predictor = MV23DPredictor(mv23d_ckt_path, mv23d_cfg_path)
        self.mv23d_predictor.model.set_train(False)
        self.order = [0, 1, 2, 3, 4, 5] if use_lite else [0, 2, 4, 5, 3, 1]

        set_parameter_grad_false(self.mv23d_predictor.model)
        print("view2mesh model", get_parameter_number(self.mv23d_predictor.model))

    @timing_decorator("views to mesh")
    def __call__(self, *args, **kwargs):
        res = self.call(*args, **kwargs)
        return res

    def call(
        self,
        views_pil=None,
        cond_pil=None,
        gif_pil=None,
        seed=0,
        target_face_count=10000,
        do_texture_mapping=True,
        save_folder="./outputs/test",
    ):
        """
        can set views_pil, cond_pil simutaously or set gif_pil only
        seed: int
        target_face_count: int
        save_folder: path to save mesh files
        """
        save_dir = save_folder
        os.makedirs(save_dir, exist_ok=True)

        if views_pil is not None and cond_pil is not None:
            # show_image = rearrange(np.asarray(views_pil, dtype=np.uint8), '(n h) (m w) c -> (n m) h w c', n=3, m=2)
            show_image = np.asarray(views_pil, dtype=np.uint8)
            # '(n h) (m w) c -> (n m) h w c', n=3, m=2
            nh, mw, c = show_image.shape  # default 3x2 images
            assert (nh % 3 == 0) and (mw % 2 == 0)
            n = 3
            m = 2
            h = nh // n
            w = mw // m
            show_image = show_image.reshape(n, h, m, w, c).transpose(0, 2, 1, 3, 4).reshape(-1, h, w, c)

            views = [Image.fromarray(show_image[idx]) for idx in self.order]
            image_list = [cond_pil] + views
            image_list = [img.convert("RGB") for img in image_list]
        elif gif_pil is not None:
            image_list = [img.convert("RGB") for img in ImageSequence.Iterator(gif_pil)]

        image_input = image_list[0]
        image_list = image_list[1:] + image_list[:1]

        seed_everything(seed)
        ops.stop_gradient(
            self.mv23d_predictor.predict(
                image_list,
                save_dir=save_dir,
                image_input=image_input,
                target_face_count=target_face_count,
                do_texture_mapping=do_texture_mapping,
            )
        )
        return save_dir


if __name__ == "__main__":
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument("--views_path", type=str, required=True)
        parser.add_argument("--cond_path", type=str, required=True)
        parser.add_argument("--save_folder", default="./outputs/test/", type=str)
        parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
        parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
        parser.add_argument(
            "--max_faces_num",
            default=90000,
            type=int,
            help="max num of face, suggest 90000 for effect, 10000 for speed",
        )
        parser.add_argument("--device", default="Ascend", type=str)
        parser.add_argument("--mode", default=1, type=int, help="0 for GRAPH_MODE, 1 for PYNATIVE_MODE")
        parser.add_argument("--use_lite", default="false", type=str)
        parser.add_argument("--do_texture_mapping", default="false", type=str)

        return parser.parse_args()

    args = get_args()
    if args.mode == 1:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)
        print("Using PYNATIVE_MODE")
    else:  # NOTE: it may be slower
        ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device)
        print("Using GRAPH_MODE")

    args.use_lite = str_to_bool(args.use_lite)
    args.do_texture_mapping = str_to_bool(args.do_texture_mapping)

    views = Image.open(args.views_path)
    cond = Image.open(args.cond_path)

    views_to_mesh_model = Views2Mesh(args.mv23d_cfg_path, args.mv23d_ckt_path, use_lite=args.use_lite)

    views_to_mesh_model(
        views,
        cond,
        0,
        target_face_count=args.max_faces_num,
        save_folder=args.save_folder,
        do_texture_mapping=args.do_texture_mapping,
    )
