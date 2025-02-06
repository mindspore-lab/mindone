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
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.l

import argparse
import os
import warnings

from infer import GifRenderer, Image2Views, Removebg, Text2Image, Views2Mesh
from PIL import Image

import mindspore as ms

warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_lite", default=False, action="store_true")
    parser.add_argument("--mvd_ckt_path", default="weights/mvd_pretrain", type=str)
    parser.add_argument("--mv23d_cfg_path", default="./svrm/configs/svrm.yaml", type=str)
    parser.add_argument("--mv23d_ckt_path", default="weights/svrm/svrm.safetensors", type=str)
    parser.add_argument("--text2image_path", default="weights/hunyuanDiT", type=str)
    parser.add_argument("--save_folder", default="./outputs/test/", type=str)
    parser.add_argument(
        "--text_prompt",
        default="",
        type=str,
    )
    parser.add_argument("--image_prompt", default="", type=str)
    parser.add_argument("--device", default="Ascend", type=str)
    parser.add_argument("--mode", default=1, type=int, help="0 for GRAPH_MODE (not supported), 1 for PYNATIVE_MODE")
    parser.add_argument("--t2i_seed", default=0, type=int)
    parser.add_argument("--t2i_steps", default=25, type=int)
    parser.add_argument("--gen_seed", default=0, type=int)
    parser.add_argument("--gen_steps", default=50, type=int)
    parser.add_argument(
        "--max_faces_num", default=80000, type=int, help="max num of face, suggest 80000 for effect, 10000 for speed"
    )
    parser.add_argument("--do_texture_mapping", default=False, action="store_true")
    parser.add_argument("--do_render", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    assert not (args.text_prompt and args.image_prompt), "Text and image can only be given to one"
    assert args.text_prompt or args.image_prompt, "Text and image can only be given to one"

    if args.mode == 1:
        ms.set_context(mode=ms.PYNATIVE_MODE, device_target=args.device)  # pynative_synchronize=True
        print("Using PYNATIVE_MODE")
    else:  # do not fully support yet
        ms.set_context(mode=ms.GRAPH_MODE, device_target=args.device)
        print("Using GRAPH_MODE")

    # init model
    print("Initializing all models...")
    rembg_model = Removebg()
    image_to_views_model = Image2Views(use_lite=args.use_lite, ckpt_path=args.mvd_ckt_path)

    views_to_mesh_model = Views2Mesh(
        args.mv23d_cfg_path,
        args.mv23d_ckt_path,
        use_lite=args.use_lite,
    )

    if args.text_prompt:
        text_to_image_model = Text2Image(
            pretrain=args.text2image_path,
        )
    if args.do_render:
        gif_renderer = GifRenderer()

    print("Initialized all models.")
    # ---- ----- ---- ---- ---- ----

    os.makedirs(args.save_folder, exist_ok=True)

    # stage 1, text to image
    if args.text_prompt:
        print("START - Text to Image")
        res_rgb_pil = text_to_image_model(args.text_prompt, seed=args.t2i_seed, steps=args.t2i_steps)
        res_rgb_pil.save(os.path.join(args.save_folder, "img.jpg"))
        print("END - Text to Image")
    elif args.image_prompt:
        res_rgb_pil = Image.open(args.image_prompt)

    # stage 2, remove back ground
    print("START - Remove Image Background")
    res_rgba_pil = rembg_model(res_rgb_pil)
    res_rgba_pil.save(os.path.join(args.save_folder, "img_nobg.png"))
    print("END - Remove Image Background")

    # stage 3, image to views
    print("START - Image to Views")
    (views_grid_pil, cond_img), view_pil_list = image_to_views_model(
        res_rgba_pil, seed=args.gen_seed, steps=args.gen_steps
    )
    views_grid_pil.save(os.path.join(args.save_folder, "views.jpg"))
    print("END - Image to Views")

    # stage 4, views to mesh
    print("START - Views to Mesh")
    views_to_mesh_model(
        views_grid_pil,
        cond_img,
        seed=args.gen_seed,
        target_face_count=args.max_faces_num,
        save_folder=args.save_folder,
        do_texture_mapping=args.do_texture_mapping,
    )
    print("END - Views to Mesh")

    #  stage 5, render gif
    if args.do_render:
        print("START - GIF Rendering")
        gif_renderer(
            os.path.join(args.save_folder, "mesh.obj"),
            gif_dst_path=os.path.join(args.save_folder, "output.gif"),
        )
        print("END - GIF Rendering")
