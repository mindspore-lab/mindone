import argparse
from time import time
from typing import List

import numpy as np
from models import build_vae_var
from PIL import Image
from utils.data import normalize_01_into_pm1, pil_loader
from utils.utils import load_from_checkpoint, make_grid

import mindspore as ms
from mindspore import mint

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_env
from mindone.utils.seed import set_random_seed


def get_edit_mask(
    patch_nums: List[int], y0: float, x0: float, y1: float, x1: float, inpainting: bool = True
) -> ms.Tensor:
    ph, pw = patch_nums[-1], patch_nums[-1]
    edit_mask = mint.zeros((ph, pw))
    edit_mask[
        round(y0 * ph) : round(y1 * ph), round(x0 * pw) : round(x1 * pw)
    ] = 1  # outpainting mode: center would be gt
    if inpainting:
        edit_mask = 1 - edit_mask  # inpainting mode: center would be model pred
    return edit_mask  # a binary mask, 1 for keeping the tokens of the image to be edited; 0 for generating new tokens (by VAR)


def main(args):
    # init
    init_env(
        args.ms_mode,
        seed=args.seed,
        jit_level=args.jit_level,
    )
    set_random_seed(args.seed)

    model_depth = 16
    assert model_depth in {16, 20, 24, 30, 36}

    FOR_512_px = model_depth == 36
    if FOR_512_px:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    else:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)

    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=patch_nums,
        num_classes=1000,
        depth=model_depth,
        shared_aln=FOR_512_px,
    )

    if args.vae_checkpoint:
        load_from_checkpoint(vae, args.vae_checkpoint)
    else:
        print("warning!! VAE uses random initialization!")

    if args.var_checkpoint:
        load_from_checkpoint(var, args.var_checkpoint)
    else:
        print("warning!! VAR uses random initialization!")

    vae.set_train(False)
    var.set_train(False)

    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.dtype in ["fp16", "bf16"]:
        var = auto_mixed_precision(
            var,
            amp_level="O2",
            dtype=dtype_map[args.dtype],
        )
    print("prepare finished")

    # sample
    # set args
    img_to_be_edited = args.input_image_path

    # edit to class label, if it was set to 0-999, it's an class condition editing task
    # if it was set to 1000, it's an in-painting task
    class_label = 1000  # @param {type:"raw"}

    # load the image to be edited

    input_img = normalize_01_into_pm1(ms.Tensor(ms.dataset.vision.ToTensor()(pil_loader(img_to_be_edited)))).unsqueeze(
        0
    )
    input_img_tokens = vae.img_to_idxBl(input_img, var.patch_nums)

    # zero-shot edit
    # The inpainting parameter controls whether the task is inpainting or outpainting
    edit_mask = get_edit_mask(
        var.patch_nums,
        y0=0.1,
        x0=0.1,
        y1=0.8,
        x1=0.8,
        inpainting=True,
    )
    B = 1
    label_B = ms.tensor([class_label])

    start = time()
    recon_B3HW = var.autoregressive_infer_cfg(
        B=B,
        label_B=label_B,
        cfg=3,
        top_k=900,
        top_p=0.95,
        g_seed=0,
        more_smooth=True,
        input_img_tokens=input_img_tokens,
        edit_mask=edit_mask,
    )

    img = make_grid(recon_B3HW, nrow=8, padding=0, pad_value=1.0)
    img = img.permute(1, 2, 0).mul(255).asnumpy()
    img = Image.fromarray(img.astype(np.uint8))
    img.save(args.output_path)
    print(f"inference time is {time() - start}s")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)"
    )
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="model/vae-2972c247.ckpt",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    parser.add_argument(
        "--var_checkpoint",
        type=str,
        default="model/var-d16.ckpt",
        help="VAR checkpoint file path which is used to load var weight.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    )
    parser.add_argument("--output_path", default="image.png", type=str, help="output path to save inference results")
    parser.add_argument("--input_image_path", default=None, type=str, help="input image path for edit.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
