import argparse
import os
import sys
from time import time

import numpy as np
from models import build_vae_var
from PIL import Image
from utils.utils import load_from_checkpoint, make_grid

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_train_env
from mindone.utils.seed import set_random_seed


def main(args):
    # init
    init_train_env(
        args.ms_mode,
        seed=args.seed,
        jit_level=args.jit_level,
    )
    set_random_seed(args.seed)

    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    model_depth = 16
    assert model_depth in {16, 20, 24, 30}
    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=patch_nums,
        num_classes=1000,
        depth=model_depth,
        shared_aln=False,
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

    cfg = 4  # @param {type:"slider", min:1, max:10, step:0.1}
    more_smooth = False  # True for more smooth output
    class_labels = tuple(map(int, args.class_labels.replace("-", "_").split("_")))

    B = len(class_labels)
    label_B = ms.Tensor(class_labels)
    start = time()
    recon_B3HW = var.autoregressive_infer_cfg(
        B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=args.seed, more_smooth=more_smooth
    )

    img = make_grid(recon_B3HW, nrow=B, padding=0, pad_value=1.0)
    img = img.permute(1, 2, 0).mul(255).asnumpy()
    img = Image.fromarray(img.astype(np.uint8))
    img.save(args.output_path)
    print(f"inference time is {time()-start}s")


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
    parser.add_argument("--class_labels", default="980_980_437_437_22_22_562_562", type=str, help="class labels")
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

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
