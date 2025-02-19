import argparse
import os
import yaml
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
import mindspore as ms

from models import VQVAE, build_vae_var
from mindone.utils.env import init_train_env
from mindone.utils.seed import set_random_seed
from mindone.utils.amp import auto_mixed_precision


def load_from_checkpoint(model, ckpt_fp):
    assert os.path.exists(ckpt_fp), f"checkopint {ckpt_fp} NOT found"
    print(f"Loading ckpt {ckpt_fp} into network")
    param_dict = ms.load_checkpoint(ckpt_fp)
    m, u = ms.load_param_into_net(model, param_dict)
    print("net param not load: ", m, len(m))
    print("ckpt param not load: ", u, len(u))



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
        V=4096, Cvae=32, ch=160, share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        patch_nums=patch_nums,
        num_classes=1000, depth=model_depth, shared_aln=False,
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
            dtype=dtype_map[args.var_dtype],
        )

    # sample


    cfg = 4  # @param {type:"slider", min:1, max:10, step:0.1}
    class_labels = (980, 980, 437, 437, 22, 22, 562, 562)  # @param {type:"raw"}
    more_smooth = False  # True for more smooth output

    B = len(class_labels)
    label_B = ms.Tensor(class_labels)
    reconB3HW = var.autoregressive_infer_cfg(B=B, label_B=label_B, cfg=cfg, top_k=900, top_p=0.95, g_seed=args.seed, more_smooth=more_smooth)

    print(reconB3HW)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
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

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)


