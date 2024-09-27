"""
Infer and evaluate VQVAE
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from videogvt.config.vqgan3d_ucf101_config import get_config
from videogvt.data.loader import create_dataloader
from videogvt.eval import calculate_psnr, calculate_ssim
from videogvt.models.vqvae import build_model
from videogvt.models.vqvae.lpips import LPIPS

from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def _rearrange_in(x):
    if x.ndim == 5:
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = ms.ops.reshape(x, (b * t, c, h, w))
    return x


def postprocess(x, trim=True):
    pixels = (x + 1) * 127.5
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.ndim == 5:
        # b, c, t, h, w -> b t c h w
        return np.transpose(pixels, (0, 2, 1, 3, 4))
    else:
        return pixels


def visualize(recons, x=None, save_fn="tmp_vae_recons"):
    # x: (b h w c), np array
    for i in range(recons.shape[0]):
        if x is not None:
            out = np.concatenate((x[i], recons[i]), axis=-2)
        else:
            out = recons[i]
        Image.fromarray(out).save(f"{save_fn}-{i:02d}.png")


def main(args):
    ms.set_context(mode=args.mode, ascend_config={"precision_mode": "allow_mix_precision_bf16"})
    set_logger(name="", output_dir=args.output_path, rank=0)

    config = get_config()
    dtype = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]

    model = build_model(args.model_class, config, is_training=False, dtype=dtype)
    param_dict = ms.load_checkpoint(args.ckpt_path)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    logger.info(f"Loaded checkpoint from  {args.ckpt_path}")

    if args.eval_loss:
        lpips_loss_fn = LPIPS()

    ds_config = dict(
        csv_path=args.csv_path,
        data_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        sample_stride=args.frame_stride,
        sample_n_frames=args.num_frames,
        return_image=False,
        flip=False,
        random_crop=False,
    )

    ds_name = "video" if args.model_class == "vqvae-3d" else "image"
    dataset = create_dataloader(
        ds_config=ds_config,
        batch_size=args.batch_size,
        ds_name=ds_name,
        num_parallel_workers=args.num_parallel_workers,
        shuffle=False,
        drop_remainder=False,
    )
    num_batches = dataset.get_dataset_size()

    ds_iter = dataset.create_dict_iterator(1)

    logger.info("Inferene begins")
    mean_infer_time = 0
    mean_psnr = 0
    mean_ssim = 0
    mean_lpips = 0
    mean_recon = 0

    psnr_list = []
    ssim_list = []
    for step, data in tqdm(enumerate(ds_iter)):
        x = data[ds_name].to(dtype)
        start_time = time.time()

        recons = model._forward(x)

        infer_time = time.time() - start_time
        mean_infer_time += infer_time
        logger.info(f"Infer time: {infer_time}")

        generated_videos = postprocess(recons.float().asnumpy())
        real_videos = postprocess(x.float().asnumpy())

        psnr_scores = list(calculate_psnr(real_videos, generated_videos)["value"].values())
        psnr_list += psnr_scores

        ssim_scores = list(calculate_ssim(real_videos, generated_videos)["value"].values())
        ssim_list += ssim_scores

        if args.eval_loss:
            recon_loss = np.abs((real_videos - generated_videos))
            lpips_loss = lpips_loss_fn(_rearrange_in(x), _rearrange_in(recons)).asnumpy()
            mean_recon += recon_loss.mean()
            mean_lpips += lpips_loss.mean()

    mean_psnr = np.mean(psnr_list)
    mean_ssim = np.mean(ssim_list)

    mean_infer_time /= num_batches
    logger.info(f"Mean infer time: {mean_infer_time}")
    logger.info(f"Done. Results saved in {args.output_path}")

    logger.info(f"mean psnr:{mean_psnr:.4f}")
    logger.info(f"mean ssim:{mean_ssim:.4f}")

    if args.eval_loss:
        mean_recon /= num_batches
        mean_lpips /= num_batches
        logger.info(f"mean recon loss: {mean_recon:.4f}")
        logger.info(f"mean lpips loss: {mean_lpips:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        default="outputs/vae_train/ckpt/vae_kl_f8-e10.ckpt",
        type=str,
        help="checkpoint path",
    )
    parser.add_argument(
        "--model_class",
        default="vqvae-3d",
        type=str,
        choices=[
            "vqvae-2d",
            "vqvae-3d",
        ],
        help="model arch type",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file. If None, will get images from the folder of `data_path`",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument(
        "--output_path",
        default="samples/vae_recons",
        type=str,
        help="output directory to save inference results",
    )
    parser.add_argument("--size", default=384, type=int, help="image rescale size")
    parser.add_argument("--crop_size", default=256, type=int, help="image crop size")
    parser.add_argument("--num_frames", default=16, type=int, help="num frames")
    parser.add_argument("--frame_stride", default=1, type=int, help="frame sampling stride")

    parser.add_argument(
        "--mode",
        default=0,
        type=int,
        help="Specify the mode: 0 for graph mode, 1 for pynative mode",
    )
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="mixed precision type, if fp32, all layer precision is float32 (amp_level=O0), \
                if bf16 or fp16, amp_level==O2, part of layers will compute in bf16 or fp16 such as matmul, dense, conv.",
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument(
        "--num_parallel_workers",
        default=8,
        type=int,
        help="num workers for data loading",
    )
    parser.add_argument(
        "--eval_loss",
        default=False,
        type=str2bool,
        help="whether measure loss including reconstruction, kl, perceptual loss",
    )
    parser.add_argument(
        "--save_images",
        default=True,
        type=str2bool,
        help="whether save reconstructed images",
    )
    parser.add_argument(
        "--encode_only",
        default=False,
        type=str2bool,
        help="only encode to save z or distribution",
    )
    parser.add_argument(
        "--save_z_dist",
        default=False,
        type=str2bool,
        help="If True, save z distribution, mean and logvar. Otherwise, save z after sampling.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
