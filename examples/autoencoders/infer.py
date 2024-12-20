"""
Infer and evaluate autoencoders
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
from ae.data.image_dataset import create_dataloader
from ae.models.lpips import LPIPS
from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.config import instantiate_from_config, str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def postprocess(x):
    pixels = (x + 1) * 127.5
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    # b, c, h, w -> b h w c
    return np.transpose(pixels, (0, 2, 3, 1))


def visualize(recons, x=None, save_fn="tmp_vae_recons"):
    # x: (b h w c), np array
    for i in range(recons.shape[0]):
        if x is not None:
            out = np.concatenate((x[i], recons[i]), axis=-2)
        else:
            out = recons[i]
        Image.fromarray(out).save(f"{save_fn}-{i:02d}.png")


def main(args):
    ms.set_context(mode=args.mode)
    set_logger(name="", output_dir=args.output_path, rank=0)

    config = OmegaConf.load(args.model_config)
    model = instantiate_from_config(config.generator)
    model.init_from_ckpt(args.ckpt_path)
    logger.info(f"Loaded checkpoint from  {args.ckpt_path}")

    if args.eval_loss:
        lpips_loss_fn = LPIPS()

    model.set_train(False)

    ds_config = dict(
        csv_path=args.csv_path,
        image_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        flip=False,
        random_crop=False,
    )

    dataset = create_dataloader(
        ds_config=ds_config,
        batch_size=args.batch_size,
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
    num_samples = 0
    for step, data in tqdm(enumerate(ds_iter)):
        x = data["image"]
        start_time = time.time()
        if config.model_version.lower() == "kl":
            z = model.encode(x)
        elif config.model_version.lower() == "vq":
            z, _, _ = model.encode(x)
        else:
            raise ValueError(f"unknown model version: {config.model_version}")

        if not args.encode_only:
            recons = model.decode(z)

        infer_time = time.time() - start_time
        mean_infer_time += infer_time
        logger.info(f"Infer time: {infer_time}")

        if not args.encode_only:
            recons_rgb = postprocess(recons.asnumpy())
            x_rgb = postprocess(x.asnumpy())

            psnr_cur = [calc_psnr(x_rgb[i], recons_rgb[i]) for i in range(x_rgb.shape[0])]
            ssim_cur = [
                calc_ssim(x_rgb[i], recons_rgb[i], data_range=255, channel_axis=-1, multichannel=True)
                for i in range(x_rgb.shape[0])
            ]
            mean_psnr += sum(psnr_cur)
            mean_ssim += sum(ssim_cur)
            num_samples += x_rgb.shape[0]

            if args.save_images:
                save_fn = os.path.join(
                    args.output_path, "{}-{}".format(os.path.basename(args.data_path), f"step{step:03d}")
                )
                visualize(recons_rgb, x_rgb, save_fn=save_fn)

            if args.eval_loss:
                recon_loss = np.abs((x - recons).asnumpy())
                lpips_loss = lpips_loss_fn(x, recons).asnumpy()
                mean_recon += recon_loss.mean()
                mean_lpips += lpips_loss.mean()

    mean_infer_time /= num_batches
    logger.info(f"Mean infer time: {mean_infer_time}")
    logger.info(f"Done. Results saved in {args.output_path}")

    if not args.encode_only:
        mean_psnr /= num_samples
        mean_ssim /= num_samples
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
        "--model_config",
        default="configs/autoencoder_kl_f8.yaml",
        type=str,
        help="model architecture config",
    )
    parser.add_argument(
        "--ckpt_path", default="outputs/vae_train/ckpt/vae_kl-e10.ckpt", type=str, help="checkpoint path"
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file. If None, will get images from the folder of `data_path`",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument(
        "--output_path", default="samples/vae_recons", type=str, help="output directory to save inference results"
    )
    parser.add_argument("--size", default=384, type=int, help="image rescale size")
    parser.add_argument("--crop_size", default=256, type=int, help="image crop size")

    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--num_parallel_workers", default=8, type=int, help="num workers for data loading")
    parser.add_argument(
        "--eval_loss",
        default=False,
        type=str2bool,
        help="whether measure loss including reconstruction, kl, perceptual loss",
    )
    parser.add_argument("--save_images", default=True, type=str2bool, help="whether save reconstructed images")
    parser.add_argument("--encode_only", default=False, type=str2bool, help="only encode to save z or distribution")
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
