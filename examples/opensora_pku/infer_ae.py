"""
Infer and evaluate autoencoders
"""
import argparse
import logging
import os
import sys
import time

import imageio
import numpy as np
from omegaconf import OmegaConf
from opensora.data.loader import create_dataloader
from opensora.models.ae.lpips import LPIPS
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config, str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def postprocess(x, trim=True):
    # postprocess for computing metrics
    pixels = (x + 1) * 127.5
    pixels = np.clip(pixels, 0, 255).astype(np.uint8)

    if len(pixels.shape) == 4:
        # b, c, h, w -> b h w c
        return np.transpose(pixels, (0, 2, 3, 1))
    else:
        # b c t h w -> b t h w c -> b*t h w c
        b, c, t, h, w = pixels.shape
        pixels = np.transpose(pixels, (0, 2, 3, 4, 1))
        pixels = np.reshape(pixels, (b * t, h, w, c))
        return pixels


def visualize_image(recons, x=None, save_fn="tmp_vae_recons"):
    # x: (b h w c)
    for i in range(recons.shape[0]):
        if x is not None:
            out = np.concatenate((x[i], recons[i]), axis=-2)
        else:
            out = recons[i]
        Image.fromarray(out).save(f"{save_fn}-{i:02d}.png")


def visualize_video(recons, x=None, save_fn="tmp_vae3d_recons", fps=15):
    # x: (b t h w c)
    for i in range(recons.shape[0]):
        if x is not None:
            out = np.concatenate((x[i], recons[i]), axis=-2)
        else:
            out = recons[i]
        save_fp = f"{save_fn}-{i:02d}.gif"
        imageio.mimsave(save_fp, out, duration=1 / fps, loop=0)


def main(args):
    ascend_config = {"precision_mode": "must_keep_origin_dtype"}
    ms.set_context(mode=args.mode, ascend_config=ascend_config)
    set_logger(name="", output_dir=args.output_path, rank=0)

    config = OmegaConf.load(args.model_config)
    config.generator.params.resnet_micro_batch_size = args.vae_micro_batch_size
    if args.vae_micro_batch_size:
        logger.info(f"Using VAE micro batch size {args.vae_micro_batch_size}")
    model = instantiate_from_config(config.generator)
    model.init_from_ckpt(args.ckpt_path)
    model.set_train(False)
    logger.info(f"Loaded checkpoint from  {args.ckpt_path}")

    if args.eval_loss:
        lpips_loss_fn = LPIPS()

    if args.dtype != "fp32":
        amp_level = "O2"
        dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        model = auto_mixed_precision(model, amp_level, dtype)
        logger.info(f"Set mixed precision to O2 with dtype={args.dtype}")
    else:
        amp_level = "O0"
    if args.enable_tiling:
        model.enable_tiling()
        model.tile_overlap_factor = args.tile_overlap_factor
    ds_config = dict(
        csv_path=args.csv_path,
        data_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        flip=False,
        random_crop=False,
    )
    if args.dataset_name == "video":
        ds_config.update(
            dict(
                sample_stride=args.frame_stride,
                sample_n_frames=args.num_frames,
                return_image=False,
            )
        )
        assert not (
            args.num_frames % 2 == 0 and config.generator.params.ddconfig.split_time_upsample
        ), "num of frames must be odd if split_time_upsample is True"
    else:
        ds_config.update(dict(expand_dim_t=args.expand_dim_t))

    dataset = create_dataloader(
        ds_config=ds_config,
        batch_size=args.batch_size,
        ds_name=args.dataset_name,
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
        if args.dataset_name == "image":
            x = data["image"]
        else:
            x = data["video"]
        start_time = time.time()

        z = model.encode(x)
        if not args.encode_only:
            recons = model.decode(z)

        infer_time = time.time() - start_time
        mean_infer_time += infer_time
        logger.info(f"Infer time: {infer_time}")

        if not args.encode_only:
            # if args.dataset_name == 'image' and args.expand_dim_t:
            #    # b c t h w -> b c h w
            #    x = x[:,:,0,:,:]
            #    recons= recons[:,:,0,:,:]
            is_video = len(recons.shape) == 5 and (recons.shape[-3] > 1)
            t = recons.shape[-3] if is_video else 1

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

            if args.eval_loss:
                recon_loss = np.abs((x - recons).asnumpy())
                lpips_loss = lpips_loss_fn(x, recons).asnumpy()
                mean_recon += recon_loss.mean()
                mean_lpips += lpips_loss.mean()

            if args.save_vis:
                save_fn = os.path.join(
                    args.output_path, "{}-{}".format(os.path.basename(args.data_path), f"step{step:03d}")
                )
                if not is_video:
                    visualize_image(recons_rgb, x_rgb, save_fn=save_fn)
                else:
                    bt, h, w, c = recons_rgb.shape
                    recons_rgb_vis = np.reshape(recons_rgb, (bt // t, t, h, w, c))
                    x_rgb_vis = np.reshape(x_rgb, (bt // t, t, h, w, c))
                    visualize_video(recons_rgb_vis, x_rgb_vis, save_fn=save_fn)

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
        "--ckpt_path", default="outputs/vae_train/ckpt/vae_kl_f8-e10.ckpt", type=str, help="checkpoint path"
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file. If None, will get images from the folder of `data_path`",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument(
        "--dataset_name", default="image", type=str, choices=["image", "video"], help="dataset name, image or video"
    )
    parser.add_argument(
        "--output_path", default="samples/vae_recons", type=str, help="output directory to save inference results"
    )
    parser.add_argument("--num_frames", default=17, type=int, help="num frames")
    parser.add_argument("--frame_stride", default=1, type=int, help="frame sampling stride")
    parser.add_argument(
        "--expand_dim_t",
        default=False,
        type=str2bool,
        help="expand temporal axis for image data, used for vae 3d inference with image data",
    )
    parser.add_argument("--size", default=256, type=int, help="image rescale size")
    parser.add_argument("--crop_size", default=256, type=int, help="image crop size")

    parser.add_argument("--batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--num_parallel_workers", default=8, type=int, help="num workers for data loading")
    parser.add_argument(
        "--eval_loss",
        default=False,
        type=str2bool,
        help="whether measure loss including reconstruction, kl, perceptual loss",
    )
    parser.add_argument("--save_vis", default=True, type=str2bool, help="whether save reconstructed images")
    parser.add_argument("--encode_only", default=False, type=str2bool, help="only encode to save z or distribution")
    parser.add_argument(
        "--save_z_dist",
        default=False,
        type=str2bool,
        help="If True, save z distribution, mean and logvar. Otherwise, save z after sampling.",
    )
    # ms related
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="mixed precision type, if fp32, all layer precision is float32 (amp_level=O0),  \
                if bf16 or fp16, amp_level==O2, part of layers will compute in bf16 or fp16 such as matmul, dense, conv.",
    )
    parser.add_argument(
        "--vae_micro_batch_size", type=int, default=None, help="Set to one to reduce vae encoder's memory peak"
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--enable_tiling", action="store_true")
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
