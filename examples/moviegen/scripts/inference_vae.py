# flake8: noqa
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

from mindspore import nn, ops

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_dir = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_dir)


from omegaconf import OmegaConf
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from mg.datasets.tae_dataset import create_dataloader
from mg.models.tae.lpips import LPIPS
from mg.models.tae.tae import TemporalAutoencoder

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


def rearrange_in(x):
    b, c, t, h, w = x.shape
    x = ops.transpose(x, (0, 2, 3, 4, 1))
    x = ops.reshape(x, (b * t, h, w, c))
    return x


def rearrange_out(x, t):
    bt, c, h, w = x.shape
    b = bt // t
    x = ops.reshape(x, (b, t, h, w, c))
    x = ops.transpose(x, (0, 4, 1, 2, 3))
    return x


def main(args):
    ascend_config = {"precision_mode": "must_keep_origin_dtype"}
    ms.set_context(mode=args.mode, ascend_config=ascend_config)
    ms.set_context(jit_config={"jit_level": args.jit_level})
    set_logger(name="", output_dir=args.output_path, rank=0)

    # build model
    model = TemporalAutoencoder(
        pretrained=args.ckpt_path,
        use_tile=args.enable_tile,
    )

    model.set_train(False)
    logger.info(f"Loaded checkpoint from  {args.ckpt_path}")

    if args.eval_loss:
        lpips_loss_fn = LPIPS()

    if args.dtype != "fp32":
        amp_level = "O2"
        dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        # FIXME: due to AvgPool and ops.interpolate doesn't support bf16, we add them to fp32 cells
        custom_fp32_cells = [nn.GroupNorm, nn.AvgPool2d, nn.Upsample]
        model = auto_mixed_precision(model, amp_level, dtype, custom_fp32_cells)
        logger.info(f"Set mixed precision to O2 with dtype={args.dtype}")
    else:
        amp_level = "O0"

    # build dataset
    if isinstance(args.image_size, int):
        image_size = args.image_size
    else:
        if len(args.image_size) == 2:
            assert args.image_size[0] == args.image_size[1], "Currently only h==w is supported"
        image_size = args.image_size[0]

    ds_config = dict(
        csv_path=args.csv_path,
        data_folder=args.video_folder,
        size=image_size,
        crop_size=image_size,
        sample_n_frames=args.num_frames,
        sample_stride=args.frame_stride,
        video_column=args.video_column,
        random_crop=False,
        flip=False,
    )
    dataset = create_dataloader(
        ds_config,
        args.batch_size,
        mixed_strategy=None,
        mixed_image_ratio=0.0,
        num_parallel_workers=8,
        max_rowsize=256,
        shuffle=False,
        device_num=1,
        rank_id=0,
        drop_remainder=False,
    )
    num_batches = dataset.get_dataset_size()

    ds_iter = dataset.create_dict_iterator(1)

    if args.dynamic_shape:
        videos = ms.Tensor(shape=[None, 3, None, 256, 256], dtype=ms.float32)
        model.set_inputs(videos)

    logger.info("Inferene begins")
    mean_infer_time = 0
    mean_psnr = 0
    mean_ssim = 0
    mean_lpips = 0
    mean_recon = 0
    num_samples = 0
    for step, data in tqdm(enumerate(ds_iter)):
        x = data["video"]
        start_time = time.time()

        # debug
        # if args.dynamic_shape:
        #    if step % 2 == 0:
        #        x = x[:, :, : x.shape[2]//2]
        #    print('x shape: ', x.shape)

        if args.encode_only:
            z = model.encode(x)
        else:
            # recons = model.decode(z)
            recons, z, posterior_mean, posterior_logvar = model(x)

        # adapt to bf16
        recons = recons.to(ms.float32)

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

            logger.info(f"cur psnr: {psnr_cur[-1]:.4f}, mean psnr:{mean_psnr/num_samples:.4f}")
            logger.info(f"cur ssim: {ssim_cur[-1]:.4f}, mean ssim:{mean_ssim/num_samples:.4f}")

            if args.eval_loss:
                recon_loss = np.abs((x - recons).asnumpy())

                t = x.shape[2]
                x = rearrange_in(x)
                # lpips_loss = lpips_loss_fn(x, recons).asnumpy()

                mean_recon += recon_loss.mean()
                # mean_lpips += lpips_loss.mean()
                logger.info(f"mean recon loss: {mean_recon/num_batches:.4f}")

            if args.save_vis:
                save_fn = os.path.join(
                    args.output_path, "{}-{}".format(os.path.basename(args.video_folder), f"step{step:03d}")
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
            # mean_lpips /= num_batches
            logger.info(f"mean recon loss: {mean_recon:.4f}")
            # logger.info(f"mean lpips loss: {mean_lpips:.4f}")


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
        help="path to csv annotation file. If None, will get videos from the folder of `data_path`",
    )
    parser.add_argument("--video_folder", default=None, type=str, help="folder of videos")
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
    parser.add_argument("--image_size", default=256, type=int, help="image rescale size")
    # parser.add_argument("--crop_size", default=256, type=int, help="image crop size")

    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--num_parallel_workers", default=8, type=int, help="num workers for data loading")
    parser.add_argument(
        "--eval_loss",
        default=False,
        type=str2bool,
        help="whether measure loss including reconstruction, kl, perceptual loss",
    )
    parser.add_argument("--save_vis", default=True, type=str2bool, help="whether save reconstructed images")
    parser.add_argument("--use_temporal_vae", default=True, type=str2bool, help="if False, just use spatial vae")
    parser.add_argument("--encode_only", default=False, type=str2bool, help="only encode to save z or distribution")
    parser.add_argument(
        "--enable_tile", default=False, type=str2bool, help="enable temporal tiling with linear blending for decoder"
    )
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--mixed_strategy",
        type=str,
        default=None,
        choices=[None, "mixed_video_image", "image_only"],
        help="video and image mixed strategy.",
    )
    parser.add_argument(
        "--mixed_image_ratio", default=0.0, type=float, help="image ratio in mixed video and image data training"
    )
    parser.add_argument(
        "--save_z_dist",
        default=False,
        type=str2bool,
        help="If True, save z distribution, mean and logvar. Otherwise, save z after sampling.",
    )
    parser.add_argument(
        "--dynamic_shape", default=False, type=str2bool, help="whether input shape to the network is dynamic"
    )

    # ms related
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--jit_level", default="O0", type=str, help="O0 kbk, O1 dvm, O2 ge")
    parser.add_argument(
        "--dtype",
        default="fp32",
        type=str,
        choices=["fp32", "fp16", "bf16"],
        help="mixed precision type, if fp32, all layer precision is float32 (amp_level=O0),  \
                if bf16 or fp16, amp_level==O2, part of layers will compute in bf16 or fp16 such as matmul, dense, conv.",
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
