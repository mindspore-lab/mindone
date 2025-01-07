"""
Infer and evaluate autoencoders
"""
import logging
import os
import sys
import time

import imageio
import numpy as np
from jsonargparse import ArgumentParser
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim
from tqdm import tqdm

import mindspore as ms
from mindspore import amp, nn, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))

from mg.dataset.tae_dataset import VideoDataset
from mg.models.tae import TemporalAutoencoder

from mindone.data import create_dataloader
from mindone.utils import init_train_env, set_logger

# from mg.models.tae.lpips import LPIPS


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
    # set env
    # TODO: rename as train and infer are identical?
    _, rank_id, device_num = init_train_env(mode=args.mode, ascend_config={"precision_mode": "must_keep_origin_dtype"})
    set_logger(name="", output_dir=args.output_path, rank=rank_id)

    # build model
    model = TemporalAutoencoder(pretrained=args.pretrained, use_tile=args.use_tile).set_train(False)
    if args.dtype != "fp32":
        dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        # FIXME: due to AvgPool and ops.interpolate doesn't support bf16, we add them to fp32 cells
        custom_fp32_cells = [nn.GroupNorm, nn.AvgPool2d, nn.Upsample]
        model = amp.custom_mixed_precision(model, black_list=amp.get_black_list() + custom_fp32_cells, dtype=dtype)
        logger.info(f"Set mixed precision to O2 with dtype={args.dtype}")

    # if args.eval_loss:
    #     lpips_loss_fn = LPIPS()

    # build dataset
    dataset = VideoDataset(
        csv_path=args.csv_path,
        folder=args.folder,
        size=args.image_size,
        crop_size=args.image_size,
        sample_n_frames=args.sample_n_frames,
        sample_stride=args.sample_stride,
        video_column=args.video_column,
        random_crop=False,
        flip=False,
        output_columns=["video"],
    )
    dataset = create_dataloader(
        dataset,
        args.batch_size,
        num_workers=8,
        max_rowsize=256,
        shuffle=False,
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=False,
    )
    num_batches = dataset.get_dataset_size()

    ds_iter = dataset.create_dict_iterator(num_epochs=1)

    mean_infer_time, mean_psnr, mean_ssim, mean_lpips, mean_recon, num_samples = (0,) * 6
    for step, data in tqdm(enumerate(ds_iter)):
        x = data["video"]
        start_time = time.perf_counter()

        if args.encode_only:
            z, posterior_mean, posterior_logvar = model.encode(x)
        else:
            recons, z, posterior_mean, posterior_logvar = model(x)

        infer_time = time.perf_counter() - start_time
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
                save_fn = os.path.join(args.output_path, f"{os.path.basename(args.video_folder)}-{f'step{step:03d}'}")
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_function_arguments(
        init_train_env, skip={"ascend_config", "num_workers", "json_data_path", "enable_modelarts"}
    )
    parser.add_class_arguments(TemporalAutoencoder, instantiate=False)
    parser.add_class_arguments(VideoDataset, skip={"output_columns"}, instantiate=False)
    parser.add_function_arguments(
        create_dataloader,
        skip={"dataset", "transforms", "batch_transforms", "device_num", "rank_id", "debug", "enable_modelarts"},
    )
    parser.add_argument(
        "--output_path", default="samples/vae_recons", type=str, help="output directory to save inference results"
    )
    parser.add_argument(
        "--expand_dim_t",
        default=False,
        type=bool,
        help="expand temporal axis for image data, used for vae 3d inference with image data",
    )
    parser.add_argument(
        "--eval_loss",
        default=False,
        type=bool,
        help="whether measure loss including reconstruction, kl, perceptual loss",
    )
    parser.add_argument("--save_vis", default=True, type=bool, help="whether save reconstructed images")
    parser.add_argument("--use_temporal_vae", default=True, type=bool, help="if False, just use spatial vae")
    parser.add_argument("--encode_only", default=False, type=bool, help="only encode to save z or distribution")
    parser.add_argument(
        "--save_z_dist",
        default=False,
        type=bool,
        help="If True, save z distribution, mean and logvar. Otherwise, save z after sampling.",
    )
    args = parser.parse_args()
    main(args)
