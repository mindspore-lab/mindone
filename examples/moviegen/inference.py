import datetime
import glob
import logging
import os
import sys
import time
from typing import List, Tuple

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

import mindspore as ms
from mindspore import Tensor, amp, nn

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.append(mindone_lib_path)

from moviegen.pipelines import InferPipeline
from moviegen.utils.model_utils import MODEL_DTYPE, init_model

from mindone.utils import init_train_env, set_logger
from mindone.visualize.videos import save_videos

# TODO: remove when VAE is added to the project
sys.path.append(os.path.join(__dir__, "../opensora_hpcai/"))
from opensora.models.vae.vae import OpenSoraVAE_V1_2

logger = logging.getLogger(__name__)

Path_dr = path_type("dr", docstring="path to a directory that exists and is readable")


def to_numpy(x: Tensor) -> np.ndarray:
    if x.dtype == ms.bfloat16:
        x = x.astype(ms.float32)
    return x.asnumpy()


def prepare_captions(
    ul2_dir: Path_dr, metaclip_dir: Path_dr, byt5_dir: Path_dr, rank_id: int, device_num: int
) -> Tuple[List[str], List[str], List[str]]:
    ul2_emb = sorted(glob.glob(os.path.join(ul2_dir, "*.npz")))
    metaclip_emb = sorted(glob.glob(os.path.join(metaclip_dir, "*.npz")))
    byt5_emb = sorted(glob.glob(os.path.join(byt5_dir, "*.npz")))
    if len(ul2_emb) != len(metaclip_emb) or len(ul2_emb) != len(byt5_emb):
        raise ValueError(
            f"ul2_dir ({len(ul2_emb)}), metaclip_dir ({len(metaclip_emb)}),"
            f" and byt5_dir ({len(byt5_emb)}) must contain the same number of files"
        )
    ul2_emb = ul2_emb[rank_id::device_num]
    logger.info(f"Number of captions for rank {rank_id}: {len(ul2_emb)}")
    return ul2_emb, metaclip_emb[rank_id::device_num], byt5_emb[rank_id::device_num]


def main(args):
    # TODO: CFG error
    save_dir = os.path.join(__dir__, args.output_path.relative)
    if args.append_timestamp:
        time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        save_dir = os.path.join(save_dir, time_str)
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    latent_dir = os.path.join(save_dir, "denoised_latents")
    if args.save_latent:
        os.makedirs(latent_dir, exist_ok=True)

    # 1. init env
    _, rank_id, device_num = init_train_env(**args.env)  # TODO: rename as train and infer are identical?

    # 1.1 read caption embeddings
    ul2_emb, metaclip_emb, byt5_emb = prepare_captions(**args.text_emb, rank_id=rank_id, device_num=device_num)

    # 2. model initiate and weight loading
    # 2.1 vae
    logger.info("vae init")
    vae_args = args.vae.as_dict()
    vae_dtype = vae_args.pop("dtype")
    vae = OpenSoraVAE_V1_2(**vae_args).set_train(False)
    if vae_dtype != "fp32":
        # FIXME: remove AMP and add custom dtype conversion support for better compatibility with PyNative
        amp.custom_mixed_precision(vae, black_list=amp.get_black_list() + [nn.GroupNorm], dtype=MODEL_DTYPE[vae_dtype])

    img_h, img_w = args.image_size if isinstance(args.image_size, list) else (args.image_size, args.image_size)
    num_frames = args.num_frames
    latent_size = vae.get_latent_size((num_frames, img_h, img_w))

    # 2.2 Llama 3
    model = init_model(in_channels=vae.out_channels, **args.model).set_train(False)

    # 2.3 text embeddings
    prompt_prefix = [os.path.basename(emb)[:-4] for emb in ul2_emb]
    ul2_emb = ms.Tensor([np.load(emb)["text_emb"] for emb in ul2_emb], dtype=ms.float32)
    metaclip_emb = ms.Tensor([np.load(emb)["text_emb"] for emb in metaclip_emb], dtype=ms.float32)
    byt5_emb = ms.Tensor([np.load(emb)["text_emb"] for emb in byt5_emb], dtype=ms.float32)
    num_prompts = ul2_emb.shape[0]

    # 3. build inference pipeline
    pipeline = InferPipeline(
        model,
        vae,
        latent_size,
        scale_factor=args.scale_factor,  # FIXME: refactor
        guidance_scale=args.guidance_scale,
        num_sampling_steps=args.num_sampling_steps,
        sample_method=args.sample_method,
        micro_batch_size=args.micro_batch_size,
    )

    # 4. print key info
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.env.mode}",
            f"Num of captions: {num_prompts}",
            f"Model dtype: {args.model.dtype}",
            f"VAE dtype: {vae_dtype}",
            f"Image size: {(img_h, img_w)}",
            f"Num frames: {num_frames}",
            f"Sampling steps {args.num_sampling_steps}",
            f"CFG guidance scale: {args.guidance_scale}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for i in range(0, num_prompts, args.batch_size):
        end_i = min(i + args.batch_size, num_prompts)
        logger.info("Sampling captions:")
        for j in range(i, end_i):
            logger.info(prompt_prefix[j])

        # infer
        start_time = time.perf_counter()
        sample, latent = pipeline(
            ul2_emb=ul2_emb[i:end_i],
            metaclip_emb=metaclip_emb[i:end_i],
            byt5_emb=byt5_emb[i:end_i],
            num_frames=num_frames,
        )
        batch_time = time.perf_counter() - start_time
        logger.info(
            f"Batch time cost: {batch_time:.3f}s,"
            f" sampling speed: {args.num_sampling_steps * (end_i - i) / batch_time:.2f} step/s"
        )

        # save result
        for j in range(0, end_i - i):
            fn = prompt_prefix[i + j]
            save_fp = f"{save_dir}/{fn}.{args.save_format}"
            latent_save_fp = f"{latent_dir}/{fn}.npy"

            # save videos
            if sample is not None:
                save_videos(to_numpy(sample[j]), save_fp, fps=args.fps)
                logger.info(f"Video saved in {save_fp}")
            # save decoded latents
            if args.save_latent:
                np.save(latent_save_fp, to_numpy(latent[j : j + 1]))
                logger.info(f"Denoised latents saved in {latent_save_fp}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Movie Gen inference script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_train_env, "env")
    parser.add_function_arguments(init_model, "model", skip={"in_channels"})
    vae_group = parser.add_argument_group("VAE parameters")
    vae_group.add_function_arguments(OpenSoraVAE_V1_2, "vae", fail_untyped=False)
    vae_group.add_argument(
        "--vae.dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="VAE model precision."
    )
    infer_group = parser.add_argument_group("Inference parameters")
    infer_group.add_class_arguments(InferPipeline, skip={"model", "vae", "latent_size"}, instantiate=False)
    infer_group.add_argument("--image_size", type=int, nargs="+", help="Output video size")
    infer_group.add_argument("--num_frames", type=int, default=17, help="number of frames")
    infer_group.add_argument("--fps", type=int, default=16, help="FPS in the saved video")
    infer_group.add_function_arguments(prepare_captions, "text_emb", skip={"rank_id", "device_num"})
    infer_group.add_argument("--batch_size", type=int, default=1)
    save_group = parser.add_argument_group("Saving options")
    save_group.add_argument(
        "--save_format",
        default="mp4",
        choices=["gif", "mp4", "png"],
        type=str,
        help="video format for saving the sampling output: gif, mp4 or png",
    )
    save_group.add_argument(
        "--output_path",
        default="output/",
        type=path_type("dcc"),  # path to a directory that can be created if it does not exist
        help="Output directory to save training results.",
    )
    save_group.add_argument(
        "--append_timestamp",
        type=bool,
        default=True,
        help="If true, a subfolder named with timestamp under output_path will be created to save the sampling results",
    )
    save_group.add_argument(
        "--save_latent",
        type=bool,
        default=False,
        help="Save denoised video latent. If True, the denoised latents will be saved in $output_path/denoised_latents",
    )
    cfg = parser.parse_args()
    main(cfg)
