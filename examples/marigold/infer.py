# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------

import argparse
import logging
import os

import numpy as np
from marigold import MarigoldPipeline
from omegaconf import OmegaConf
from PIL import Image
from src.dataset import BaseDepthDataset, DatasetMode, get_dataset, get_pred_name
from src.util.msckpt_utils import build_model_from_config, load_pretrained_model, replace_unet_conv_in
from src.util.seeding import seed_all
from tqdm.auto import tqdm

import mindspore.dataset as ds
from mindspore import context
from mindspore import dtype as mstype

from mindone.diffusers import DDIMScheduler

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(description="Run single-image depth estimation using Marigold.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="marigold-checkpoint/marigold-v1-0",
        help="Checkpoint path or hub name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="Ascend",
        choices=["Ascend", "CPU"],
        help="Device to run the inference.",
    )
    parser.add_argument(
        "--ms_ckpt",
        action="store_true",
        default=False,
        help="Use checkpoint we retrain by mindspore.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument("--output_dir", type=str, required=True, help="Output directory.")

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=10,
        help="Number of predictions to be ensembled, more inference gives better results but runs slower.",
    )
    parser.add_argument(
        "--half_precision",
        "--fp16",
        default=False,
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--output_processing_res",
        action="store_true",
        help="When input is resized, out put depth at resized operating resolution. Default: False.",
    )
    parser.add_argument(
        "--resample_method",
        type=str,
        default="bilinear",
        help="Resampling method used to resize images. This can be one of 'bilinear' or 'nearest'.",
    )

    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir
    device = args.device

    denoise_steps = args.denoise_steps
    ensemble_size = args.ensemble_size
    if ensemble_size > 15:
        logging.warning("Running with large ensemble size will be slow.")

    processing_res = args.processing_res
    match_input_res = not args.output_processing_res
    if 0 == processing_res and match_input_res is False:
        logging.warning(
            "Processing at native resolution without resizing output might NOT lead to exactly the same resolution, \
            due to the padding and pooling properties of conv layers."
        )
    resample_method = args.resample_method

    if args.half_precision:
        dtype = mstype.float16
        logging.warning(f"Running with half precision ({dtype}), might lead to suboptimal result.")
    else:
        dtype = mstype.float32

    seed = args.seed

    print(f"arguments: {args}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"Inference settings: checkpoint = `{checkpoint_path}`, "
        f"with denoise_steps = {denoise_steps}, ensemble_size = {ensemble_size}, "
        f"processing resolution = {processing_res}, seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    def check_directory(directory):
        if os.path.exists(directory):
            response = (
                input(f"The directory '{directory}' already exists. Are you sure to continue? (y/n): ").strip().lower()
            )
            if "y" == response:
                pass
            elif "n" == response:
                print("Exiting...")
                exit()
            else:
                print("Invalid input. Please enter 'y' (for Yes) or 'n' (for No).")
                check_directory(directory)  # Recursive call to ask again

    check_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    context.set_context(mode=context.PYNATIVE_MODE, device_target=device)
    if "CPU" == device:
        logging.warning("Running on CPU will be slow.")
    logging.info(f"Inference device = {device}, with mode = {context.get_context('mode')}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY, dtype=dtype
    )

    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=["dict"],
        num_parallel_workers=1,
    ).batch(1)

    # -------------------- Model --------------------
    if not args.ms_ckpt:
        pipe: MarigoldPipeline = MarigoldPipeline.from_pretrained(checkpoint_path, mindspore_dtype=dtype)
    else:
        latent_diffusion_with_loss = build_model_from_config("./config/v2-vpred-train.yaml", False)
        load_pretrained_model(checkpoint_path, latent_diffusion_with_loss, unet_initialize_random=False)
        replace_unet_conv_in(latent_diffusion_with_loss, args.half_precision)
        pipe = MarigoldPipeline(
            unet=latent_diffusion_with_loss.model.diffusion_model,
            vae=latent_diffusion_with_loss.first_stage_model,
            scheduler=DDIMScheduler.from_config("./config/scheduler_config.json"),
            text_encoder=latent_diffusion_with_loss.cond_stage_model,
            tokenizer=latent_diffusion_with_loss.cond_stage_model,
            scale_invariant=True,
            shift_invariant=True,
            default_denoising_steps=10,
            default_processing_resolution=768,
            is_ms_ckpt=True,
        )

    logging.info(f"scale_invariant: {pipe.scale_invariant}, shift_invariant: {pipe.shift_invariant}")

    # -------------------- Inference and saving --------------------
    for batch in tqdm(dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True):
        batch = batch[0]
        # Read input image
        rgb_int = batch["rgb_int"].squeeze().asnumpy().astype(np.uint8)  # [3, H, W]
        rgb_int = np.moveaxis(rgb_int, 0, -1)  # [H, W, 3]
        input_image = Image.fromarray(rgb_int)

        # Predict depth
        pipe_out = pipe(
            input_image,
            denoising_steps=denoise_steps,
            ensemble_size=ensemble_size,
            processing_res=processing_res,
            match_input_res=match_input_res,
            batch_size=0,
            color_map=None,
            show_progress_bar=True,
            resample_method=resample_method,
        )

        depth_pred: np.ndarray = pipe_out.depth_np

        # Save predictions
        rgb_filename = batch["rgb_relative_path"].item()
        rgb_basename = os.path.basename(rgb_filename)
        scene_dir = os.path.join(output_dir, os.path.dirname(rgb_filename))
        if not os.path.exists(scene_dir):
            os.makedirs(scene_dir)
        pred_basename = get_pred_name(rgb_basename, dataset.name_mode, suffix=".npy")
        save_to = os.path.join(scene_dir, pred_basename)
        if os.path.exists(save_to):
            logging.warning(f"Existing file: '{save_to}' will be overwritten")

        np.save(save_to, depth_pred)
