""" Note that this file is adapted from `simple_video_sample.py`, as modular design is current not supported under examples.

Adapted from https://github.com/Stability-AI/generative-models/blob/main/scripts/sampling/simple_video_sample.py
python simple_video_sample.py --version sv3d_u
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from loguru import logger
from glob import glob
from pathlib import Path
from typing import List, Optional

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

import cv2
import imageio
import numpy as np
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from rembg import remove

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

from sgm.util import default
from sgm.helpers import create_model_sv3d as create_model

import mindspore as ms
from mindspore.dataset.vision import ToTensor
from mindspore import Tensor, ops, nn
from mindone.utils.seed import set_random_seed
from utils import mixed_precision


class SVD3InferPipeline(nn.Cell):
    def __init__(self,
                 model_config: str,
                 ckpt_path: str,
                 num_frames: Optional[int],  # 21 for SV3D
                 device: str,
                 num_steps: int,
                 version: str,
                 motion_bucket_id: int,
                 fps_id: int,
                 cond_aug: int,
                 decoding_t: int = 14,
                 verbose=False,
                 amp_level: Literal["O0", "O2"] = "O0"
                 ):
        super().__init__()
        model_config = OmegaConf.load(model_config)
        model_config.model.params.sampler_config.params.verbose = verbose
        model_config.model.params.sampler_config.params.num_steps = num_steps
        model_config.model.params.sampler_config.params.guider_config.params.num_frames = (
            num_frames
        )
        self.model, _ = create_model(
            model_config,
            checkpoints=ckpt_path,
            freeze=True,
            amp_level=amp_level
        )
        self.num_frames = num_frames
        self.version = version
        self.motion_bucket_id = motion_bucket_id
        self.fps_id = fps_id
        self.cond_aug = cond_aug
        self.decoding_t = decoding_t
        self.device = device

        if amp_level == "O2":
            mixed_precision(self.model)

    def construct(self, image: Tensor) -> Tensor:
        image = image * 2.0 - 1.0

        image = image.unsqueeze(0)
        H, W = image.shape[2:]
        assert image.shape[1] == 3
        F = 8
        C = 4
        shape = (self.num_frames, C, H // F, W // F)
        if (H, W) != (576, 1024) and "sv3d" not in self.version:
            print(
                "WARNING: The conditioning frame you provided is not 576x1024. This leads to suboptimal performance as model was only trained on 576x1024. Consider increasing `cond_aug`."
            )
        if (H, W) != (576, 576) and "sv3d" in self.version:
            print(
                "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as model was only trained on 576x576."
            )
        if self.motion_bucket_id > 255:
            print(
                "WARNING: High motion bucket! This may lead to suboptimal performance."
            )

        if self.fps_id < 5:
            print("WARNING: Small fps value! This may lead to suboptimal performance.")

        if self.fps_id > 30:
            print("WARNING: Large fps value! This may lead to suboptimal performance.")

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = self.motion_bucket_id
        value_dict["fps_id"] = self.fps_id
        value_dict["cond_aug"] = self.cond_aug
        value_dict["cond_frames"] = image + self.cond_aug * ops.randn_like(image)

        # with torch.no_grad():
        # ops.stop_gradient(model)
        # with torch.autocast(device):
        batch, batch_uc = get_batch(
            get_unique_embedder_keys_from_conditioner(self.model.conditioner),
            value_dict,
            [1, self.num_frames],
            T=self.num_frames,
        )
        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc=batch_uc,
            force_uc_zero_embeddings=[
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )
        expand_dims_ops = ops.ExpandDims()

        for k in ["crossattn", "concat"]:
            uc[k] = expand_dims_ops(uc[k], 1)
            uc[k] = uc[k].repeat(self.num_frames, axis=1)
            uc[k] = uc[k].flatten(order='C', start_dim=0, end_dim=1)
            c[k] = expand_dims_ops(c[k], 1)
            c[k] = c[k].repeat(self.num_frames, axis=1)
            c[k] = c[k].flatten(order='C', start_dim=0, end_dim=1)

        randn = ops.randn(shape)

        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = ops.zeros(
            (2, self.num_frames)
        )
        # additional_model_inputs["num_video_frames"] = batch["num_video_frames"]

        # calling self.model.model under .construct() is NOT working
        # def denoiser(input, sigma, c):
        #     return self.model.denoiser(
        #         self.model.model, input, sigma, c, **additional_model_inputs
        #     )

        samples_z = self.model.sampler(self.model, randn, cond=c, uc=uc, num_frames=self.num_frames, **additional_model_inputs)

        # # unlike the sv3d version of sdxl, we followed the original sdxl in mindone by passing in the whole model to sampler, rather than a denoiser in sv3d, as in the openai_wrapper the ms concat func does not support the current rank setup
        # samples_z = self.model.sampler(model, randn, cond=c, uc=uc, num_frames=num_frames)
        self.model.en_and_decode_n_samples_a_time = self.decoding_t
        samples_x = self.model.decode_first_stage(samples_z)
        if "sv3d" in self.version:
            samples_x[-1:] = value_dict["cond_frames_without_noise"]
        samples = ops.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        return samples


def sample(
        input_path: str,
        ckpt_path: str,
        num_steps: Optional[int] = None,
        version: str = "sv3d_u",
        fps_id: int = 6,
        motion_bucket_id: int = 127,
        seed: int = 42,
        decoding_t: int = 7,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
        device: str = "Ascend",
        output_folder: Optional[str] = None,
        image_frame_ratio: Optional[float] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "sv3d_u":
        num_frames = 21
        num_steps = default(num_steps, 50)
        output_folder = default(output_folder, "outputs/sv3d_u/")
        model_config = "configs/sampling/sv3d_u.yaml"
        cond_aug = 1e-5
    else:
        raise ValueError(f"Version {version} is not supported for this example yet.")

    ms.context.set_context(mode=1, device_target=device, device_id=2)
    set_random_seed(seed)
    path = Path(input_path)
    pipeline = SVD3InferPipeline(model_config, ckpt_path, num_frames, device, num_steps,
                                 version, motion_bucket_id, fps_id, cond_aug, decoding_t)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError("Input img path unavailable.")

    for input_img_path in all_img_paths:
        if "sv3d" in version:
            image = Image.open(input_img_path)
            if image.mode == "RGBA":
                pass
            else:
                # remove bg
                image.thumbnail([768, 768], Image.Resampling.LANCZOS)
                image = remove(image.convert("RGBA"), alpha_matting=True)

            # resize object in frame
            image_arr = np.array(image)
            in_w, in_h = image_arr.shape[:2]
            ret, mask = cv2.threshold(
                np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = (
                int(max_size / image_frame_ratio)
                if image_frame_ratio is not None
                else in_w
            )
            padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image[
            center - h // 2: center - h // 2 + h,
            center - w // 2: center - w // 2 + w,
            ] = image_arr[y: y + h, x: x + w]

            # resize frame to 576x576
            rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
            # white bg
            rgba_arr = np.array(rgba) / 255.0
            rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
            input_image = Image.fromarray((rgb * 255).astype(np.uint8))

        else:
            with Image.open(input_img_path) as image:
                if image.mode == "RGBA":
                    input_image = image.convert("RGB")
                w, h = image.size

                if h % 64 != 0 or w % 64 != 0:
                    width, height = map(lambda x: x - x % 64, (w, h))
                    input_image = input_image.resize((width, height))
                    print(
                        f"WARNING: Your image is of size {h}x{w} which is not divisible by 64. We are resizing to {height}x{width}!"
                    )
        image = Tensor(ToTensor()(input_image))
        logger.info("Starting 3D generation, this may take a while...")
        samples = pipeline(image)
        logger.info("Samples generated")
        os.makedirs(output_folder, exist_ok=True)
        base_count = len(glob(os.path.join(output_folder, "*.mp4")))

        imageio.imwrite(
            os.path.join(output_folder, f"{base_count:06d}.jpg"), input_image
        )
        # samples = embed_watermark(samples)  # TODO get the filtering/watermarking work
        # samples = filter(samples)
        samples = samples.asnumpy()
        vid = (
            (rearrange(samples, "t c h w -> t h w c") * 255)
            .astype(np.uint8)
        )

        video_path = os.path.join(output_folder, f"{base_count:06d}.mp4")
        imageio.mimwrite(video_path, vid)


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                Tensor([value_dict["fps_id"]])
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                Tensor([value_dict["motion_bucket_id"]])
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = Tensor([value_dict["cond_aug"]]).repeat(math.prod(N))
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            # batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
            # nothing happens when bsize==N[0]==1, so just skip first
            batch[key] = value_dict[key]
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = Tensor(value_dict[key]).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], Tensor):
            batch_uc[key] = batch[key].copy()
    return batch, batch_uc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='PATHTOYOURCKPT', type=str, help='path to the ckpt')
    parser.add_argument('--input', default='data/test_image.png', type=str, help='path to the input img')
    args = parser.parse_args()
    print(args)
    sample(args.input, args.ckpt)

