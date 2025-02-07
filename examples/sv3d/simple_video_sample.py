import argparse
import math
import os
import sys
from glob import glob
from pathlib import Path
from typing import Optional

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

from sgm.helpers import create_model_sv3d
from sgm.util import default, instantiate_from_config
from tools.vid2gif import DumperGif

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.dataset.vision import ToTensor

from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed


class SV3DInferPipeline:
    def __init__(
        self,
        model_config: str,
        ckpt_path: str,
        num_frames: Optional[int],
        num_steps: int,
        version: str,
        cond_aug: float,
        decoding_t: int = 14,
        elevations_deg: float = None,  # for sv3d_p
        azimuths_deg: float = None,
        verbose=False,
        amp_level: Literal["O0", "O2"] = "O0",
    ):
        model_config = OmegaConf.load(model_config)
        model_config.model.params.sampler_config.params.verbose = verbose
        model_config.model.params.sampler_config.params.num_steps = num_steps
        model_config.model.params.sampler_config.params.guider_config.params.num_frames = num_frames
        _config_arch_toload_vanilla_sv3d_ckpt = True if version == "sv3d_u" else False
        self.model, _ = create_model_sv3d(
            model_config,
            checkpoints=ckpt_path,
            freeze=True,
            amp_level=amp_level,
            config_arch_toload_vanilla_sv3d_ckpt=_config_arch_toload_vanilla_sv3d_ckpt,
        )
        self.model.en_and_decode_n_samples_a_time = decoding_t

        # for the new sampler only
        sampler_cfg = model_config.model.params.sampler_config
        sampler_cfg.params["network_config"] = model_config
        sampler_cfg.params["network_ckpt"] = ckpt_path
        self.sampler = instantiate_from_config(sampler_cfg)

        # for alignment of randomness with th
        img_shape = (576, 576)
        self.cond_c = np.random.randn(*img_shape).astype(np.float32)
        self.randn_n = np.random.randn(num_frames, 4, img_shape[0] // 8, img_shape[1] // 8).astype(np.float32)

        self.cond_aug = cond_aug
        self.num_frames = num_frames
        self.version = version

        # preproc for graph mode
        self.expand_dims_ops = ops.ExpandDims()

        if amp_level == "O2":
            auto_mixed_precision(self.model)

    @ms.jit
    def get_batch(
        self,
        cond_frames: Tensor,
        cond_frames_without_noise: Tensor,
    ):
        batch = {
            "cond_frames": cond_frames,
            "cond_frames_without_noise": cond_frames_without_noise,
        }
        batch_uc = {}
        batch["cond_aug"] = Tensor(self.cond_aug).repeat(math.prod([1, self.num_frames]))

        for key in batch.keys():
            if key not in batch_uc and isinstance(batch[key], Tensor):
                batch_uc[key] = batch[key].copy()
        return batch, batch_uc

    @ms.jit
    def extract_samples(self, samples_z: Tensor, cond_frames_without_noise: Tensor):
        samples_x = self.model.decode_first_stage(samples_z)
        samples_x[-1:] = cond_frames_without_noise
        samples = ops.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        return samples

    def __call__(self, image: Tensor) -> Tensor:
        image = image * 2.0 - 1.0
        image = image.unsqueeze(0)
        assert image.shape[1] == 3
        _cond_aug = Tensor(self.cond_aug, dtype=ms.float32)
        cond_frames = image + _cond_aug * Tensor(self.cond_c)
        cond_frames_without_noise = image
        batch, batch_uc = self.get_batch(cond_frames, cond_frames_without_noise)

        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch,
            batch_uc,
            force_uc_zero_embeddings=[
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )

        for k in ["crossattn", "concat"]:
            uc[k] = self.expand_dims_ops(uc[k], 1)
            uc[k] = uc[k].repeat(self.num_frames, axis=1)
            uc[k] = uc[k].flatten(order="C", start_dim=0, end_dim=1)
            c[k] = self.expand_dims_ops(c[k], 1)
            c[k] = c[k].repeat(self.num_frames, axis=1)
            c[k] = c[k].flatten(order="C", start_dim=0, end_dim=1)

        randn = Tensor(self.randn_n)
        additional_model_inputs = {}
        additional_model_inputs["image_only_indicator"] = ops.zeros((2, self.num_frames))
        samples_z = self.sampler(randn, cond=c, uc=uc, num_frames=self.num_frames, **additional_model_inputs)

        samples = self.extract_samples(samples_z, cond_frames_without_noise)

        return samples


def sample(
    input_path: str,
    ckpt_path: str,
    version: str = "sv3d_u",
    num_steps: Optional[int] = None,
    seed: int = 42,
    decoding_t: int = 7,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    mode: int = 1,
    output_folder: Optional[str] = None,
    image_frame_ratio: Optional[float] = None,
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """

    if version == "sv3d_u":
        num_frames = 21
    elif version == "sv3d_u_overfitted_ckpt":
        num_frames = 6  # overfitted under this number, larger leads to oom during overfitting
    else:
        raise ValueError(f"Version {version} is not supported for this example yet.")
    num_steps = default(num_steps, 50)
    output_folder = default(output_folder, "outputs/sv3d_u/")
    model_config = "configs/sampling/sv3d_u.yaml"
    cond_aug = 1e-5

    logger = set_logger(
        name="",
        output_dir=str(output_folder),
    )  # all the logger needs to follow name, to use the mindone callbacks directly, need to put name as ""
    logger.info("program started")

    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=mode, device_target="Ascend", device_id=device_id)
    set_random_seed(seed)
    path = Path(input_path)
    pipeline = SV3DInferPipeline(model_config, ckpt_path, num_frames, num_steps, version, cond_aug, decoding_t)
    all_img_paths = []
    logger.info(f"path posix: {path}")
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [f for f in path.iterdir() if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        )
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError("Input img path unavailable.")

    gif_dumper = DumperGif()

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
            ret, mask = cv2.threshold(np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY)
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = int(max_size / image_frame_ratio) if image_frame_ratio is not None else in_w
            padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            padded_image[
                center - h // 2 : center - h // 2 + h,
                center - w // 2 : center - w // 2 + w,
            ] = image_arr[y : y + h, x : x + w]

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
        image = Tensor(ToTensor()(input_image), ms.float32)
        logger.info(f"Starting 3D generation for {input_img_path}, this may take a while...")
        samples = pipeline(image)
        logger.info("program finished")

        logger.info(f"Samples of input {input_img_path} are generated")
        os.makedirs(output_folder, exist_ok=True)
        base_count = len(glob(os.path.join(output_folder, "*.mp4")))

        imageio.imwrite(os.path.join(output_folder, f"{base_count: 06d}.jpg"), input_image)
        samples = samples.asnumpy()
        vid = (rearrange(samples, "t c h w -> t h w c") * 255).astype(np.uint8)

        video_path = os.path.join(output_folder, f"{base_count: 06d}.mp4")
        imageio.mimwrite(video_path, vid)
        gif_path = os.path.join(output_folder, f"{base_count: 06d}.gif")
        gif_dumper.vid2gif(gif_path, vid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default="PATHTOYOURCKPT",
        type=str,
        help="path to the ckpt",
    )
    parser.add_argument(
        "--input",
        default="PATHTOYOURINPUT",
        type=str,
        help="path to the input img, or the input imgs dir path",
    )
    parser.add_argument(
        "--mode", default=1, type=int, help="MindSpore execution mode: Graph mode[0] or Pynative mode[1]"
    )
    parser.add_argument("--decoding_t", default=7, type=int, help="# of decoding steps")
    parser.add_argument("--version", default="sv3d_u", choices=["sv3d_u", "sv3d_u_overfitted_ckpt"], type=str)
    args = parser.parse_args()
    print(args)
    sample(args.input, args.ckpt, args.version, decoding_t=args.decoding_t, mode=args.mode)
