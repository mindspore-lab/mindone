import logging
import sys
from datetime import datetime
from pathlib import Path

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import cv2
import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import Path_fr, path_type
from omegaconf import OmegaConf
from PIL import Image
from utils import mixed_precision, seed_everything

import mindspore as ms
from mindspore import Tensor, nn, ops

sys.path.append("../../")  # FIXME: loading modules from the SDXL directory
from modules.helpers import create_model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


class SVDInferPipeline(nn.Cell):
    """
    Stable Video Diffusion inference pipeline.

    Args:
        config: Path to the SVD model config file.
        checkpoint: Path to the checkpoint file.
        num_frames: The number of frames to generate. Recommended: 14 for svd and 25 for svd-xt
        fps: Frames per second. Default: 7.
        motion_bucket_id: The motion bucket ID. Higher values lead to more motion in the generated video.
                          It is recommended to keep it under 255. Default: 127.
        noise_aug_strength: The amount of noise added to the input image. The higher the value, the less the video
                            will resemble the input image. Increase it for more motion. Default: 0.02.
        decode_chunk_size: The number of frames to decode at a time. The higher the chunk size, the higher the temporal
                           consistency between frames, but also the higher the memory consumption. By default, the
                           decoder will decode all frames at once for maximal quality. Reduce the value to reduce memory
                           usage.
        sampling_steps: The number of sampling steps. Default: following the config file.
        amp_level: The Automatic Mixed Precision level to use. Default: "O2".
    """

    def __init__(
        self,
        config: Path_fr,
        checkpoint: Path_fr,
        num_frames: int,
        fps: int,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 0,
        sampling_steps: int = 0,
        amp_level: Literal["O0", "O2"] = "O2",
    ):
        super().__init__()

        if motion_bucket_id > 255:
            logger.warning(
                "High motion bucket may lead to suboptimal performance. It's recommended to keep it under 255."
            )

        if fps < 5 or fps > 30:
            logger.warning(
                "Too low / high FPS may lead to suboptimal performance. It's recommended to keep it between 5 and 30."
            )

        config = OmegaConf.load(config.absolute)
        if sampling_steps:
            config.model.params.sampler_config.params.num_steps = sampling_steps

        self.model, _ = create_model(config, checkpoints=checkpoint.absolute, freeze=True, amp_level="O2")

        self._num_frames = num_frames
        self._in_channels = self.model.model.diffusion_model.in_channels
        self._f = 2 ** (self.model.first_stage_model.encoder.num_resolutions - 1)

        if amp_level == "O2":
            mixed_precision(self.model)

        self._fps_id = Tensor(fps - 1, dtype=ms.float32)
        self._motion_bucket_id = Tensor(motion_bucket_id, dtype=ms.float32)
        self._noise_aug_strength = Tensor(noise_aug_strength, dtype=ms.float32)
        self._decode_chunk_size = decode_chunk_size or num_frames

    def _get_batch(self, cond_frames: Tensor, cond_frames_without_noise: Tensor):
        batch = {
            "cond_frames": cond_frames,
            "cond_frames_without_noise": cond_frames_without_noise,
        }

        keys = set([x.input_key for x in self.model.conditioner.embedders])

        if "fps_id" in keys:
            batch["fps_id"] = self._fps_id.repeat(self._num_frames)
        if "motion_bucket_id" in keys:
            batch["motion_bucket_id"] = self._motion_bucket_id.repeat(self._num_frames)
        if "cond_aug" in keys:
            batch["cond_aug"] = self._noise_aug_strength.repeat(self._num_frames)

        return batch

    def construct(self, image: Tensor) -> Tensor:
        cond_frames_without_noise = image
        cond_frames = image + self._noise_aug_strength * ops.randn_like(image)

        batch = self._get_batch(cond_frames, cond_frames_without_noise)

        c, uc = self.model.conditioner.get_unconditional_conditioning(
            batch,
            force_uc_zero_embeddings=[
                "cond_frames",
                "cond_frames_without_noise",
            ],
        )

        for k in ["crossattn", "concat"]:
            c[k] = c[k].repeat(self._num_frames, axis=0)
            uc[k] = uc[k].repeat(self._num_frames, axis=0)

        H, W = image.shape[2:]
        noise = ops.randn(self._num_frames, self._in_channels // 2, H // self._f, W // self._f)

        samples_z = self.model.sampler(self.model, noise, cond=c, uc=uc, num_frames=self._num_frames)
        self.model.en_and_decode_n_samples_a_time = self._decode_chunk_size
        samples_x = self.model.decode_first_stage(samples_z)

        samples = ops.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        vid = samples.transpose((0, 2, 3, 1))  # t c h w -> t h w c

        return (vid * 255).to(ms.uint8)


def prepare_image(image_path: str) -> np.ndarray:
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        w, h = image.size

        if h % 64 != 0 or w % 64 != 0:
            width, height = map(lambda x: x - x % 64, (w, h))
            image = image.resize((width, height))
            logger.warning(f"Image size  is not divisible by 64 ({w}x{h}). Resizing it to ({width}x{height})!")

        return np.expand_dims(np.array(image).transpose((2, 0, 1)) / 127.5 - 1, axis=0).astype(np.float32)


def write_video(vid: np.ndarray, video_path: str, fps: int = 25, codec: str = "mp4v"):
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*codec), fps, (vid.shape[2], vid.shape[1]))
    for frame in vid:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    logger.info(f"Generated video saved to {video_path}")


def main(args):
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    video_path = output_path / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".mp4")

    # set ms context
    ms.context.set_context(mode=int(args.mode), device_target="Ascend")
    seed_everything(args.seed)

    # load image
    image = prepare_image(args.image.absolute)

    # instantiate the model and generate a video
    pipeline = SVDInferPipeline(**args.SVD)
    logger.info("Starting video generation, this may take a while...")
    vid = pipeline(Tensor(image))

    write_video(vid.numpy(), str(video_path), fps=args.SVD.fps)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument(
        "--mode",
        type=str,
        default="1",
        choices=["0", "1"],
        help="MindSpore execution mode: Graph mode[0] or Pynative mode[1]",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_class_arguments(SVDInferPipeline, "SVD")
    parser.add_argument("--image", type=Path_fr, required=True)
    parser.add_argument("--output_dir", type=Path_dcc, default="./output")

    cfg = parser.parse_args()
    main(cfg)
