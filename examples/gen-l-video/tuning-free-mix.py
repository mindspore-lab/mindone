import argparse
import os
import sys

from omegaconf import OmegaConf

import mindspore as ms

workspace = os.path.dirname(os.path.abspath(__file__))
print("workspace:", workspace, flush=True)
sys.path.append(workspace)
print("workspace:", workspace + "/../stable_diffusion_v2", flush=True)
sys.path.append(workspace + "/../stable_diffusion_v2")

from conditions.depth import DepthEstimator
from depth_to_image import load_model_from_config
from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.train.tools import set_random_seed
from utils.download import download_checkpoint

_version_cfg = {
    "2.0": ("sd_v2_depth-186e18a0.ckpt", "v2-depth-inference.yaml", 512),
}
_URL_PREFIX = "https://download.mindspore.cn/toolkits/mindone/stable_diffusion"


def main(args):
    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=args.ms_mode, device_id=device_id)
    # ms.context.set_context(mode=args.ms_mode, device_target="Ascend", device_id=device_id, max_device_memory="30GB")

    set_random_seed(args.seed)

    config = OmegaConf.load(f"{args.config}")
    unet3d_config = OmegaConf.load(f"{args.unet3d_config}")

    model = load_model_from_config(config, args.ckpt_path)
    noise_scheduler = PLMSSampler(model)
    text_encoder = model.cond_stage_model
    tokenizer = text_encoder.tokenizer
    vae = model.first_stage_model
    unet = load_model_from_config(unet3d_config)
    depth_estimator = DepthEstimator(amp_level=args.depth_est_amp_level)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ms_mode",
        type=int,
        default=0,
        help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to config which constructs model.",
    )

    parser.add_argument(
        "--unet3d_config",
        type=str,
        default=None,
        help="path to config which constructs UNet3D.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="path to checkpoint of model",
    )

    parser.add_argument(
        "--depth_est_amp_level",
        type=str,
        default="O3",
        help="amp level for running depth estimator",
    )

    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.0",
        help="Stable diffusion version, wukong or 2.0",
    )

    args = parser.parse_args()

    # overwrite env var by parsed arg
    if args.ckpt_path is None:
        ckpt_name = _version_cfg[args.version][0]
        args.ckpt_path = "models/" + ckpt_name
        if not os.path.exists(args.ckpt_path):
            print(f"Start downloading checkpoint {ckpt_name} ...")
            download_checkpoint(os.path.join(_URL_PREFIX, ckpt_name), "models/")

    if args.config is None:
        args.config = "configs/tuning-free-mix.yaml"

    if args.unet3d_config is None:
        args.unet3d_config = "configs/unet3d.yaml"

    main(args)
