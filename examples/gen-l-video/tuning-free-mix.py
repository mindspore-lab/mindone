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

from ldm.models.diffusion.plms import PLMSSampler
from ldm.modules.train.tools import set_random_seed
from depth_to_image import load_model_from_config, get_depth_estimator


def main(args):
    work_dir = os.path.dirname(os.path.abspath(__file__))

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=args.ms_mode, device_id=device_id)
    # ms.context.set_context(mode=args.ms_mode, device_target="Ascend", device_id=device_id, max_device_memory="30GB")

    set_random_seed(args.seed)

    # create model
    if not os.path.isabs(args.config):
        args.config = os.path.join(work_dir, args.config)

    config = OmegaConf.load(f"{args.config}")

    model = load_model_from_config(config, args.ckpt_path)
    noise_scheduler = PLMSSampler(model)
    depth_estimator = get_depth_estimator()


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

    args = parser.parse_args()

    # overwrite env var by parsed arg
    if args.ckpt_path is None:
        args.ckpt_path = "models/sd_v2_base-57526ee4.ckpt"

    if args.config is None:
        args.config = "configs/tuning-free-mix.yaml"

    main(args)
