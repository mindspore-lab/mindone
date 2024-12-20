import logging
import os
import sys
from pathlib import Path

import numpy as np
from jsonargparse import ArgumentParser
from jsonargparse.typing import path_type
from tqdm import tqdm

from mindspore import amp, get_context, nn

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.append(mindone_lib_path)

from mg.dataset.tae_dataset import VideoDataset
from mg.models.tae import TemporalAutoencoder
from mg.utils import MODEL_DTYPE, to_numpy

from mindone.data import create_dataloader
from mindone.utils import init_train_env, set_logger

logger = logging.getLogger(__name__)


def main(args):
    # 1. init env
    _, rank_id, device_num = init_train_env(**args.env)  # TODO: rename as train and infer are identical?
    mode = get_context("mode")  # `init_train_env()` may change the mode during debugging

    save_dir = Path(args.output_path.absolute)
    save_dir.mkdir(parents=True, exist_ok=True)
    set_logger(name="", output_dir=str(save_dir), rank=rank_id)

    # 2 build dataset
    dataset = VideoDataset(
        **args.data,
        sample_n_frames=10**5,  # read the full video, limitation of `albumentations` (i.e., `additional_targets`)
        output_columns=["video", "rel_path"],
    )
    dataloader = create_dataloader(
        dataset, drop_remainder=False, device_num=device_num, rank_id=rank_id, **args.dataloader
    )

    # 3. TAE initiate and weight loading
    logger.info("TAE init")
    tae_args = args.tae.as_dict()
    tae_dtype = tae_args.pop("dtype")
    tae = TemporalAutoencoder(**tae_args).set_train(False)
    if tae_dtype != "fp32":
        # FIXME: remove AMP and add custom dtype conversion support for better compatibility with PyNative
        amp.custom_mixed_precision(
            tae,
            black_list=amp.get_black_list() + [nn.GroupNorm, nn.AvgPool2d, nn.Upsample],
            dtype=MODEL_DTYPE[tae_dtype],
        )
    # TODO: add dynamic shape support

    # 4. print key info
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {mode}",
            f"Debug mode: {args.env.debug}",
            f"Num of batches: {dataloader.get_dataset_size()}",
            f"TAE dtype: {tae_dtype}",
            f"Image size: {args.data.size}",
            f"Crop size: {args.data.crop_size}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    for samples in tqdm(dataloader.create_tuple_iterator(num_epochs=1), total=dataloader.get_dataset_size()):
        _, mean, logvar = tae.encode(samples[0])
        mean, logvar = to_numpy(mean), to_numpy(logvar)
        std = np.exp(0.5 * np.clip(logvar, -30.0, 20.0))

        for m, s, path in zip(mean, std, samples[1].tolist()):
            out_path = save_dir / path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(out_path.with_suffix(".npz"), latent_mean=m, latent_std=s)
    logger.info(f"Completed. Latents saved in {save_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="TAE inference script.")
    parser.add_function_arguments(init_train_env, "env")
    tae_group = parser.add_argument_group("TAE parameters")
    tae_group.add_class_arguments(TemporalAutoencoder, "tae", instantiate=False)
    tae_group.add_argument(
        "--tae.dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="TAE model precision."
    )
    parser.add_class_arguments(
        VideoDataset,
        "data",
        skip={"random_crop", "flip", "sample_n_frames", "return_image", "output_columns"},
        instantiate=False,
    )
    parser.add_function_arguments(
        create_dataloader,
        "dataloader",
        skip={
            "dataset",
            "transforms",
            "batch_transforms",
            "project_columns",
            "shuffle",
            "num_workers",  # no transformations inside `.map()`
            "drop_remainder",
            "device_num",
            "rank_id",
            "enable_modelarts",
        },
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_argument(
        "--output_path",
        default="output/",
        type=path_type("dcc"),  # path to a directory that can be created if it does not exist
        help="Output directory to save training results.",
    )
    cfg = parser.parse_args()
    main(cfg)
