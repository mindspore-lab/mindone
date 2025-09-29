import logging
import os
import sys
from pathlib import Path

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from tqdm import tqdm

from mindspore import Tensor, tensor

from mindone.data import create_dataloader
from mindone.utils import init_env, set_logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from opensora.datasets.video_dataset_refactored import VideoDatasetRefactored
from opensora.models.hunyuan_vae import CausalVAE3D_HUNYUAN

logger = logging.getLogger(__name__)


def save_batch(out_dir: Path, paths: list[str], means: Tensor, logvars: Tensor) -> None:
    means, logvars = means.asnumpy().astype(np.float32), logvars.asnumpy().astype(np.float32)
    stds = np.exp(0.5 * np.clip(logvars, -30.0, 20.0))
    for path, mean, std in zip(paths, means, stds):
        path = (out_dir / path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)  # if subfolders in the path
        np.savez(path, latent_mean=mean, latent_std=std)


def main(args):
    _, rank_id, device_num = init_env(**args.env)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_logger("", output_dir=str(args.output_dir), rank=rank_id)

    dataset = VideoDatasetRefactored(
        **args.dataset, v2_pipeline=True, apply_transforms_dataset=True, output_columns=["video", "path"]
    )
    dataloader = create_dataloader(dataset, **args.dataloader, drop_remainder=False)
    dataset_size = dataloader.get_dataset_size()
    logger.info(f"Number of batches: {dataset_size}")

    # model initiate and weight loading
    logger.info("Initializing VAE...")
    model_ae = CausalVAE3D_HUNYUAN(**args.ae).set_train(False)  # TODO: add DC-AE support
    del model_ae.decoder

    logger.info("Starting the latents embedding...")
    for batch in tqdm(dataloader.create_tuple_iterator(num_epochs=1, output_numpy=True), total=dataset_size):
        videos, paths = batch
        latent_mean, latent_logvar = model_ae._encode(  # FIXME: do not use protected method
            tensor(videos, dtype=model_ae.dtype)
        )
        save_batch(args.output_dir, paths.tolist(), latent_mean, latent_logvar)
    logger.info(f"Completed. Embeddings saved in {args.output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser(description="OpenSora v2 VAE Embedding.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_env, "env")
    parser.add_function_arguments(CausalVAE3D_HUNYUAN, "ae")
    parser.add_class_arguments(
        VideoDatasetRefactored,
        "dataset",
        skip={
            "text_emb_folder",
            "empty_text_emb",
            "text_drop_prob",
            "vae_latent_folder",
            "vae_downsample_rate",
            "vae_scale_factor",
            "vae_shift_factor",
            "frames_mask_generator",
            "latent_compress_func",
            "pre_patchify",
            "patch_size",
            "embed_dim",
            "num_heads",
            "max_target_size",
            "input_sq_size",
            "in_channels",
            "buckets",
            "apply_transforms_dataset",
            "tokenizer",
            "v2_pipeline",
            "output_columns",
        },
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
            "drop_remainder",
            "device_num",
            "rank_id",
        },
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_argument("--output_dir", type=Path, default="latents")

    cfg = parser.parse_args()
    main(cfg)
