import logging
import os
import sys
from typing import Dict, Tuple, Union

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

import mindspore.dataset as ds
from mindspore import GRAPH_MODE, get_context, nn, set_context, set_seed

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))
from hyvideo.acceleration import create_parallel_group
from hyvideo.dataset import ImageVideoDataset, bucket_split_function
from hyvideo.utils import EMA, init_model, resume_train_net
from hyvideo.utils.callbacks import ReduceLROnPlateauByStep
from hyvideo.vae import AutoencoderKLCausal3D, load_vae

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback
from mindone.trainers.zero import prepare_train_network
from mindone.utils import init_train_env, set_logger

logger = logging.getLogger(__name__)


def initialize_dataset(
    dataset_args, dataloader_args, device_num: int, shard_rank_id: int
) -> Tuple[Union[ds.BatchDataset, ds.BucketBatchByLengthDataset], int]:
    dataset = ImageVideoDataset(**dataset_args)
    transforms = (
        dataset.train_transforms(dataset_args.target_size) if not dataset_args.apply_transforms_dataset else None
    )

    dataloader_args = dataloader_args.as_dict()
    batch_size = dataloader_args.pop("batch_size")
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size if isinstance(batch_size, int) else 0,  # Turn off batching if using buckets
        transforms=transforms,
        device_num=device_num,
        rank_id=shard_rank_id,
        **dataloader_args,
    )
    if isinstance(batch_size, dict):  # if buckets are used
        hash_func, bucket_boundaries, bucket_batch_sizes = bucket_split_function(**batch_size)
        dataloader = dataloader.bucket_batch_by_length(
            ["video"],
            bucket_boundaries,
            bucket_batch_sizes,
            element_length_function=hash_func,
            drop_remainder=dataloader_args["drop_remainder"],
        )

    # Extract and print the shape and dtype of the first three samples from the dataset
    print("Dataset Sample Shapes and Dtypes:")
    for i in range(3):
        sample = dataset[i]
        print(f"Sample {i}:")
        for value in sample:
            print(f" shape={value.shape}, dtype={value.dtype}")

    # Extract and print the shape and dtype of the first three batches from the dataloader
    print("\nDataloader Batch Shapes and Dtypes:")
    batch_count = 0
    for batch in dataloader:
        if batch_count >= 3:
            break
        print(f"Batch {batch_count}:")
        for key, value in batch.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        batch_count += 1

    return dataloader, len(dataset)


def main(args):
    # 1. init env
    args.train.output_path = os.path.abspath(args.train.output_path)
    os.makedirs(args.train.output_path, exist_ok=True)
    device_id, rank_id, device_num = init_train_env(**args.env)
    mode = get_context("mode")  # `init_train_env()` may change the mode during debugging

    # if bucketing is used in Graph mode, activate dynamic mode
    if mode == GRAPH_MODE and isinstance(args.dataloader.batch_size, dict):
        set_context(graph_kernel_flags="--disable_packet_ops=Reshape")

    # 1.1 init model parallel
    shard_rank_id = rank_id
    if args.train.sequence_parallel.shards > 1:
        create_parallel_group(**args.train.sequence_parallel)
        device_num = device_num // args.train.sequence_parallel.shards
        shard_rank_id = rank_id // args.train.sequence_parallel.shards

    # FIXME: Improve seed setting
    set_seed(args.env.seed + shard_rank_id)  # set different seeds per NPU for sampling different timesteps
    ds.set_seed(args.env.seed)  # keep MS.dataset's seed consistent as datasets first shuffled and then distributed

    set_logger("", output_dir=args.train.output_path, rank=rank_id)

    # instantiate classes only after initializing training environment
    initializer = parser.instantiate_classes(cfg)

    # 2. model initialize and weight loading
    # 2.1 vae
    if not args.dataset.vae_latent_folder or (
        args.valid.dataset and not args.valid.dataset.init_args.vae_latent_folder
    ):
        logger.info("Initializing vae...")
        vae, _, s_ratio, t_ratio = load_vae(
            args.vae.vae_type,
            logger=logger,
            vae_precision=args.vae.vae_precision,
        )
        if args.vae.vae_tiling:
            vae.enable_tiling()

    else:
        logger.info("vae latent folder provided. Skipping vae initialization.")
        vae = None

    # 4. build train & val datasets
    if args.train.sequence_parallel.shards > 1:
        logger.info(
            f"Initializing the dataloader: assigning shard ID {shard_rank_id} out of {device_num} total shards."
        )
    dataloader, dataset_len = initialize_dataset(args.dataset, args.dataloader, device_num, shard_rank_id)

    if args.valid.dataset is not None:
        val_dataloader, val_dataloader_len = initialize_dataset(
            args.valid.dataset.init_args, args.valid.dataloader, device_num, shard_rank_id
        )


if __name__ == "__main__":
    parser = ArgumentParser(description="Hunyuan Video training script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_train_env, "env")
    parser.add_function_arguments(init_model, "model", skip={"resume"})
    parser.add_class_arguments(AutoencoderKLCausal3D, "vae", instantiate=False)
    parser.add_class_arguments(
        ImageVideoDataset, "dataset", skip={"frames_mask_generator", "t_compress_func"}, instantiate=False
    )
    parser.add_function_arguments(
        create_dataloader,
        "dataloader",
        skip={"dataset", "batch_size", "transforms", "batch_transforms", "device_num", "rank_id"},
    )
    parser.add_argument(  # FIXME: support bucketing
        "--dataloader.batch_size", default=1, type=Union[int, Dict[str, int]], help="Number of samples per batch"
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_function_arguments(create_parallel_group, "train.sequence_parallel")
    parser.add_function_arguments(create_scheduler, "train.lr_scheduler", skip={"steps_per_epoch", "num_epochs"})
    parser.add_class_arguments(
        ReduceLROnPlateauByStep, "train.lr_reduce_on_plateau", skip={"optimizer"}, instantiate=False
    )
    parser.add_function_arguments(create_optimizer, "train.optimizer", skip={"params", "lr"})
    parser.add_subclass_arguments(
        nn.Cell,
        "train.loss_scaler",
        fail_untyped=False,  # no typing in mindspore
        help="mindspore.nn.FixedLossScaleUpdateCell or mindspore.nn.DynamicLossScaleUpdateCell",
    )
    parser.add_function_arguments(
        prepare_train_network, "train.settings", skip={"network", "optimizer", "scale_sense", "ema"}
    )
    parser.add_subclass_arguments(EMA, "train.ema", skip={"network"}, required=False, instantiate=False)
    parser.add_function_arguments(resume_train_net, "train", skip={"train_net"})
    parser.add_argument(
        "--train.output_path",
        default="output/",
        type=path_type("dcc"),  # path to a directory that can be created if it does not exist
        help="Output directory to save training results.",
    )
    parser.add_argument("--train.steps", default=100, type=int, help="Number of steps to train. Default: 100.")
    parser.link_arguments("train.steps", "train.lr_scheduler.total_steps", apply_on="parse")
    parser.add_class_arguments(
        EvalSaveCallback,
        "train.save",
        skip={
            "network",
            "rank_id",
            "shard_rank_id",
            "ckpt_save_dir",
            "output_dir",
            "ema",
            "start_epoch",
            "model_name",
            "step_mode",
            "use_step_unit",
            "train_steps",
            "resume_prefix_blacklist",
        },
        instantiate=False,
    )

    # validation
    val_group = parser.add_argument_group("Validation")
    val_group.add_argument(
        "valid.sampling_steps", type=int, default=10, help="Number of sampling steps for validation."
    )
    val_group.add_argument("valid.frequency", type=int, default=1, help="Frequency of validation in steps.")
    val_group.add_subclass_arguments(
        ImageVideoDataset,
        "valid.dataset",
        skip={"frames_mask_generator", "t_compress_func"},
        instantiate=False,
        required=False,
    )
    val_group.add_function_arguments(
        create_dataloader, "valid.dataloader", skip={"dataset", "transforms", "device_num", "rank_id"}
    )
    parser.link_arguments("env.debug", "valid.dataloader.debug", apply_on="parse")

    cfg = parser.parse_args()
    main(cfg)
