import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Union

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

import mindspore.dataset as ds
from mindspore import GRAPH_MODE, get_context, nn, set_context, set_seed

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, "../.."))
from hyvideo.acceleration import create_parallel_group
from hyvideo.dataset import ImageVideoDataset, bucket_split_function
from hyvideo.utils import EMA, init_model, resume_train_net
from hyvideo.utils.callbacks import ReduceLROnPlateauByStep
from hyvideo.utils.data_utils import align_to
from hyvideo.vae import load_vae

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback
from mindone.trainers.zero import prepare_train_network
from mindone.utils import init_env, set_logger
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def initialize_dataset(
    dataset_args, dataloader_args, device_num: int, shard_rank_id: int
) -> Tuple[Union[ds.BatchDataset, ds.BucketBatchByLengthDataset], int]:
    dataset = ImageVideoDataset(**dataset_args)

    dataloader_args = dataloader_args.as_dict()
    batch_size = dataloader_args.pop("batch_size")
    dataloader = create_dataloader(
        dataset,
        batch_size=batch_size if isinstance(batch_size, int) else 0,  # Turn off batching if using buckets
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
    return dataloader, len(dataset)


def main(args):
    # 1. init env
    args.train.output_path = os.path.abspath(args.train.output_path)
    os.makedirs(args.train.output_path, exist_ok=True)
    device_id, rank_id, device_num = init_env(**args.env)
    set_random_seed(getattr(args.env, "seed", 42))
    mode = get_context("mode")

    # if bucketing is used in Graph mode, activate dynamic mode
    if mode == GRAPH_MODE and isinstance(args.dataloader.batch_size, dict):
        set_context(graph_kernel_flags="--disable_packet_ops=Reshape")
    # if graph mode and vae tiling is ON, uise dfs exec order
    if mode == GRAPH_MODE and args.vae.tiling:
        set_context(exec_order="dfs")

    # 1.1 init model parallel
    shard_rank_id = rank_id
    if args.train.sequence_parallel.shards > 1:
        create_parallel_group(**args.train.sequence_parallel)
        device_num = device_num // args.train.sequence_parallel.shards
        shard_rank_id = rank_id // args.train.sequence_parallel.shards

    # set different seeds per NPU for sampling different timesteps, but if sp is enabled, force the timestep to be the same as rank_0
    set_seed(args.env.seed + shard_rank_id)
    ds.set_seed(args.env.seed)  # keep MS.dataset's seed consistent as datasets first shuffled and then distributed

    set_logger("", output_dir=args.train.output_path, rank=rank_id)

    sample_n_frames = args.dataset.sample_n_frames
    # size verification: num_frames -1 should be a multiple of 4, height and width should be a multiple of 16
    if (sample_n_frames - 1) % 4 != 0:
        raise ValueError(f"`sample_n_frames - 1` must be a multiple of 4, got {sample_n_frames}")
    if not isinstance(args.dataset.target_size, (list, tuple)):
        args.dataset.target_size = [args.dataset.target_size, args.dataset.target_size]
    height, width = args.dataset.target_size
    target_height = align_to(height, 16)
    target_width = align_to(width, 16)
    if target_height != height or target_width != width:
        logger.warning(
            f"The target size {height}x{width} is not a multiple of 16, "
            f"so it will be aligned to {target_height}x{target_width}."
        )
        args.dataset.target_size = [target_height, target_width]

    # 4. build train & val datasets
    if args.train.sequence_parallel.shards > 1:
        logger.info(
            f"Initializing the dataloader: assigning shard ID {shard_rank_id} out of {device_num} total shards."
        )
    dataloader, dataset_len = initialize_dataset(args.dataset, args.dataloader, device_num, shard_rank_id)
    logger.info(f"Num train batches: {dataloader.get_dataset_size()}")
    save_path_prefix = f"./rank{rank_id}_batch"
    batch_count = 0
    max_batches_to_save = 5

    for batch in dataloader:
        if batch_count >= max_batches_to_save:
            break

        data = []
        for item in batch:
            item = item.float().asnumpy()
            data.append(item)

        save_path = f"{save_path_prefix}{batch_count}.npz"
        save_path = str(Path(save_path).absolute())

        np.savez(save_path, *data)

        batch_count += 1
        print(f"The batch {batch_count} data has been saved to {save_path}")


if __name__ == "__main__":
    parser = ArgumentParser(description="Hunyuan Video training script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_env, "env")
    parser.add_function_arguments(init_model, "model", skip={"resume"})
    parser.add_function_arguments(load_vae, "vae", skip={"logger"})
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
    parser.add_argument("--train.data_sink_mode", default=False, type=bool, help="Whether to turn on data sink mode.")
    parser.add_argument("--train.data_sink_size", default=-1, type=int, help="The data sink size when sink mode is ON.")
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
