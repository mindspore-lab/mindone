import logging
import os
import sys
from typing import Optional, Tuple, Union

from jsonargparse import ActionConfigFile, ArgumentParser

import mindspore.dataset as mds
from mindspore import GRAPH_MODE, Model, Symbol
from mindspore import dtype as mstype
from mindspore import get_context, nn, tensor

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, StopAtStepCallback
from mindone.trainers.zero import prepare_train_network
from mindone.utils import count_params, init_env, set_logger

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from opensora.acceleration.parallel_states import create_parallel_group
from opensora.datasets.bucket import bucket_split_function
from opensora.datasets.bucket_v2 import Bucket
from opensora.datasets.video_dataset_refactored import VideoDatasetRefactored
from opensora.models.hunyuan_vae.autoencoder_kl_causal_3d import CausalVAE3D_HUNYUAN
from opensora.models.mmdit import Flux
from opensora.pipelines.train_pipeline_v2 import DiffusionWithLoss
from opensora.utils.callbacks import PerfRecorderCallback
from opensora.utils.ema import EMA
from opensora.utils.saving import TrainingSavingOptions
from opensora.utils.training import TrainingOptions

logger = logging.getLogger(__name__)


def initialize_dataset(
    dataset_args,
    dataloader_args,
    model,
    vae,
    bucket_config: Optional[dict] = None,
    validation: bool = False,
    device_num: int = 1,
    rank_id: int = 0,
) -> Tuple[Union[mds.BatchDataset, mds.BucketBatchByLengthDataset], int]:
    if validation:
        pass
        # all_buckets, individual_buckets = None, [None]
        # if bucket_config is not None:
        #     all_buckets = Bucket(bucket_config)
        #     # Build a new bucket for each resolution and number of frames for the validation stage
        #     individual_buckets = [
        #         Bucket({res: {num_frames: [1.0, bucket_config[res][num_frames][1]]}})
        #         for res in bucket_config.keys()
        #         for num_frames in bucket_config[res].keys()
        #     ]
    else:
        all_buckets = Bucket(**bucket_config.init_args) if bucket_config is not None else None
        individual_buckets = [all_buckets]

    datasets = [
        VideoDatasetRefactored(
            **dataset_args,
            latent_compress_func=vae.get_latent_size if vae is not None else None,
            buckets=buckets,
            patch_size=(1, model.patch_size, model.patch_size),
        )
        for buckets in individual_buckets
    ]

    num_src_samples = sum([len(ds) for ds in datasets])

    dataloader_args = dataloader_args.as_dict()
    batch_size = dataloader_args.pop("batch_size")
    dataloaders = [
        create_dataloader(
            dataset,
            batch_size=batch_size if all_buckets is None else 0,  # Turn off batching if using buckets
            **dataloader_args,
            rank_id=rank_id,
            device_num=device_num,
        )
        for dataset in datasets
    ]
    dataloader = mds.ConcatDataset(dataloaders) if len(dataloaders) > 1 else dataloaders[0]

    if all_buckets is not None:
        hash_func, bucket_boundaries, bucket_batch_sizes = bucket_split_function(all_buckets, v2=True)
        dataloader = dataloader.bucket_batch_by_length(
            ["video"],
            bucket_boundaries,
            bucket_batch_sizes,
            element_length_function=hash_func,
            drop_remainder=False,
        )
        dataloader.dataset_size = 1  # prevent MS from iterating over the dataset once at the beginning
    return dataloader, num_src_samples


def main(args):
    # 1. init env
    _, rank_id, device_num = init_env(**args.env)
    mode = get_context("mode")  # `init_env()` may change the mode during debugging

    saving_options = TrainingSavingOptions(**args.save)
    os.makedirs(saving_options.output_path, exist_ok=True)
    set_logger("", output_dir=saving_options.output_path, rank=rank_id)

    # 1.1 init model parallel
    shard_rank_id = rank_id
    if args.train.sequence_parallel.shards > 1:
        create_parallel_group(**args.train.sequence_parallel)
        device_num = device_num // args.train.sequence_parallel.shards
        shard_rank_id = rank_id // args.train.sequence_parallel.shards

    # instantiate classes only after initializing the training environment
    initializer = parser.instantiate_classes(cfg)

    if args.dataset.target_size and args.bucket_config:
        logger.info("Image size is provided, bucket configuration will be ignored.")
        args.bucket_config = None

    # 2. model initialize and weight loading
    # 2.1 VAE
    if not args.dataset.vae_latent_folder:
        logger.info("Initializing VAE...")
        vae = CausalVAE3D_HUNYUAN(**args.ae).set_train(False)  # TODO: add DC-AE support
        del vae.decoder
        for param in vae.get_parameters():  # turn grads off
            param.requires_grad = False
    else:
        logger.info("VAE latent folder provided. Skipping VAE initialization.")
        vae = None

    # 2.2 MMDiT
    logger.info("Initializing transformer...")
    model_name = "OpenSora 2.0"
    network = Flux(**args.model)
    # 3. build training network
    condition_config = initializer.train.pipeline.pop("condition_config")  # do it properly
    latent_diffusion_with_loss = DiffusionWithLoss(
        network,
        vae,
        patch_size=(1, network.patch_size, network.patch_size),
        condition_config=condition_config,
        **initializer.train.pipeline,
    )

    # 4. build train & val datasets
    if args.train.sequence_parallel.shards > 1:
        logger.info(
            f"Initializing the dataloader: assigning shard ID {shard_rank_id} out of {device_num} total shards."
        )
    dataloader, dataset_len = initialize_dataset(
        args.dataset, args.dataloader, network, vae, args.bucket_config, device_num=device_num, rank_id=shard_rank_id
    )

    # 5. build training utils: lr, optim, callbacks, trainer
    # 5.1 LR
    lr = create_scheduler(steps_per_epoch=0, **args.train.lr_scheduler)

    # 5.2 optimizer
    optimizer = create_optimizer(latent_diffusion_with_loss.trainable_params(), lr=lr, **args.train.optimizer)

    # 5.3 trainer (standalone and distributed)
    ema = EMA(latent_diffusion_with_loss.network, **args.train.ema.init_args) if args.train.ema else None
    loss_scaler = initializer.train.loss_scaler
    net_with_grads = prepare_train_network(
        latent_diffusion_with_loss, optimizer=optimizer, scale_sense=loss_scaler, ema=ema, **args.train.settings
    )

    start_epoch, global_step = 0, 0
    # TODO: train resume
    # if args.train.resume_ckpt is not None:
    # start_epoch, global_step = resume_train_net(net_with_grads, resume_ckpt=os.path.abspath(args.train.resume_ckpt))

    # Activate dynamic graph mode because bucketing is used
    if mode == GRAPH_MODE:
        if args.train.sequence_parallel.shards <= 1:
            bs = Symbol(unique=True)
            if args.dataset.vae_latent_folder is not None:
                video = tensor(shape=[bs, None, 64], dtype=mstype.float32)
            else:
                video = tensor(shape=[bs, 3, None, None, None], dtype=mstype.float32)
            img_ids = tensor(shape=[bs, None, 3], dtype=mstype.int32)
            text_embed = tensor(shape=[bs, 512, 4096], dtype=mstype.float32)
            txt_ids = tensor(shape=[bs, 512, 3], dtype=mstype.int32)
            y_vec = tensor(shape=[bs, 768], dtype=mstype.float32)
            shift_alpha = tensor(shape=[bs], dtype=mstype.float32)
            net_with_grads.set_inputs(video, img_ids, text_embed, txt_ids, y_vec, shift_alpha)
            logger.info("Dynamic inputs are initialized for bucket config training in Graph mode.")
        elif args.train.sequence_parallel.shards > 1:
            logger.warning(
                "Dynamic shape is not supported with sequence parallelism. The graph will be re-compiled for each new shape."
            )

    model = Model(net_with_grads)

    # 5.4 callbacks
    callbacks = [OverflowMonitor()]
    if args.train.settings.zero_stage == 3 or rank_id == 0:
        callbacks.append(
            EvalSaveCallback(
                network=latent_diffusion_with_loss.network,
                model_name=model_name,
                rank_id=rank_id,
                ckpt_save_dir=os.path.join(saving_options.output_path, "ckpt"),
                ema=ema,
                step_mode=True,
                use_step_unit=True,
                start_epoch=start_epoch,
                resume_prefix_blacklist=("vae.", "swap."),
                train_steps=args.train.options.steps,
                **args.train.save,
            )
        )

    if rank_id == 0:
        callbacks.append(
            PerfRecorderCallback(saving_options.output_path, file_name="result_val.log", metric_names=["val_loss"])
        )

    callbacks.append(StopAtStepCallback(train_steps=args.train.options.steps, global_step=global_step))

    # 5.5 print out key info and save config
    if rank_id == 0:
        num_params_vae, num_params_trainable_vae = count_params(vae) if vae is not None else (0, 0)
        num_params_network, num_params_trainable_network = count_params(network)
        num_params = num_params_vae + num_params_network
        num_params_trainable = num_params_trainable_vae + num_params_trainable_network
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {mode}",
                f"Debug mode: {args.env.debug}",
                f"JIT level: {args.env.jit_level}",
                f"Distributed mode: {args.env.distributed}",
                f"Data path: {args.dataset.csv_path}",
                f"Number of samples: {dataset_len}",
                f"Model name: {model_name}",
                f"Model dtype: {args.model.dtype}",
                f"VAE dtype: {args.ae.dtype}",
                f"Num params: {num_params:,} (network: {num_params_network:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Learning rate: {args.train.lr_scheduler.lr:.0e}",
                f"Batch size: {args.dataloader.batch_size}",
                f"Image size: {args.dataset.target_size}",
                f"Frames: {args.dataset.sample_n_frames}",
                f"Weight decay: {args.train.optimizer.weight_decay}",
                f"Grad accumulation steps: {args.train.settings.gradient_accumulation_steps}",
                f"Number of training steps: {args.train.options.steps}",
                f"Loss scaler: {args.train.loss_scaler.class_path}",
                f"Init loss scale: {args.train.loss_scaler.init_args.loss_scale_value}",
                f"Grad clipping: {args.train.settings.clip_grad}",
                f"Max grad norm: {args.train.settings.clip_norm}",
                f"EMA: {ema is not None}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)
        parser.save(args, saving_options.output_path + "/config.yaml", format="yaml", overwrite=True)

    # 6. train
    logger.info("Start training...")
    # train() uses epochs, so the training will be terminated by the StopAtStepCallback
    model.train(args.train.options.steps, dataloader, callbacks=callbacks, initial_epoch=start_epoch)


if __name__ == "__main__":
    parser = ArgumentParser(description="Movie Gen training script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_env, "env")
    parser.add_function_arguments(create_parallel_group, "train.sequence_parallel")
    parser.add_function_arguments(Flux, "model")
    parser.add_function_arguments(CausalVAE3D_HUNYUAN, "ae")
    parser.add_class_arguments(
        VideoDatasetRefactored,
        "dataset",
        skip={"buckets", "frames_mask_generator", "latent_compress_func", "patch_size"},
        instantiate=False,
    )
    parser.add_subclass_arguments(Bucket, "bucket_config", required=False, instantiate=False)
    parser.add_function_arguments(
        create_dataloader,
        "dataloader",
        skip={"dataset", "transforms", "batch_transforms", "device_num", "rank_id"},
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_class_arguments(
        DiffusionWithLoss, "train.pipeline", skip={"network", "vae", "patch_size"}, instantiate=False
    )
    parser.add_function_arguments(create_scheduler, "train.lr_scheduler", skip={"steps_per_epoch", "num_epochs"})
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
    parser.add_argument("save", type=TrainingSavingOptions)
    parser.add_argument("train.options", type=TrainingOptions)
    parser.link_arguments("train.options.steps", "train.lr_scheduler.total_steps", apply_on="parse")
    parser.add_class_arguments(
        EvalSaveCallback,
        "train.save",
        skip={
            "network",
            "rank_id",
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
    parser.link_arguments("train.settings.zero_stage", "train.save.zero_stage", apply_on="parse")

    cfg = parser.parse_args()
    main(cfg)
