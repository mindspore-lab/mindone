import logging
import os
import sys
from typing import Dict, Tuple, Union

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

import mindspore as ms
import mindspore.dataset as ds
from mindspore import GRAPH_MODE, Model, get_context, nn, set_context, set_seed

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))
from hyvideo.acceleration import create_parallel_group
from hyvideo.dataset import ImageVideoDataset, bucket_split_function
from hyvideo.diffusion.pipelines import DiffusionWithLoss
from hyvideo.diffusion.schedulers import RFlowEvalLoss, RFlowLossWrapper
from hyvideo.utils import EMA, init_model, resume_train_net
from hyvideo.utils.callbacks import PerfRecorderCallback, ReduceLROnPlateauByStep, ValidationCallback
from hyvideo.utils.data_utils import align_to
from hyvideo.vae import load_vae

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, StopAtStepCallback
from mindone.trainers.zero import prepare_train_network
from mindone.utils import count_params, init_train_env, set_logger

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
    device_id, rank_id, device_num = init_train_env(**args.env)
    mode = get_context("mode")  # `init_train_env()` may change the mode during debugging

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
            logger=logger,
            **args.vae,
        )
        # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
        vae_dtype = args.vae.precision

    else:
        logger.info("vae latent folder provided. Skipping vae initialization.")
        vae = None

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

    # 2.2 Llama 3
    logger.info("Transformer init")
    network = init_model(resume=args.train.resume_ckpt is not None, **args.model)
    model_dtype = args.model.factor_kwargs["dtype"]
    if network.guidance_embed:
        embed_cfg_scale = 6.0
    else:
        embed_cfg_scale = None

    # 2.3 LossWrapper
    rflow_loss_wrapper = RFlowLossWrapper(network)

    # 3. build training network
    latent_diffusion_with_loss = DiffusionWithLoss(
        rflow_loss_wrapper,
        vae,
        video_emb_cached=bool(args.dataset.vae_latent_folder),
        embedded_guidance_scale=embed_cfg_scale,
    )

    # 4. build train & val datasets
    if args.train.sequence_parallel.shards > 1:
        logger.info(
            f"Initializing the dataloader: assigning shard ID {shard_rank_id} out of {device_num} total shards."
        )
    dataloader, dataset_len = initialize_dataset(args.dataset, args.dataloader, device_num, shard_rank_id)

    eval_diffusion_with_loss, val_dataloader = None, None
    if args.valid.dataset is not None:
        val_dataloader, _ = initialize_dataset(
            args.valid.dataset.init_args, args.valid.dataloader, device_num, shard_rank_id
        )
        eval_rflow_loss = RFlowEvalLoss(rflow_loss_wrapper, num_sampling_steps=args.valid.sampling_steps)
        eval_diffusion_with_loss = DiffusionWithLoss(
            eval_rflow_loss,
            vae,
            video_emb_cached=bool(args.valid.dataset.init_args.vae_latent_folder),
            embedded_guidance_scale=embed_cfg_scale,
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
    if args.train.resume_ckpt is not None:
        start_epoch, global_step = resume_train_net(net_with_grads, resume_ckpt=os.path.abspath(args.train.resume_ckpt))

    # TODO: validation graph?
    # if bucketing is used in Graph mode, activate dynamic inputs
    if mode == GRAPH_MODE and isinstance(args.dataloader.batch_size, dict):
        _bs = ms.Symbol(unique=True)
        video = ms.Tensor(shape=[_bs, 3, None, None, None], dtype=ms.float32)  # (b, c, f, h, w)
        text_embed_cache = args.dataset.text_emb_folder is not None
        text_tokens = (
            ms.Tensor(shape=[_bs, None, None], dtype=ms.float32)
            if text_embed_cache
            else ms.Tensor(shape=[_bs, None], dtype=ms.float32)
        )
        encoder_attention_mask = ms.Tensor(shape=[_bs, None], dtype=ms.uint8)

        text_tokens_2 = (
            ms.Tensor(shape=[_bs, None], dtype=ms.float32)  # pooled hidden states
            if text_embed_cache
            else ms.Tensor(shape=[_bs, None], dtype=ms.float32)
        )
        encoder_attention_mask_2 = ms.Tensor(shape=[_bs, None], dtype=ms.uint8)
        net_with_grads.set_inputs(video, text_tokens, encoder_attention_mask, text_tokens_2, encoder_attention_mask_2)
        logger.info("Dynamic inputs are initialized for training!")

    model = Model(net_with_grads)

    # 5.4 callbacks
    callbacks = [OverflowMonitor()]
    if val_dataloader is not None:
        callbacks.append(
            ValidationCallback(
                network=eval_diffusion_with_loss,
                dataset=val_dataloader,
                alpha_smooth=0.01,  # FIXME
                valid_frequency=args.valid.frequency,
                ema=ema,
            )
        )
        if args.train.lr_reduce_on_plateau is not None:
            callbacks.append(
                ReduceLROnPlateauByStep(optimizer, **args.train.lr_reduce_on_plateau),
            )

    if args.train.settings.zero_stage == 3 or rank_id == 0:
        ckpt_save_dir = (
            os.path.join(args.train.output_path, f"rank_{rank_id}/ckpt")
            if args.train.settings.zero_stage == 3
            else os.path.join(args.train.output_path, "ckpt")
        )
        save_kwargs = args.train.save.as_dict()
        log_interval = save_kwargs.get("log_interval", 1)
        if args.train.data_sink_mode:
            if args.train.data_sink_size == -1:
                sink_size = len(dataloader)
            else:
                sink_size = args.train.data_sink_size
            new_log_interval = sink_size * log_interval
            if new_log_interval != log_interval:
                logger.info(
                    f"Because of data sink mode ON and sink size {sink_size}, log_interval is changed from {log_interval} to {new_log_interval}"
                )
            log_interval = new_log_interval
        save_kwargs["log_interval"] = log_interval

        callbacks.append(
            EvalSaveCallback(
                network=latent_diffusion_with_loss.network,
                model_name=args.model.name.replace("/", "-"),
                rank_id=0 if args.train.settings.zero_stage == 3 else rank_id,  # ZeRO-3 shards across all ranks
                ckpt_save_dir=ckpt_save_dir,
                ema=ema,
                step_mode=True,
                use_step_unit=True,
                start_epoch=start_epoch,
                resume_prefix_blacklist=("vae.", "swap."),
                train_steps=args.train.steps,
                **save_kwargs,
            )
        )

    if rank_id == 0:
        callbacks.append(
            PerfRecorderCallback(
                args.train.output_path, file_name="result_val.log", metric_names=["eval_loss", "eval_loss_smoothed"]
            )
        )

    callbacks.append(StopAtStepCallback(train_steps=args.train.steps, global_step=global_step))

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
                f"Data sink mode (sink size): {args.train.data_sink_mode} ({args.train.data_sink_size})",
                f"Number of samples: {dataset_len}",
                f"Model name: {args.model.name}",
                f"Model dtype: {model_dtype}",
                f"vae dtype: {vae_dtype}",
                f"Num params: {num_params:,} (network: {num_params_network:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Learning rate: {args.train.lr_scheduler.lr:.0e}",
                f"Batch size: {args.dataloader.batch_size}",
                f"Image size: {args.dataset.target_size}",
                f"Frames: {args.dataset.sample_n_frames}",
                f"Weight decay: {args.train.optimizer.weight_decay}",
                f"Grad accumulation steps: {args.train.settings.gradient_accumulation_steps}",
                f"Number of training steps: {args.train.steps}",
                f"Loss scaler: {args.train.loss_scaler.class_path}",
                f"Init loss scale: {args.train.loss_scaler.init_args.loss_scale_value}",
                f"Grad clipping: {args.train.settings.clip_grad}",
                f"Max grad norm: {args.train.settings.clip_norm}",
                f"EMA: {ema is not None}",
                f"Attention mode: {args.model.factor_kwargs['attn_mode']}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)
        parser.save(args, args.train.output_path + "/config.yaml", format="yaml", overwrite=True)

    # 6. train
    logger.info("Start training...")
    # train() uses epochs, so the training will be terminated by the StopAtStepCallback
    model.train(
        args.train.steps,
        dataloader,
        callbacks=callbacks,
        initial_epoch=start_epoch,
        dataset_sink_mode=args.train.data_sink_mode,
        sink_size=args.train.data_sink_size,
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
