import logging
import os
import re
import sys
from math import ceil

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

from mindspore import Model, amp, nn, set_seed
from mindspore.train.callback import TimeMonitor

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.append(mindone_lib_path)

from moviegen.dataset import ImageVideoDataset
from moviegen.parallel import create_parallel_group
from moviegen.pipelines import DiffusionWithLoss
from moviegen.schedulers import RFlowEvalLoss, RFlowLossWrapper
from moviegen.utils import EMA, MODEL_DTYPE, init_model
from moviegen.utils.callbacks import PerfRecorderCallback, ReduceLROnPlateauByStep, ValidationCallback

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, StopAtStepCallback
from mindone.trainers.zero import prepare_train_network
from mindone.utils import count_params, init_train_env, set_logger

# TODO: remove when VAE is added to the project
sys.path.append(os.path.join(__dir__, "../opensora_hpcai/"))
from opensora.models.vae.vae import OpenSoraVAE_V1_2

logger = logging.getLogger(__name__)


def main(args):
    # 1. init env
    args.train.output_path = args.train.output_path.absolute
    os.makedirs(args.train.output_path, exist_ok=True)
    device_id, rank_id, device_num = init_train_env(**args.env)

    # 1.1 init model parallel
    shard_rank_id = rank_id
    if (shards := args.train.model_parallel.model_parallel_shards) > 1:
        create_parallel_group(**args.train.model_parallel)
        device_num = device_num // shards
        shard_rank_id = rank_id // shards

    set_seed(args.env.seed + shard_rank_id)  # TODO: do it better
    set_logger("", output_dir=args.train.output_path, rank=rank_id)

    # instantiate classes only after initializing training environment
    initializer = parser.instantiate_classes(cfg)

    # 2. model initialize and weight loading
    # 2.1 VAE
    logger.info("vae init")
    # TODO: add support of training with latents
    vae_args = args.vae.as_dict()
    vae_dtype = vae_args.pop("dtype")
    vae = OpenSoraVAE_V1_2(**vae_args).set_train(False)
    if vae_dtype != "fp32":
        # FIXME: remove AMP and add custom dtype conversion support for better compatibility with PyNative
        amp.custom_mixed_precision(vae, black_list=amp.get_black_list() + [nn.GroupNorm], dtype=MODEL_DTYPE[vae_dtype])

    # 2.2 Llama 3
    network = init_model(in_channels=vae.out_channels, **args.model)
    # 2.3 LossWrapper
    rflow_loss_wrapper = RFlowLossWrapper(network)

    # 3. build training network
    latent_diffusion_with_loss = DiffusionWithLoss(rflow_loss_wrapper, vae)

    # 4. build dataset
    dataset = ImageVideoDataset(**args.dataset)
    transforms = (
        dataset.train_transforms(args.dataset.target_size) if not args.dataset.apply_transforms_dataset else None
    )
    dataloader = create_dataloader(
        dataset, transforms=transforms, device_num=device_num, rank_id=shard_rank_id, **args.dataloader
    )

    eval_diffusion_with_loss, val_dataloader = None, None
    if args.valid.dataset is not None:
        val_dataset = ImageVideoDataset(**args.valid.dataset.init_args)
        transforms = None
        if not args.valid.dataset.init_args.apply_transforms_dataset:
            transforms = val_dataset.train_transforms(args.valid.dataset.init_args.target_size)
        val_dataloader = create_dataloader(
            val_dataset, transforms=transforms, device_num=device_num, rank_id=shard_rank_id, **args.valid.dataloader
        )
        eval_rflow_loss = RFlowEvalLoss(rflow_loss_wrapper, num_sampling_steps=args.valid.sampling_steps)
        eval_diffusion_with_loss = DiffusionWithLoss(eval_rflow_loss, vae)

    # 5. build training utils: lr, optim, callbacks, trainer
    # 5.1 LR
    epochs = ceil(args.train.steps / dataloader.get_dataset_size())
    lr = create_scheduler(steps_per_epoch=0, **args.train.lr_scheduler)

    # 5.2 optimizer
    optimizer = create_optimizer(latent_diffusion_with_loss.trainable_params(), lr=lr, **args.train.optimizer)

    # 5.3 trainer (standalone and distributed)
    ema = EMA(latent_diffusion_with_loss.network, **args.train.ema.init_args) if args.train.ema else None
    loss_scaler = initializer.train.loss_scaler
    net_with_grads = prepare_train_network(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        ema=ema,
        need_reduce=tuple(bool(re.search(r"layers\.(\d+)\.mlp", param.name)) for param in optimizer.parameters),
        **args.train.settings,
    )

    model = Model(net_with_grads)

    # 5.4 callbacks
    callbacks = [OverflowMonitor()]
    if val_dataloader is not None:
        callbacks.extend(
            [
                ValidationCallback(
                    network=eval_diffusion_with_loss,
                    dataset=val_dataloader,
                    alpha_smooth=0.01,  # FIXME
                    valid_frequency=args.valid.frequency,
                    ema=ema,
                ),
                ReduceLROnPlateauByStep(optimizer, **args.train.lr_reduce_on_plateau),
            ]
        )

    if rank_id == 0:
        callbacks.extend(
            [
                TimeMonitor(args.train.save.log_interval),
                EvalSaveCallback(
                    network=latent_diffusion_with_loss.network,
                    model_name=args.model.name,
                    rank_id=rank_id,
                    ckpt_save_dir=os.path.join(args.train.output_path, "ckpt"),
                    ema=ema,
                    step_mode=True,
                    use_step_unit=True,
                    train_steps=args.train.steps,
                    **args.train.save,
                ),
                PerfRecorderCallback(
                    args.train.output_path, file_name="result_val.log", metric_names=["eval_loss", "eval_loss_smoothed"]
                ),
            ]
        )

    callbacks.append(StopAtStepCallback(train_steps=args.train.steps))

    # 5.5 print out key info and save config
    if rank_id == 0:
        num_params_vae, num_params_trainable_vae = count_params(vae)
        num_params_network, num_params_trainable_network = count_params(network)
        num_params = num_params_vae + num_params_network
        num_params_trainable = num_params_trainable_vae + num_params_trainable_network
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.env.mode}",
                f"Debug mode: {args.env.debug}",
                f"JIT level: {args.env.jit_level}",
                f"Distributed mode: {args.env.distributed}",
                f"Data path: {args.dataset.csv_path}",
                f"Number of samples: {len(dataset)}",
                f"Model name: {args.model.name}",
                f"Model dtype: {args.model.dtype}",
                f"VAE dtype: {args.vae.dtype}",
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
                f"Enable flash attention: {args.model.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)
        parser.save(args, args.train.output_path + "/config.yaml", format="yaml", overwrite=True)

    # 6. train
    logger.info("Start training...")
    model.train(epochs, dataloader, callbacks=callbacks)


if __name__ == "__main__":
    parser = ArgumentParser(description="Movie Gen training script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_train_env, "env")
    parser.add_function_arguments(init_model, "model", skip={"in_channels"})
    parser.add_function_arguments(OpenSoraVAE_V1_2, "vae", fail_untyped=False)
    parser.add_argument(
        "--vae.dtype", default="fp32", type=str, choices=["fp32", "fp16", "bf16"], help="VAE model precision."
    )
    parser.add_class_arguments(
        ImageVideoDataset, "dataset", skip={"frames_mask_generator", "t_compress_func"}, instantiate=False
    )
    parser.add_function_arguments(
        create_dataloader, "dataloader", skip={"dataset", "transforms", "device_num", "rank_id"}
    )
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_function_arguments(create_parallel_group, "train.model_parallel")
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
        prepare_train_network, "train.settings", skip={"network", "optimizer", "scale_sense", "ema", "need_reduce"}
    )
    parser.add_subclass_arguments(EMA, "train.ema", skip={"network"}, required=False, instantiate=False)
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
            "ckpt_save_dir",
            "output_dir",
            "ema",
            "start_epoch",
            "model_name",
            "step_mode",
            "use_step_unit",
            "train_steps",
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
