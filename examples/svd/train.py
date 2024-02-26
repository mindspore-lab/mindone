import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import Path_fr, path_type
from modules.encoders.modules import VideoPredictionEmbedderWithEncoder
from omegaconf import OmegaConf

from mindspore import Model, amp, float32, load_checkpoint, load_param_into_net, nn
from mindspore.train.callback import LossMonitor, TimeMonitor

sys.path.append("../../")  # FIXME: remove in future when mindone is ready for install
from mindone.data import BaseDataset, build_dataloader
from mindone.env import init_train_env
from mindone.utils import count_params, set_logger

sys.path.append("../stable_diffusion_xl")
from gm.helpers import create_model

sys.path.append("../stable_diffusion_v2")
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper

logging.basicConfig(level=logging.INFO)

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


def mixed_precision(network):
    from gm.modules.diffusionmodules.util import GroupNorm as GroupNorm3D

    from mindspore.nn import GroupNorm, SiLU

    black_list = amp.get_black_list() + [SiLU, GroupNorm, GroupNorm3D]
    return amp.custom_mixed_precision(network, black_list=black_list)


def main(args, initializer):
    # step 1: initialize environment
    device_id, rank_id, device_num = init_train_env(**args.environment)

    output_dir = Path(args.train.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = set_logger(name=__name__, output_dir=str(output_dir), rank=rank_id)

    # step 2: load SD model
    config = OmegaConf.load(args.svd_config.absolute)
    ldm_with_loss, _ = create_model(config, checkpoints=args.train.pretrained.absolute, freeze=False, amp_level="O0")
    ldm_with_loss.model.set_train(True)  # only unet

    temporal_param_names = ldm_with_loss.model.diffusion_model.get_temporal_param_names(prefix="model.diffusion_model.")
    # new_weights_map = ldm_with_loss.model.diffusion_model.get_weights_map(prefix="model.diffusion_model.")

    # if param_not_load:
    #     diff = set(param_not_load).difference(temporal_param_names)
    #     if not diff:
    #         logger.info("Temporal parameters are not loaded.")
    #     else:
    #         logger.warning(f"Failed to load parameters: {diff}")

    if args.train.temporal_only:
        for param in ldm_with_loss.trainable_params():
            if param.name not in temporal_param_names:
                param.requires_grad = False

    # Set mixed precision on certain modules only
    cells = ldm_with_loss.name_cells()
    for cell in cells:
        if not cells[cell] is ldm_with_loss.loss_fn and not (
            config.model.params.disable_first_stage_amp and cells[cell] is ldm_with_loss.first_stage_model
        ):
            setattr(ldm_with_loss, cell, mixed_precision(cells[cell]))
    for emb in ldm_with_loss.conditioner._backbone.embedders:
        if isinstance(emb, VideoPredictionEmbedderWithEncoder) and emb.disable_encoder_amp:
            emb.to_float(float32)

    # step 3: prepare train dataset and dataloader
    dataset = initializer.train.dataset
    train_dataloader = build_dataloader(
        dataset,
        transforms=dataset.train_transforms(),
        device_num=device_num,
        rank_id=rank_id,
        debug=args.environment.debug,
        enable_modelarts=args.environment.enable_modelarts,
        **args.train.dataloader,
    )

    # step 5: create optimizer and train the same way as regular SD
    # FIXME: set decay steps in the scheduler function
    args.train.scheduler.decay_steps = (
        args.train.epochs * train_dataloader.get_dataset_size() - args.train.scheduler.warmup_steps
    )
    lr = create_scheduler(
        steps_per_epoch=train_dataloader.get_dataset_size(), num_epochs=args.train.epochs, **args.train.scheduler
    )
    optimizer = build_optimizer(ldm_with_loss, lr=lr, **args.train.optimizer)

    loss_scaler = nn.DynamicLossScaleUpdateCell(**args.LossScale)

    net_with_grads = TrainOneStepWrapper(
        ldm_with_loss, optimizer=optimizer, scale_sense=loss_scaler, **args.train.settings
    )

    if not args.environment.debug and args.train.sink_size != -1:
        if train_dataloader.get_dataset_size() % args.train.sink_size:
            raise ValueError(
                f"Number of batches in dataset ({train_dataloader.get_dataset_size()})"
                f" must be divisible by sink_size ({args.train.sink_size})."
            )
        args.train.epochs = args.train.epochs * train_dataloader.get_dataset_size() // args.train.sink_size

    callbacks = [OverflowMonitor()]

    if rank_id == 0:
        ckpt_save_interval = (
            1 if args.train.sink_size == -1 else train_dataloader.get_dataset_size() // args.train.sink_size
        )
        callbacks.extend(
            [
                TimeMonitor(1),
                LossMonitor(),
                EvalSaveCallback(
                    network=ldm_with_loss,
                    model_name="svd",
                    rank_id=rank_id,
                    ckpt_save_dir=str(output_dir / "ckpt"),
                    ckpt_save_interval=ckpt_save_interval,
                    ckpt_save_policy="latest_k",
                    ckpt_max_keep=10,
                    record_lr=False,
                ),
            ]
        )

        # num_params_unet, _ = count_params(ldm_with_loss.model.diffusion_model)
        num_params_text_encoder, _ = count_params(ldm_with_loss.conditioner)
        num_params_vae, _ = count_params(ldm_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(ldm_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"Debugging: {args.environment.debug}",
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.environment.mode}",
                f"Distributed mode: {args.environment.distributed}",
                "Model: StableDiffusion v2.1",
                # f"Num params SD: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision SD: {args.train.amp_level}",
                f"Num epochs: {args.train.epochs} {f'(adjusted to sink size {args.train.sink_size})' if not args.environment.debug else ''}",
                f"Learning rate: {args.train.scheduler.lr}",
                f"Batch size: {args.train.dataloader.batch_size}",
                f"Number of frames: {args.train.dataset.init_args.frames}",
                f"Weight decay: {args.train.optimizer.weight_decay}",
                f"Grad accumulation steps: {args.train.settings.gradient_accumulation_steps}",
                f"Grad clipping: {args.train.settings.clip_grad}",
                f"Max grad norm: {args.train.settings.clip_norm}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        # shutil.copyfile(args.svd_config.absolute, output_dir / "sd_config.yaml")  # SD's parameters are not modified
        # ArgumentParser().save(args, output_dir / "svd_config.yaml", format="yaml", skip_check=True)

    model = Model(net_with_grads)
    model.train(
        args.train.epochs,
        train_dataloader,
        callbacks=callbacks,
        dataset_sink_mode=False,
        sink_size=args.train.sink_size,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(init_train_env, "environment")
    parser.add_argument("--train.epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--train.sink_size", type=int, default=-1, help="Number of steps in each data sinking.")
    parser.add_argument("--train.temporal_only", type=bool, default=True, help="Train temporal layers only.")
    parser.add_argument("--train.pretrained", type=Path_fr, required=True, help="Path to pretrained model.")
    parser.add_argument(
        "--train.amp_level", choices=["O0", "O1", "O2", "O3"], default="O2", help="Automatic Mixed Precision."
    )
    parser.add_argument(
        "--train.output_dir",
        type=Path_dcc,
        default="output/",
        help="Output directory for saving training results.",
    )
    parser.add_subclass_arguments(BaseDataset, "train.dataset")
    parser.add_function_arguments(
        build_dataloader,
        "train.dataloader",
        skip={"dataset", "transforms", "device_num", "rank_id", "debug", "enable_modelarts"},
    )
    parser.add_function_arguments(create_scheduler, "train.scheduler", skip={"steps_per_epoch", "num_epochs"})
    parser.add_function_arguments(build_optimizer, "train.optimizer", skip={"model", "lr"})
    parser.add_class_arguments(
        TrainOneStepWrapper, "train.settings", skip={"network", "optimizer", "scale_sense", "ema"}, instantiate=False
    )
    parser.add_argument(
        "--svd_config",
        type=Path_fr,
        default="configs/svd.yaml",
        help="Stable Video Diffusion model configuration.",
    )
    parser.add_class_arguments(nn.DynamicLossScaleUpdateCell, "LossScale", instantiate=False, fail_untyped=False)

    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)

    main(cfg, init)
