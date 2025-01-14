import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import Path_fr, path_type
from omegaconf import OmegaConf
from utils import mixed_precision

import mindspore as ms
from mindspore import Callback, Model, nn
from mindspore.train.callback import LossMonitor

sys.path.append("../../")  # FIXME: remove in future when mindone is ready for install
from modules.helpers import create_model

from mindone.data import BaseDataset, create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.utils import count_params, set_logger
from mindone.utils.env import init_train_env

sys.path.append("../stable_diffusion_v2")
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.trainer import TrainOneStepWrapper

logging.basicConfig(level=logging.INFO)

Path_dcc = path_type("dcc")  # path to a directory that can be created if it does not exist


class SetTrainCallback(Callback):
    # TODO: is it necessary?
    def on_train_begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params["network"].network.set_train(False)
        cb_params["network"].network.model.set_train(True)


def main(args, initializer):
    # step 1: initialize environment
    device_id, rank_id, device_num = init_train_env(**args.environment)
    if args.environment.mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": args.jit_level})
    output_dir = Path(args.train.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = set_logger(name=__name__, output_dir=str(output_dir), rank=rank_id)

    # step 2: load SVD model
    config = OmegaConf.load(args.svd_config.absolute)
    ldm_with_loss, _ = create_model(config, checkpoints=args.train.pretrained.absolute, freeze=False, amp_level="O0")

    temporal_param_names = ldm_with_loss.model.diffusion_model.get_temporal_param_names(prefix="model.diffusion_model.")
    if args.train.temporal_only:
        for param in ldm_with_loss.trainable_params():
            if param.name not in temporal_param_names:
                param.requires_grad = False

    if args.train.amp_level == "O2":  # Set mixed precision for certain modules only
        mixed_precision(ldm_with_loss)

    # step 3: prepare train dataset and dataloader
    dataset = initializer.train.dataset
    train_dataloader = create_dataloader(
        dataset,
        transforms=dataset.train_transforms(),
        device_num=device_num,
        rank_id=rank_id,
        debug=args.environment.debug,
        enable_modelarts=args.environment.enable_modelarts,
        **args.train.dataloader,
    )

    # step 4: create optimizer and train the same way as regular SD
    # FIXME: set decay steps in the scheduler function
    args.train.scheduler.decay_steps = (
        args.train.epochs * train_dataloader.get_dataset_size() - args.train.scheduler.warmup_steps
    )
    lr = create_scheduler(
        steps_per_epoch=train_dataloader.get_dataset_size(), num_epochs=args.train.epochs, **args.train.scheduler
    )
    optimizer = create_optimizer(ldm_with_loss.trainable_params(), lr=lr, **args.train.optimizer)

    loss_scaler = nn.DynamicLossScaleUpdateCell(**args.LossScale)

    net_with_grads = TrainOneStepWrapper(
        ldm_with_loss, optimizer=optimizer, scale_sense=loss_scaler, **args.train.settings
    )

    callbacks = [OverflowMonitor(), SetTrainCallback()]

    if rank_id == 0:
        callbacks.extend(
            [
                LossMonitor(),
                EvalSaveCallback(
                    network=ldm_with_loss,
                    model_name="svd",
                    rank_id=rank_id,
                    ckpt_save_dir=str(output_dir / "ckpt"),
                    ckpt_save_interval=args.train.save_interval,
                    ckpt_save_policy="latest_k",
                    ckpt_max_keep=10,
                    record_lr=False,
                ),
            ]
        )

        num_params_unet, _ = count_params(ldm_with_loss.model._backbone.diffusion_model)
        num_params_text_encoder, _ = count_params(ldm_with_loss.conditioner)
        num_params_vae, _ = count_params(ldm_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(ldm_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"Debugging: {args.environment.debug}",
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.environment.mode}",
                f"JIT level: {args.jit_level}",
                f"Distributed mode: {args.environment.distributed}",
                f"Num params SVD: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision SVD: {args.train.amp_level}",
                f"Num epochs: {args.train.epochs}",
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
        shutil.copyfile(args.svd_config.absolute, output_dir / "svd.yaml")
        parser.save(args, output_dir / "svd_train.yaml", format="yaml")

    model = Model(net_with_grads)
    model.train(
        args.train.epochs,
        train_dataloader,
        callbacks=callbacks,
        dataset_sink_mode=False,  # Sinking is not supported due to memory limitations
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(init_train_env, "environment")
    parser.add_argument("--train.epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--train.save_interval", type=int, default=10, help="Checkpoints saving interval.")
    parser.add_argument("--train.temporal_only", type=bool, default=True, help="Train temporal layers only.")
    parser.add_argument("--train.pretrained", type=Path_fr, required=True, help="Path to pretrained model.")
    parser.add_argument("--train.amp_level", choices=["O0", "O2"], default="O2", help="Automatic Mixed Precision.")
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports ['O0', 'O1', 'O2']."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )

    parser.add_argument(
        "--train.output_dir",
        type=Path_dcc,
        default="output/",
        help="Output directory for saving training results.",
    )
    parser.add_subclass_arguments(BaseDataset, "train.dataset")
    parser.add_function_arguments(
        create_dataloader,
        "train.dataloader",
        skip={"dataset", "transforms", "batch_transforms", "device_num", "rank_id", "debug", "enable_modelarts"},
    )
    parser.add_function_arguments(create_scheduler, "train.scheduler", skip={"steps_per_epoch", "num_epochs"})
    parser.add_function_arguments(create_optimizer, "train.optimizer", skip={"params", "lr"})
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
