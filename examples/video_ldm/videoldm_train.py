import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import Path_fr

from mindspore import Model, Tensor, amp, load_checkpoint, load_param_into_net, nn
from mindspore.dataset.vision import CenterCrop, Resize
from mindspore.train.callback import LossMonitor, TimeMonitor

sys.path.append("examples/stable_diffusion_v2")  # FIXME: loading modules from the SD directory
from examples.stable_diffusion_v2.common import build_model_from_config, init_env
from examples.stable_diffusion_v2.ldm.data.loader import build_dataloader
from examples.stable_diffusion_v2.ldm.data.transforms import TokenizerWrapper
from examples.stable_diffusion_v2.ldm.modules.logger import set_logger
from examples.stable_diffusion_v2.ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from examples.stable_diffusion_v2.ldm.modules.train.lr_schedule import create_scheduler
from examples.stable_diffusion_v2.ldm.modules.train.optim import build_optimizer
from examples.stable_diffusion_v2.ldm.modules.train.trainer import TrainOneStepWrapper
from examples.stable_diffusion_v2.ldm.util import count_params
from examples.video_ldm.data.video_dataset import VideoDataset


class ModelWrapper(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self._net = network

    def construct(self, batch: Tensor, context: Tensor):
        # merge the batch dimension with the frame dimension b t c h w -> (b t) c h w
        batch = batch.reshape(-1, *batch.shape[2:])
        context = context.reshape(-1, *context.shape[2:])
        return self._net(batch, context)


def load_pretrained_model(logger, net, pretrained_ckpt, weights_map: Optional[dict] = None):
    logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    param_dict = load_checkpoint(pretrained_ckpt)
    if weights_map is not None:
        for old_name, new_name in weights_map.items():
            assert new_name not in param_dict  # FIXME
            param_dict[new_name] = param_dict.pop(old_name)

    param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)
    return param_not_load, ckpt_not_load


def build_transforms(tokenizer, frames_num) -> List[dict]:
    return [
        {
            "operations": [
                Resize(320),
                CenterCrop((320, 512)),
                lambda x: (x / 127.5 - 1.0).astype(np.float32),
            ],
            "input_columns": ["frames"],
        },
        {
            "operations": [
                TokenizerWrapper(tokenizer),
                lambda x: np.tile(x, (frames_num, 1)),  # expand the number of prompts to match the number of frames
            ],
            "input_columns": ["caption"],
        },
    ]


def mixed_precision(network):
    from mindspore.nn import GroupNorm, SiLU

    from examples.video_ldm.modules.unet3d import GroupNorm3D

    black_list = amp.get_black_list() + [SiLU, GroupNorm, GroupNorm3D]
    return amp.custom_mixed_precision(network, black_list=black_list)


def main(args, initializer):
    # step 1: initialize environment
    logger = logging.getLogger(__name__)
    device_id, rank_id, device_num = init_env(**args.environment)

    output_dir = Path(args.train.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_logger(output_dir=str(output_dir), rank=rank_id)

    # step 2: load SD model
    ldm_with_loss = build_model_from_config(args.SD.absolute)
    temporal_param_names = ldm_with_loss.model.diffusion_model.get_temporal_params(prefix="model.diffusion_model.")
    new_weights_map = ldm_with_loss.model.diffusion_model.get_weights_map(prefix="model.diffusion_model.")
    param_not_load, _ = load_pretrained_model(logger, ldm_with_loss, args.train.pretrained, weights_map=new_weights_map)

    if param_not_load:
        diff = set(param_not_load).difference(temporal_param_names)
        if not diff:
            logger.info("Temporal parameters are not loaded.")
        else:
            logger.warning(f"Failed to load parameters: {diff}")

    if args.train.temporal_only:
        for param in ldm_with_loss.trainable_params():
            if param.name not in temporal_param_names:
                param.requires_grad = False

    ldm_with_loss_wrap = ModelWrapper(ldm_with_loss)
    ldm_with_loss_wrap = mixed_precision(ldm_with_loss_wrap)

    # step 3: prepare train dataset and dataloader
    transforms = build_transforms(ldm_with_loss.cond_stage_model.tokenizer, args.train.dataset.init_args.frames)
    train_dataloader = build_dataloader(
        initializer.train.dataset,
        transforms=transforms,
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
        ldm_with_loss_wrap, optimizer=optimizer, scale_sense=loss_scaler, **args.train.settings
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
                    model_name="videoldm",
                    rank_id=rank_id,
                    ckpt_save_dir=str(output_dir / "ckpt"),
                    ckpt_save_interval=ckpt_save_interval,
                    ckpt_save_policy="latest_k",
                    ckpt_max_keep=10,
                ),
            ]
        )

        num_params_unet, _ = count_params(ldm_with_loss.model.diffusion_model)
        num_params_text_encoder, _ = count_params(ldm_with_loss.cond_stage_model)
        num_params_vae, _ = count_params(ldm_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(ldm_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"Debugging: {args.environment.debug}",
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.environment.mode}",
                f"Distributed mode: {args.environment.distributed}",
                "Model: StableDiffusion v2.1",  # Support 1.x
                f"Num params SD: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision SD: {ldm_with_loss.model.diffusion_model.dtype}",
                f"Num epochs: {args.train.epochs} {f'(adjusted to sink size {args.train.sink_size})' if not args.environment.debug else ''}",
                f"Learning rate: {args.train.scheduler.lr}",
                f"Batch size: {args.train.dataloader.batch_size}",
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
        shutil.copyfile(args.SD.absolute, output_dir / "sd_config.yaml")  # SD's parameters are not modified
        ArgumentParser().save(args, output_dir / "videoldm_config.yaml", format="yaml", skip_check=True)

    model = Model(net_with_grads)
    model.train(
        args.train.epochs,
        train_dataloader,
        callbacks=callbacks,
        dataset_sink_mode=not args.environment.debug,
        sink_size=args.train.sink_size,
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(init_env, "environment")
    parser.add_argument("--train.epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--train.sink_size", type=int, default=-1, help="Number of steps in each data sinking.")
    parser.add_argument("--train.temporal_only", type=bool, default=True, help="Train temporal layers only.")
    parser.add_argument("--train.pretrained", type=str, required=True, help="Path to pretrained model.")
    parser.add_argument(
        "--train.output_dir",
        type=str,
        default="output/videoldm/",
        help="Output directory for saving training results.",
    )
    parser.add_subclass_arguments(VideoDataset, "train.dataset")
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
        "--SD",
        type=Path_fr,
        default="examples/video_ldm/configs/sd_v2.1.yaml",
        help="Stable Diffusion model configuration.",
    )
    parser.add_class_arguments(nn.DynamicLossScaleUpdateCell, "LossScale", instantiate=False, fail_untyped=False)

    cfg = parser.parse_args()
    init = parser.instantiate_classes(cfg)

    main(cfg, init)
