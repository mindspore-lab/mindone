import logging
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import Path_fr

from mindspore import Model, Tensor, nn
from mindspore.dataset.vision import CenterCrop, Resize
from mindspore.train.callback import LossMonitor, TimeMonitor

from examples.stable_diffusion_v2.common import build_model_from_config, init_env
from examples.stable_diffusion_v2.ldm.data.loader import build_dataloader
from examples.stable_diffusion_v2.ldm.data.transforms import TokenizerWrapper
from examples.stable_diffusion_v2.ldm.modules.logger import set_logger
from examples.stable_diffusion_v2.ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from examples.stable_diffusion_v2.ldm.modules.train.optim import build_optimizer
from examples.stable_diffusion_v2.ldm.modules.train.trainer import TrainOneStepWrapper
from examples.stable_diffusion_v2.ldm.util import count_params
from examples.video_ldm.data.video_dataset import VideoDataset

sys.path.append("examples/stable_diffusion_v2")  # FIXME: loading modules from the SD directory


class ModelWrapper(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self._net = network

    def construct(self, batch: Tensor, context: Tensor):
        # merge the batch dimension with the frame dimension b t c h w -> (b t) c h w
        batch = batch.reshape(-1, *batch.shape[2:])
        context = context.reshape(-1, *context.shape[2:])
        return self._net(batch, context)


def filter_temporal_layers(model):
    for param in model.trainable_params():
        if not (
            re.match(".*input_blocks.[124578].[1|3].*", param.name)
            or re.match(".*output_blocks.([3-9]|1[01]).[1|3].*", param.name)
        ):
            param.requires_grad = False
    return model


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


def main(args, initializer):
    # step 1: initialize environment
    logger = logging.getLogger(__name__)
    device_id, rank_id, device_num = init_env(logger, **args.environment)

    output_dir = Path(args.train.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    set_logger(output_dir=str(output_dir), rank=rank_id)

    # step 2: load SD model
    ldm_with_loss = build_model_from_config(args.SD.absolute)
    ldm_with_loss = filter_temporal_layers(ldm_with_loss)
    ldm_with_loss_wrap = ModelWrapper(ldm_with_loss)

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
    optimizer = build_optimizer(ldm_with_loss, **args.train.optimizer)

    loss_scaler = nn.DynamicLossScaleUpdateCell(**args.LossScale)

    net_with_grads = TrainOneStepWrapper(
        ldm_with_loss_wrap, optimizer=optimizer, scale_sense=loss_scaler, **args.train.settings
    )

    callbacks = [OverflowMonitor()]

    if rank_id == 0:
        callbacks.extend(
            [
                TimeMonitor(1),
                LossMonitor(),
                EvalSaveCallback(
                    network=ldm_with_loss,
                    model_name="videoldm",
                    rank_id=rank_id,
                    ckpt_save_dir=str(output_dir / "ckpt"),
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
                f"Num epochs: {args.train.epochs}",
                f"Learning rate: {args.train.optimizer.lr}",
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
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_function_arguments(init_env, "environment", skip={"logger"})
    parser.add_argument("--train.epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument(
        "--train.output_dir",
        type=str,
        default="output/output/videoldm/",
        help="Output directory for saving training results.",
    )
    parser.add_subclass_arguments(VideoDataset, "train.dataset")
    parser.add_function_arguments(
        build_dataloader,
        "train.dataloader",
        skip={"dataset", "transforms", "device_num", "rank_id", "debug", "enable_modelarts"},
    )
    parser.add_function_arguments(build_optimizer, "train.optimizer", skip={"model"})
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
