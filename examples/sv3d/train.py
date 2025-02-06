import os
import sys
from datetime import datetime
from pathlib import Path

from jsonargparse import ArgumentParser
from omegaconf import OmegaConf

sys.path.append("../..")  # FIXME: loading mindone, remove in future when mindone is ready for install

from sgm.helpers import create_model_sv3d
from sgm.modules.train.callback import LossMonitor
from sgm.util import get_obj_from_str
from utils import mixed_precision

from mindspore import Callback, Model, nn

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params


class SetTrainCallback(Callback):
    # TODO: is it necessary?
    def on_train_begin(self, run_context):
        cb_params = run_context.original_args()
        cb_params["network"].network.set_train(False)
        cb_params["network"].network.model.set_train(True)


def main(args):
    # step 1: initialize environment
    train_cfg = OmegaConf.load(args.train_cfg)
    loss_scaler = nn.DynamicLossScaleUpdateCell(**train_cfg.LossScale)
    device_id, rank_id, device_num = init_train_env(**train_cfg.environment)
    _debug = args.debug
    _mode = train_cfg.environment.mode
    if not _debug:
        output_dir = Path(train_cfg.train.output_dir) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = train_cfg.train.output_dir
    logger = set_logger(
        name="", output_dir=str(output_dir), rank=rank_id
    )  # all the logger needs to follow name, to use the mindone callbacks directly, need to put name as ""
    train_cfg = train_cfg.train

    # step 2: load SV3D model
    model_cfg = OmegaConf.load(args.model_cfg)

    ldm_with_loss, _ = create_model_sv3d(
        model_cfg,
        train_cfg,
        checkpoints=train_cfg.pretrained,
        freeze=False,
        amp_level=train_cfg.amp_level,
    )

    temporal_param_names = ldm_with_loss.model.diffusion_model.get_temporal_param_names(prefix="model.diffusion_model.")
    if train_cfg.temporal_only:
        for param in ldm_with_loss.trainable_params():
            if param.name not in temporal_param_names:
                param.requires_grad = False

    if train_cfg.amp_level == "O2":  # Set mixed precision for certain modules only
        mixed_precision(ldm_with_loss)

    # step 3: prepare train dataset and dataloader
    dataset = get_obj_from_str(train_cfg.dataset.class_path)(**train_cfg.dataset.init_args)
    train_dataloader = create_dataloader(
        dataset,
        transforms=dataset.train_transforms(),
        device_num=device_num,
        rank_id=rank_id,
        debug=False,  # ms240_sept4 daily pkg err. THIS CANNOT BE TRUE, OTHERWISE the dataloader err: num_worker modulo
        **train_cfg.dataloader,
    )

    # step 4: create optimizer and train the same way as regular SD
    # FIXME: set decay steps in the scheduler function
    train_cfg.scheduler.decay_steps = (
        train_cfg.epochs * train_dataloader.get_dataset_size() - train_cfg.scheduler.warmup_steps
    )
    lr = create_scheduler(
        steps_per_epoch=train_dataloader.get_dataset_size(), num_epochs=train_cfg.epochs, **train_cfg.scheduler
    )
    optimizer = create_optimizer(ldm_with_loss.trainable_params(), lr=lr, **train_cfg.optimizer)

    net_with_grads = TrainOneStepWrapper(
        ldm_with_loss, optimizer=optimizer, scale_sense=loss_scaler, **train_cfg.settings
    )

    callbacks = [SetTrainCallback()]

    if rank_id == 0:
        callbacks.extend(
            [
                LossMonitor(),
                EvalSaveCallback(
                    network=ldm_with_loss,
                    model_name="sv3d",
                    rank_id=rank_id,
                    ckpt_save_dir=os.path.join(output_dir, "ckpt"),
                    ckpt_save_interval=train_cfg.save_interval,
                    ckpt_save_policy="latest_k",
                    log_interval=train_cfg.log_interval,
                    ckpt_max_keep=train_cfg.ckpt_max_keep,
                    record_lr=True,
                ),
            ]
        )

        if train_cfg.amp_level == "O0":
            num_params_unet, _ = count_params(ldm_with_loss.model.diffusion_model)
        else:
            num_params_unet, _ = count_params(ldm_with_loss.model._backbone.diffusion_model)
        num_params_text_encoder, _ = count_params(ldm_with_loss.conditioner)
        num_params_vae, _ = count_params(ldm_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(ldm_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"Debugging: {_debug}",
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {_mode}",
                f"Num params sv3d: {num_params:,} (unet: {num_params_unet:,}, \
                    text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision sv3d: {train_cfg.amp_level}",
                f"Num epochs: {train_cfg.epochs}",
                f"Learning rate: {train_cfg.scheduler.lr}",
                f"Batch size: {train_cfg.dataloader.batch_size}",
                f"Number of frames: {train_cfg.dataset.init_args.frames}",
                f"Weight decay: {train_cfg.optimizer.weight_decay}",
                f"Grad accumulation steps: {train_cfg.settings.gradient_accumulation_steps}",
                f"Grad clipping: {train_cfg.settings.clip_grad}",
                f"Max grad norm: {train_cfg.settings.clip_norm}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

    model = Model(net_with_grads)
    model.train(
        train_cfg.epochs,
        train_dataloader,
        callbacks=callbacks,
        dataset_sink_mode=False,  # Sinking is not supported due to memory limitations
    )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--train_cfg",
        default="configs/sv3d_u_train.yaml",
        help="SV3D training configuration.",
    )
    parser.add_argument(
        "--model_cfg",
        default="configs/sampling/sv3d_u.yaml",
        help="SV3D model configuration.",
    )
    parser.add_argument("--debug", action="store_true")
    cfg = parser.parse_args()
    main(cfg)
