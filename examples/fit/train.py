#!/usr/bin/env python
"""
FiT training pipeline
"""
import datetime
import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

import yaml
from args_train import parse_args
from data.imagenet_dataset import create_dataloader_imagenet_latent
from diffusion import create_diffusion
from pipelines.train_pipeline import NetworkWithLoss
from utils.model_utils import load_fit_ckpt_params

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

from mindone.models.fit import FiT_models

# load training modules
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def set_fit_all_params(fit_model, train=True, **kwargs):
    n_params_trainable = 0
    for param in fit_model.get_parameters():
        param.requires_grad = train
        if train:
            n_params_trainable += 1
    logger.info(f"Set {n_params_trainable} params to train.")


def set_fit_params(fit_model, ft_all_params, **kwargs):
    if ft_all_params:
        set_fit_all_params(fit_model, **kwargs)
    else:
        raise ValueError("Fintuning partial params is not supported!")


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str)

    # 1. init
    _, rank_id, device_num = init_train_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
    )

    if args.mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": args.jit_level})

    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    # 2.1 fit
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    fit_model = FiT_models[args.model_name](
        num_classes=1000,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        pos=args.embed_method,
    )
    if args.use_fp16:
        fit_model = auto_mixed_precision(fit_model, amp_level="O2")

    if args.fit_checkpoint:
        fit_model = load_fit_ckpt_params(fit_model, args.fit_checkpoint)
    else:
        logger.info("Initialize FIT randomly.")
    fit_model.set_train(True)

    set_fit_params(fit_model, ft_all_params=True, train=True)

    diffusion = create_diffusion(timestep_respacing="")

    model_config = dict(C=4, H=args.image_size // 8, W=args.image_size // 8, patch_size=args.patch_size)
    latent_diffusion_with_loss = NetworkWithLoss(
        fit_model,
        diffusion,
        vae=None,
        scale_factor=args.sd_scale_factor,
        condition=args.condition,
        model_config=model_config,
    )

    # image dataset
    if args.imagenet_format:
        data_config = dict(
            data_folder=args.data_path,
            sample_size=args.image_size,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_parallel_workers=args.num_parallel_workers,
            patch_size=args.patch_size,
            embed_dim=args.embed_dim,
            embed_method=args.embed_method,
        )
        dataset = create_dataloader_imagenet_latent(
            data_config,
            device_num=device_num,
            rank_id=rank_id,
        )
    else:
        raise NotImplementedError("FiT support ImageNet format dataset only")

    dataset_size = dataset.get_dataset_size()

    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler
    if not args.decay_steps:
        args.decay_steps = args.epochs * dataset_size - args.warmup_steps  # fix lr scheduling
        if args.decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.decay_steps = 1

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.scheduler,
        lr=args.start_learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    optimizer = create_optimizer(
        latent_diffusion_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    # resume ckpt
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    start_epoch = 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(fit_model, optimizer, resume_ckpt)
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss.network,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    model = Model(net_with_grads)
    # callbacks
    callback = [TimeMonitor(args.callback_size)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss.network,  # save fit only
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=args.ckpt_max_keep,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            log_interval=args.callback_size,
            start_epoch=start_epoch,
            model_name="FiT",
            record_lr=True,
        )
        callback.append(save_cb)
        if args.profile:
            callback.append(ProfilerCallback())

    # 5. log and save config
    if rank_id == 0:
        # 4. print key info
        num_params_fit, num_params_fit_trainable = count_params(fit_model)
        num_params = num_params_fit
        num_params_trainable = num_params_fit_trainable
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"JIT level: {args.jit_level}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Num params: {num_params:,} (fit: {num_params_fit:,})",
                f"Num trainable params: {num_params_trainable:,}",
                f"Use FP16: {args.use_fp16}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Image size: {args.image_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Total training steps: {dataset_size * args.epochs:,}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Enable flash attention: {args.enable_flash_attention}",
                f"Dataset sink: {args.dataset_sink_mode}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    # 6. train
    model.train(
        args.epochs,
        dataset,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
