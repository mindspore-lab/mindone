"""Trainer for VQVAE"""

import logging
import os
import shutil
import sys
import time

import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from mindcv.optim import create_optimizer
from utils.env import init_env, set_all_reduce_fusion
from videogvt.config.vqgan3d_ucf101_config import get_config
from videogvt.config.vqvae_train_args import parse_args
from videogvt.data.loader import create_dataloader
from videogvt.models.vqvae import StyleGANDiscriminator, build_model
from videogvt.models.vqvae.net_with_loss import DiscriminatorWithLoss, GeneratorWithLoss

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler

# from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

logger = logging.getLogger(__name__)


def create_loss_scaler(loss_scaler_type, init_loss_scale, loss_scale_factor=2, scale_window=1000):
    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=init_loss_scale,
            scale_factor=loss_scale_factor,
            scale_window=scale_window,
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(init_loss_scale)
    else:
        raise ValueError

    return loss_scaler


def main(args):
    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        jit_level=args.jit_level,
        global_bf16=args.global_bf16,
        debug=args.debug,
    )

    set_logger(
        name="",
        output_dir=args.output_path,
        rank=rank_id,
        log_level=eval(args.log_level),
    )

    # 2. build models
    #  vqvae (G)
    model_config = get_config()
    dtype = {"fp32": ms.float32, "fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
    vqvae = build_model(args.model_class, model_config, is_training=True, pretrained=args.pretrained, dtype=dtype)

    # discriminator (D)
    use_discriminator = args.use_discriminator and (model_config.lr_configs.disc_weight > 0.0)

    if args.use_discriminator and (model_config.lr_configs.disc_weight <= 0.0):
        logging.warning("use_discriminator is True but disc_weight is 0.")

    if use_discriminator:
        crop_size = int(args.crop_size)
        frame_size = int(args.num_frames)
        disc = StyleGANDiscriminator(model_config.discriminator, crop_size, crop_size, frame_size, dtype=dtype)
    else:
        disc = None

    # mixed precision
    # TODO: set softmax, sigmoid computed in FP32. manually set inside network since they are ops, instead of layers whose precision will be set by AMP level.
    if args.dtype not in ["fp16", "bf16"]:
        amp_level = "O2"
        if not args.global_bf16:
            vqvae = auto_mixed_precision(
                vqvae,
                amp_level=auto_mixed_precision(
                    vqvae,
                    amp_level=amp_level,
                    dtype=dtype,
                    custom_fp32_cells=[nn.GroupNorm] if args.vae_keep_gn_fp32 else [],
                ),
            )
    else:
        amp_level = "O0"

    # 3. build net with loss (core)
    # G with loss
    vqvae_with_loss = GeneratorWithLoss(
        vqvae,
        discriminator=disc,
        is_video=(args.dataset_name == "video"),
        **model_config.lr_configs,
        dtype=dtype,
    )
    disc_start = model_config.lr_configs.disc_start

    # D with loss
    if use_discriminator:
        disc_with_loss = DiscriminatorWithLoss(vqvae, disc, disc_start)

    tot_params, trainable_params = count_params(vqvae_with_loss)
    logger.info("Total params {:,}; Trainable params {:,}".format(tot_params, trainable_params))

    # 4. build dataset
    ds_config = dict(
        csv_path=args.csv_path,
        data_folder=args.data_path,
        size=args.size,
        crop_size=args.crop_size,
        random_crop=args.random_crop,
        flip=args.flip,
    )
    if args.dataset_name == "video":
        ds_config.update(
            dict(
                sample_stride=args.frame_stride,
                sample_n_frames=args.num_frames,
                return_image=False,
            )
        )
        assert not (
            # model_config.generator.params.ddconfig.split_time_upsample
            args.num_frames % 2 == 0
            and False
        ), "num of frames must be odd if split_time_upsample is True"
    else:
        ds_config.update(dict(expand_dim_t=args.expand_dim_t))
    dataset = create_dataloader(
        ds_config=ds_config,
        batch_size=args.batch_size,
        ds_name=args.dataset_name,
        num_parallel_workers=args.num_parallel_workers,
        shuffle=args.shuffle,
        device_num=device_num,
        rank_id=rank_id,
    )
    dataset_size = dataset.get_dataset_size()

    # 5. build training utils
    # torch scale lr by: model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    if args.scale_lr:
        learning_rate = args.base_learning_rate * args.batch_size * args.gradient_accumulation_steps * device_num
    else:
        learning_rate = args.base_learning_rate

    total_train_steps = args.epochs * dataset_size

    if not args.decay_steps:
        args.decay_steps = max(1, total_train_steps - args.warmup_steps)

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.scheduler,
        lr=learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    set_all_reduce_fusion(
        vqvae_with_loss.trainable_params(),
        split_num=7,
        distributed=args.use_parallel,
        parallel_mode=args.parallel_mode,
    )

    # build optimizer
    update_logvar = False  # in torch, vqvae_with_loss.logvar  is not updated.
    if update_logvar:
        vqvae_params_to_update = [
            vqvae_with_loss.vqvae.trainable_params(),
            vqvae_with_loss.logvar,
        ]
    else:
        vqvae_params_to_update = vqvae_with_loss.vqvae.trainable_params()

    optim_vqvae = create_optimizer(
        vqvae_params_to_update,
        opt=args.optim,
        weight_decay=args.weight_decay,
        lr=lr,
        eps=1e-08,
        beta1=0.9,
        beta2=0.999,
        weight_decay_filter="norm_and_bias",
    )

    loss_scaler_vqvae = create_loss_scaler(
        args.loss_scaler_type,
        args.init_loss_scale,
        args.loss_scale_factor,
        args.scale_window,
    )

    if use_discriminator:
        optim_disc = create_optimizer(
            disc_with_loss.discriminator.trainable_params(),
            opt=args.optim,
            weight_decay=args.weight_decay,
            lr=lr,
            eps=1e-08,
            beta1=0.9,
            beta2=0.999,
            weight_decay_filter="norm_and_bias",
        )

        loss_scaler_disc = create_loss_scaler(
            args.loss_scaler_type,
            args.init_loss_scale,
            args.loss_scale_factor,
            args.scale_window,
        )

    ema = (
        EMA(
            vqvae_with_loss.vqvae,
            ema_decay=args.ema_decay,
            offloading=False,
        ).to_float(dtype)
        if args.use_ema
        else None
    )

    # resume training states
    # TODO: resume Discriminator if used
    ckpt_dir = os.path.join(args.output_path, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            vqvae_with_loss, optim_vqvae, resume_ckpt
        )
        loss_scaler_vqvae.loss_scale_value = loss_scale
        loss_scaler_vqvae.cur_iter = cur_iter
        loss_scaler_vqvae.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    # training step
    training_step_vqvae = TrainOneStepWrapper(
        vqvae_with_loss,
        optimizer=optim_vqvae,
        scale_sense=loss_scaler_vqvae,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    if use_discriminator:
        training_step_disc = TrainOneStepWrapper(
            disc_with_loss,
            optimizer=optim_disc,
            scale_sense=loss_scaler_disc,
            drop_overflow_update=args.drop_overflow_update,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            clip_grad=args.clip_grad,
            clip_norm=args.max_grad_norm,
            ema=None,  # No ema for disriminator
        )

    if rank_id == 0:
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"amp level: {amp_level}",
                f"dtype: {args.dtype}",
                f"Data path: {args.data_path}",
                f"Learning rate: {learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Rescale size: {args.size}",
                f"Crop size: {args.crop_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

    # 6. training process
    # backup config files
    args.config = "videogvt/config/vqgan3d_ucf101_config.py"
    shutil.copyfile(args.config, os.path.join(args.output_path, os.path.basename(args.config)))
    with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

    if not use_discriminator:
        if args.global_bf16:
            model = Model(training_step_vqvae, amp_level="O0")
        else:
            model = Model(training_step_vqvae)

        # callbacks
        callback = [TimeMonitor(args.log_interval)]
        ofm_cb = OverflowMonitor()
        callback.append(ofm_cb)

        if rank_id == 0:
            save_cb = EvalSaveCallback(
                network=vqvae_with_loss.vqvae,
                rank_id=rank_id,
                ckpt_save_dir=ckpt_dir,
                ema=ema,
                ckpt_save_policy="latest_k",
                ckpt_max_keep=args.ckpt_max_keep,
                ckpt_save_interval=args.ckpt_save_interval,
                log_interval=args.log_interval,
                start_epoch=start_epoch,
                model_name="vqvae_3d",
                record_lr=False,
            )
            callback.append(save_cb)
            if args.profile:
                callback.append(ProfilerCallback())

            logger.info("Start training...")

        model.train(
            args.epochs,
            dataset,
            callbacks=callback,
            dataset_sink_mode=args.dataset_sink_mode,
            # sink_size=args.sink_size,
            initial_epoch=start_epoch,
        )

    else:
        if rank_id == 0:
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)

        # output_numpy=True ?
        ds_iter = dataset.create_dict_iterator(args.epochs - start_epoch)
        bp_steps = 0

        logger.info("Start training...")
        for epoch in range(start_epoch, args.epochs):
            epoch_loss = 0.0
            avg_loss = 0.0
            start_time_e = time.time()

            for step, data in enumerate(ds_iter):
                start_time_s = time.time()
                x = data[args.dataset_name].to(dtype)

                cur_global_step = epoch * dataset_size + step + 1

                # NOTE: inputs must match the order in GeneratorWithLoss.construct
                loss_vqvae_t, overflow, scaling_sens = training_step_vqvae(x)
                loss_disc_t, overflow_d, scaling_sens_d = training_step_disc(x)

                if overflow:
                    logger.warning(f"Overflow occurs in step {cur_global_step}")

                # loss
                loss = float(loss_vqvae_t.asnumpy()) + float(loss_disc_t.asnumpy())
                avg_loss += loss
                epoch_loss += loss

                # log
                step_time = time.time() - start_time_s
                if (step + 1) % args.log_interval == 0:
                    avg_loss /= float(args.log_interval)
                    logger.info(
                        f"E: {epoch+1}, S: {step+1}, Loss vqvae avg: {avg_loss:.4f}, Step time: {step_time*1000:.2f}ms"
                    )
                    avg_loss = 0.0
                    bp_steps += 1

                if rank_id == 0 and args.step_mode:
                    cur_epoch = epoch + 1
                    if (cur_global_step % args.ckpt_save_interval == 0) or (cur_global_step == total_train_steps):
                        ckpt_name = f"vqvae-s{cur_global_step}.ckpt"
                        if ema is not None:
                            ema.swap_before_eval()
                        vqvae_with_loss.set_train(False)
                        disc_with_loss.set_train(False)
                        ckpt_manager.save(vqvae_with_loss.vqvae, None, ckpt_name=ckpt_name, append_dict=None)

                        if ema is not None:
                            ema.swap_after_eval()
                        vqvae_with_loss.set_train(True)
                        disc_with_loss.set_train(True)

                if cur_global_step == total_train_steps:
                    break

            epoch_cost = time.time() - start_time_e
            per_step_time = epoch_cost / dataset_size
            cur_epoch = epoch + 1
            epoch_loss /= dataset_size
            logger.info(
                f"Epoch:[{int(cur_epoch):>3d}/{int(args.epochs):>3d}], loss avg: {epoch_loss:.4f},"
                f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time*1000:.2f}ms, "
            )

            if rank_id == 0 and args.step_mode:
                cur_epoch = epoch + 1
                if (cur_global_step % args.ckpt_save_interval == 0) or (cur_global_step == total_train_steps):
                    ckpt_name = f"vqvae-e{cur_epoch}.ckpt"
                    if ema is not None:
                        ema.swap_before_eval()
                    vqvae_with_loss.set_train(False)
                    disc_with_loss.set_train(False)
                    ckpt_manager.save(vqvae_with_loss.vqvae, None, ckpt_name=ckpt_name, append_dict=None)

                    if ema is not None:
                        ema.swap_after_eval()
                    vqvae_with_loss.set_train(True)
                    disc_with_loss.set_train(True)

            if cur_global_step == total_train_steps:
                break


if __name__ == "__main__":
    args = parse_args()
    main(args)
