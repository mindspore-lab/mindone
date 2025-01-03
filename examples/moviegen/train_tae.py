import logging
import os
import shutil
import sys
import time

import yaml

import mindspore as ms
from mindspore import Model, amp, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from args_train_tae import parse_args
from mg.dataset.tae_dataset import BatchTransform, VideoDataset
from mg.models.tae import TemporalAutoencoder
from mg.models.tae.losses import GeneratorWithLoss
from mg.models.tae.modules import SpatialDownsample, SpatialUpsample, TemporalDownsample, TemporalUpsample

from mindone.data import create_dataloader
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
os.environ["MS_ASCEND_CHECK_OVERFLOW_MODE"] = "INFNAN_MODE"

logger = logging.getLogger(__name__)


def create_loss_scaler(loss_scaler_type, init_loss_scale, loss_scale_factor=2, scale_window=1000):
    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    return loss_scaler


def main(args):
    # 1. init
    _, rank_id, device_num = init_train_env(
        mode=args.mode,
        seed=args.seed,
        distributed=args.distributed,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        jit_level=args.jit_level,
        debug=args.debug,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. build data loader
    if isinstance(args.image_size, int):
        image_size = args.image_size
    else:
        if len(args.image_size) == 2:
            assert args.image_size[0] == args.image_size[1], "Currently only h==w is supported"
        image_size = args.image_size[0]

    dataset = VideoDataset(
        csv_path=args.csv_path,
        folder=args.folder,
        size=args.image_size,
        crop_size=args.crop_size,
        sample_n_frames=args.sample_n_frames,
        sample_stride=args.sample_stride,
        video_column=args.video_column,
        random_crop=args.random_crop,
        flip=args.flip,
        output_columns=["video"],
    )
    transform = BatchTransform(mixed_strategy=args.mixed_strategy, mixed_image_ratio=args.mixed_image_ratio)
    transform = {"operations": transform, "input_columns": ["video"]}
    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        batch_transforms=transform,
        num_workers=args.num_workers,
        max_rowsize=256,
        shuffle=True,
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=True,
    )
    dataset_size = dataloader.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")

    # 3. build models
    ae = TemporalAutoencoder(
        pretrained=args.pretrained,
        use_recompute=args.use_recompute,
    )

    # mixed precision
    # TODO: set softmax, sigmoid computed in FP32. manually set inside network since they are ops, instead of layers whose precision will be set by AMP level.
    if args.dtype != "fp32":
        dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        # TODO: check ResizeNearest bf16 support for ms>2.3.1
        ae = amp.custom_mixed_precision(
            ae,
            black_list=amp.get_black_list()
            + (
                [SpatialDownsample, SpatialUpsample, TemporalDownsample, TemporalUpsample]
                if args.vae_keep_updown_fp32
                else [] + ([nn.GroupNorm] if args.vae_keep_gn_fp32 else [])
            ),
            dtype=dtype,
        )

    # 4. build net with loss
    ae_with_loss = GeneratorWithLoss(
        ae,
        kl_weight=args.kl_loss_weight,
        perceptual_weight=args.perceptual_loss_weight,
        use_outlier_penalty_loss=args.use_outlier_penalty_loss,
        dtype=args.dtype,
    )

    tot_params, trainable_params = count_params(ae_with_loss)
    logger.info("Total params {:,}; Trainable params {:,}".format(tot_params, trainable_params))

    # 5. build training utils
    # torch scale lr by: model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    if args.scale_lr:
        learning_rate = args.start_learning_rate * args.batch_size * args.gradient_accumulation_steps * device_num
        logger.info(f"Learning rate is scaled to {learning_rate}")
    else:
        learning_rate = args.start_learning_rate
    if not args.decay_steps:
        args.decay_steps = max(1, args.epochs * dataset_size - args.warmup_steps)

    if args.scheduler != "constant":
        assert (
            args.optim != "adamw_exp"
        ), "For dynamic LR, mindspore.experimental.optim.AdamW needs to work with LRScheduler"
        lr = create_scheduler(
            steps_per_epoch=dataset_size,
            name=args.scheduler,
            lr=learning_rate,
            end_lr=args.end_learning_rate,
            warmup_steps=args.warmup_steps,
            decay_steps=args.decay_steps,
            num_epochs=args.epochs,
        )
    else:
        lr = learning_rate

    # build optimizer
    update_logvar = False  # in torch, ae_with_loss.logvar  is not updated.
    if update_logvar:
        ae_params_to_update = [ae_with_loss.autoencoder.trainable_params(), ae_with_loss.logvar]
    else:
        ae_params_to_update = ae_with_loss.autoencoder.trainable_params()
    optim_ae = create_optimizer(
        ae_params_to_update,
        name=args.optim,
        betas=args.betas,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )
    loss_scaler_ae = create_loss_scaler(
        args.loss_scaler_type, args.init_loss_scale, args.loss_scale_factor, args.scale_window
    )

    ema = (
        EMA(
            ae,
            ema_decay=args.ema_decay,
            offloading=False,
        )
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
            ae_with_loss, optim_ae, resume_ckpt
        )
        loss_scaler_ae.loss_scale_value = loss_scale
        loss_scaler_ae.cur_iter = cur_iter
        loss_scaler_ae.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume training from {resume_ckpt}")

    # training step
    training_step_ae = TrainOneStepWrapper(
        ae_with_loss,
        optimizer=optim_ae,
        scale_sense=loss_scaler_ae,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    # support dynamic shape in graph mode
    if args.mode == 0 and args.mixed_strategy == "mixed_video_random":
        # (b c t h w), drop_remainder so bs fixed
        # videos = ms.Tensor(shape=[args.batch_size, 3, None, image_size, image_size], dtype=ms.float32)
        videos = ms.Tensor(shape=[None, 3, None, image_size, image_size], dtype=ms.float32)
        training_step_ae.set_inputs(videos)
        logger.info("Dynamic inputs are initialized for mixed_video_random training in Graph mode!")

    if rank_id == 0:
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"amp level: {args.amp_level}",
                f"dtype: {args.dtype}",
                f"csv path: {args.csv_path}",
                f"Video folder: {args.video_folder}",
                f"Learning rate: {learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Rescale size: {args.image_size}",
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
    use_flexible_train = False
    if not use_flexible_train:
        model = Model(training_step_ae)

        # callbacks
        callback = [TimeMonitor(args.log_interval)]
        ofm_cb = OverflowMonitor()
        callback.append(ofm_cb)

        if rank_id == 0:
            save_cb = EvalSaveCallback(
                network=ae,
                rank_id=rank_id,
                ckpt_save_dir=ckpt_dir,
                ema=ema,
                ckpt_save_policy="latest_k",
                ckpt_max_keep=args.ckpt_max_keep,
                ckpt_save_interval=args.ckpt_save_interval,
                log_interval=args.log_interval,
                start_epoch=start_epoch,
                model_name="tae",
                record_lr=False,
            )
            callback.append(save_cb)
            if args.profile:
                callback.append(ProfilerCallback())

            logger.info("Start training...")
            # backup config files
            shutil.copyfile(args.config, os.path.join(args.output_path, os.path.basename(args.config)))

            with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
                yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

        model.train(
            args.epochs,
            dataloader,
            callbacks=callback,
            dataset_sink_mode=args.dataset_sink_mode,
            # sink_size=args.sink_size,
            initial_epoch=start_epoch,
        )
    else:
        if rank_id == 0:
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
        # output_numpy=True ?
        ds_iter = dataloader.create_dict_iterator(args.epochs - start_epoch)

        for epoch in range(start_epoch, args.epochs):
            start_time_e = time.time()
            for step, data in enumerate(ds_iter):
                start_time_s = time.time()
                x = data["video"]

                global_step = epoch * dataset_size + step
                global_step = ms.Tensor(global_step, dtype=ms.int64)

                # NOTE: inputs must match the order in GeneratorWithLoss.construct
                loss_ae_t, overflow, scaling_sens = training_step_ae(x, global_step)

                cur_global_step = epoch * dataset_size + step + 1  # starting from 1 for logging
                if overflow:
                    logger.warning(f"Overflow occurs in step {cur_global_step}")

                # log
                step_time = time.time() - start_time_s
                if step % args.log_interval == 0:
                    loss_ae = float(loss_ae_t.asnumpy())
                    logger.info(f"E: {epoch+1}, S: {step+1}, Loss ae: {loss_ae:.4f}, Step time: {step_time*1000:.2f}ms")

            epoch_cost = time.time() - start_time_e
            per_step_time = epoch_cost / dataset_size
            cur_epoch = epoch + 1
            logger.info(
                f"Epoch:[{int(cur_epoch):>3d}/{int(args.epochs):>3d}], "
                f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time*1000:.2f}ms, "
            )
            if rank_id == 0:
                if (cur_epoch % args.ckpt_save_interval == 0) or (cur_epoch == args.epochs):
                    ckpt_name = f"vae_kl_f8-e{cur_epoch}.ckpt"
                    if ema is not None:
                        ema.swap_before_eval()

                    ckpt_manager.save(ae, None, ckpt_name=ckpt_name, append_dict=None)
                    if ema is not None:
                        ema.swap_after_eval()

            # TODO: eval while training


if __name__ == "__main__":
    args = parse_args()
    main(args)
