import logging
import os
import shutil
import sys
import time
from typing import Tuple

import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from args_train_vae import parse_args
from opensora.datasets.vae_dataset import create_dataloader
from opensora.models.layers.operation_selector import set_dynamic_mode
from opensora.models.vae.losses import GeneratorWithLoss
from opensora.models.vae.vae import OpenSoraVAE_V1_2

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed

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


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    parallel_mode: str = "data",
    jit_level: str = "O2",
    global_bf16: bool = False,
    dynamic_shape: bool = False,
    debug: bool = False,
) -> Tuple[int, int]:
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if debug and mode == ms.GRAPH_MODE:  # force PyNative mode when debugging
        logger.warning("Debug mode is on, switching execution mode to PyNative.")
        mode = ms.PYNATIVE_MODE

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    # ms.set_context(mempool_block_size="55GB")
    # ms.set_context(pynative_synchronize=True)
    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        if parallel_mode == "optim":
            print("use optim parallel")
            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL,
                enable_parallel_optimizer=True,
            )
            init()
            device_num = get_group_size()
            rank_id = get_rank()
        else:
            init()
            device_num = get_group_size()
            rank_id = get_rank()
            logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
            ms.reset_auto_parallel_context()

            ms.set_auto_parallel_context(
                parallel_mode=ms.ParallelMode.DATA_PARALLEL,
                gradients_mean=True,
                device_num=device_num,
            )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            pynative_synchronize=debug,
        )

    if mode == 0:
        ms.set_context(jit_config={"jit_level": jit_level})

    if global_bf16:
        # only effective in GE mode, i.e. jit_level: O2
        ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})

    if dynamic_shape:
        print("Dynamic shape mode enabled, repeat_interleave/split/chunk will be called from mint module")
        set_dynamic_mode(True)

    return rank_id, device_num


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
        dynamic_shape=(args.mixed_strategy == "mixed_video_random"),
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

    ds_config = dict(
        csv_path=args.csv_path,
        data_folder=args.video_folder,
        size=image_size,
        crop_size=image_size,
        sample_n_frames=args.num_frames,
        sample_stride=args.frame_stride,
        video_column=args.video_column,
        random_crop=args.random_crop,
        flip=args.flip,
    )
    dataloader = create_dataloader(
        ds_config,
        args.batch_size,
        mixed_strategy=args.mixed_strategy,
        mixed_image_ratio=args.mixed_image_ratio,
        num_parallel_workers=args.num_parallel_workers,
        max_rowsize=256,
        shuffle=True,
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=True,
    )
    dataset_size = dataloader.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")

    # 3. build models
    if args.model_type == "OpenSoraVAE_V1_2":
        logger.info(f"Loading autoencoder from {args.pretrained_model_path}")
        if args.micro_frame_size != 17:
            logger.warning(
                "If you are finetuning VAE3d pretrained from OpenSora v1.2, it's safer to set micro_frame_size to 17 for consistency."
            )
        ae = OpenSoraVAE_V1_2(
            micro_batch_size=args.micro_batch_size,
            micro_frame_size=args.micro_frame_size,
            ckpt_path=args.pretrained_model_path,
            freeze_vae_2d=args.freeze_vae_2d,
            cal_loss=True,
            use_recompute=args.use_recompute,
        )
    else:
        raise NotImplementedError("Only OpenSoraVAE_V1_2 is supported for vae training currently")

    if args.use_discriminator:
        logging.error("Discriminator is not used or supported in OpenSora v1.2")

    # mixed precision
    # TODO: set softmax, sigmoid computed in FP32. manually set inside network since they are ops, instead of layers whose precision will be set by AMP level.
    if args.dtype in ["fp16", "bf16"]:
        dtype = {"fp16": ms.float16, "bf16": ms.bfloat16}[args.dtype]
        ae = auto_mixed_precision(
            ae,
            args.amp_level,
            dtype,
            custom_fp32_cells=[nn.GroupNorm] if args.vae_keep_gn_fp32 else [],
        )

    # 4. build net with loss
    ae_with_loss = GeneratorWithLoss(
        ae,
        kl_weight=args.kl_loss_weight,
        perceptual_weight=args.perceptual_loss_weight,
        use_real_rec_loss=args.use_real_rec_loss,
        use_z_rec_loss=args.use_z_rec_loss,
        use_image_identity_loss=args.use_image_identity_loss,
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
                model_name="vae_3d",
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
