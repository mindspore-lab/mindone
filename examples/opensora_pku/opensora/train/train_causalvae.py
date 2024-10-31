"""
Train AutoEncoders with GAN loss
"""
import json
import logging
import math
import os
import shutil
import sys
import time

import yaml

import mindspore as ms
from mindspore import Model, nn
from mindspore.train.callback import TimeMonitor

sys.path.append(".")
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from opensora.models.causalvideovae.model import EMA, CausalVAEModel
from opensora.models.causalvideovae.model.dataset_videobase import VideoDataset, create_dataloader
from opensora.models.causalvideovae.model.losses.net_with_loss import DiscriminatorWithLoss, GeneratorWithLoss
from opensora.models.causalvideovae.model.modules.updownsample import TrilinearInterpolate
from opensora.models.causalvideovae.model.utils.model_utils import resolve_str_to_obj
from opensora.train.commons import create_loss_scaler, parse_args
from opensora.utils.ms_utils import init_env
from opensora.utils.utils import get_precision

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params

logger = logging.getLogger(__name__)


def main(args):
    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
    )
    if args.exp_name is not None and len(args.exp_name) > 0:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
    set_logger(name="", output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))

    # Load Config
    assert os.path.exists(args.model_config), f"{args.model_config} does not exist!"
    model_config = json.load(open(args.model_config, "r"))
    ae = CausalVAEModel.from_config(model_config, use_recompute=args.use_recompute)
    if args.load_from_checkpoint is not None:
        ae.init_from_ckpt(args.load_from_checkpoint)
    # discriminator (D)
    use_discriminator = args.use_discriminator and (args.disc_weight > 0.0)

    if args.use_discriminator and (args.disc_weight <= 0.0):
        logging.warning("use_discriminator is True but disc_weight is 0.")

    if use_discriminator:
        disc_type = args.disc_cls
        if "LPIPSWithDiscriminator3D" in disc_type:
            disc_type = "opensora.models.causalvideovae.model.losses.discriminator.NLayerDiscriminator3D"
            use_3d_disc = True
        elif "LPIPSWithDiscriminator" in disc_type:
            disc_type = "opensora.models.causalvideovae.model.losses.discriminator.NLayerDiscriminator"
            use_3d_disc = False
        disc = resolve_str_to_obj(disc_type, append=False)()
    else:
        disc = None

    # mixed precision
    # TODO: set softmax, sigmoid computed in FP32. manually set inside network since they are ops, instead of layers whose precision will be set by AMP level.
    if args.precision in ["fp16", "bf16"]:
        amp_level = args.amp_level
        dtype = get_precision(args.precision)
        if dtype == ms.float16:
            custom_fp32_cells = [nn.GroupNorm, nn.Softmax, nn.SiLU] if args.vae_keep_gn_fp32 else [nn.Softmax, nn.SiLU]
        else:
            custom_fp32_cells = [nn.AvgPool2d, TrilinearInterpolate, nn.Softmax, nn.SiLU]
        ae = auto_mixed_precision(ae, amp_level=amp_level, dtype=dtype, custom_fp32_cells=custom_fp32_cells)
        logger.info(
            f"Use amp level {amp_level} for causal 3D VAE with dtype={dtype}, custom_fp32_cells {custom_fp32_cells}"
        )

        if use_discriminator:
            disc = auto_mixed_precision(disc, amp_level, dtype)
            logger.info(f"Use amp level {amp_level} for discriminator with dtype={dtype}")
    elif args.precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {args.precision}")

    # 3. build net with loss (core)
    # G with loss
    ae_with_loss = GeneratorWithLoss(
        ae,
        discriminator=disc,
        lpips_ckpt_path=os.path.join("pretrained", "lpips_vgg-426bf45c.ckpt"),
        disc_start=args.disc_start,
        disc_weight=args.disc_weight,
        kl_weight=args.kl_weight,
        logvar_init=args.logvar_init,
        perceptual_weight=args.perceptual_weight,
        loss_type=args.loss_type,
    )
    disc_start = args.disc_start

    # D with loss
    if use_discriminator:
        disc_with_loss = DiscriminatorWithLoss(ae, disc, disc_start, use_3d_disc=use_3d_disc)
        assert (
            not args.dataset_sink_mode
        ), "Training with gan loss does not support data sink mode! Please use --dataset_sink_mode False."

    tot_params, trainable_params = count_params(ae_with_loss)
    logger.info("Total params {:,}; Trainable params {:,}".format(tot_params, trainable_params))

    # 4. build dataset
    ds_config = dict(
        data_file_path=args.data_file_path,
        video_column=args.video_column,
        data_folder=args.video_path,
        size=args.resolution,
        crop_size=args.resolution,
        random_crop=args.random_crop,
        disable_flip=True,
        sample_stride=args.sample_rate,
        sample_n_frames=args.video_num_frames,
        return_image=False,
        dynamic_sample=args.dynamic_sample,
        output_columns=["video"],  # return video only, not the file path
    )
    split_time_upsample = True
    assert not (
        args.video_num_frames % 2 == 0 and split_time_upsample
    ), "num of frames must be odd if split_time_upsample is True"

    dataset = VideoDataset(**ds_config)
    train_loader = create_dataloader(
        dataset,
        shuffle=True,
        num_parallel_workers=args.dataloader_num_workers,
        batch_size=args.train_batch_size,
        drop_remainder=True,
        device_num=device_num,
        rank_id=rank_id,
        ds_name="video",
    )
    dataset_size = train_loader.get_dataset_size()

    # 5. build training utils
    # torch scale lr by: model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    if args.scale_lr:
        learning_rate = args.start_learning_rate * args.train_batch_size * args.gradient_accumulation_steps * device_num
        end_learning_rate = (
            args.end_learning_rate * args.train_batch_size * args.gradient_accumulation_steps * device_num
        )
    else:
        learning_rate = args.start_learning_rate
        end_learning_rate = args.end_learning_rate
    if args.dataset_sink_mode and args.sink_size != -1:
        assert args.sink_size > 0, f"Expect that sink size is a positive integer, but got {args.sink_size}"
        steps_per_sink = args.sink_size
    else:
        steps_per_sink = dataset_size

    if args.max_steps is not None:
        assert args.max_steps > 0, f"max_steps should a positive integer, but got {args.max_steps}"
        total_train_steps = args.max_steps
        args.epochs = math.ceil(total_train_steps / dataset_size)
    else:
        # use args.epochs
        assert (
            args.epochs is not None and args.epochs > 0
        ), f"When args.max_steps is not provided, args.epochs must be a positive integer! but got {args.epochs}"
        total_train_steps = args.epochs * dataset_size

    sink_epochs = math.ceil(total_train_steps / steps_per_sink)
    total_train_steps = sink_epochs * steps_per_sink
    if steps_per_sink == dataset_size:
        logger.info(
            f"Number of training steps: {total_train_steps}; Number of epochs: {args.epochs}; Number of batches in a epoch (dataset_size): {dataset_size}"
        )
    else:
        logger.info(
            f"Number of training steps: {total_train_steps}; Number of sink epochs: {sink_epochs}; Number of batches in a sink (sink_size): {steps_per_sink}"
        )

    if args.save_steps is None:
        ckpt_save_interval = args.ckpt_save_interval
        step_mode = False
        use_step_unit = False
    else:
        step_mode = not args.dataset_sink_mode
        use_step_unit = True
        if not args.dataset_sink_mode:
            ckpt_save_interval = args.save_steps
        else:
            # still need to count interval in sink epochs
            ckpt_save_interval = max(1, args.save_steps // steps_per_sink)
            if args.save_steps % steps_per_sink != 0:
                logger.warning(
                    f"`save_steps` must be times of sink size or dataset_size under dataset sink mode."
                    f"Checkpoint will be saved every {ckpt_save_interval * steps_per_sink} steps."
                )
    if step_mode != args.step_mode:
        logger.info("Using args.save_steps to determine whether to use step mode to save ckpt.")
        if args.save_steps is None:
            logger.warning(f"args.save_steps is not provided. Force step_mode to {step_mode}!")
        else:
            logger.warning(
                f"args.save_steps is provided. data sink mode is {args.dataset_sink_mode}. Force step mode to {step_mode}!"
            )
    logger.info(
        "ckpt_save_interval: {} {}".format(
            ckpt_save_interval, "steps" if (not args.dataset_sink_mode and step_mode) else "sink epochs"
        )
    )

    # build learning rate scheduler
    if not args.lr_decay_steps:
        args.lr_decay_steps = total_train_steps - args.lr_warmup_steps  # fix lr scheduling
        if args.lr_decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.lr_decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.lr_decay_steps = 1
    assert (
        args.lr_warmup_steps >= 0
    ), f"Expect args.lr_warmup_steps to be no less than zero,  but got {args.lr_warmup_steps}"

    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        name=args.lr_scheduler,
        lr=learning_rate,
        end_lr=end_learning_rate,
        warmup_steps=args.lr_warmup_steps,
        decay_steps=args.lr_decay_steps,
        total_steps=total_train_steps,
    )

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
    loss_scaler_ae = create_loss_scaler(args)

    if use_discriminator:
        optim_disc = create_optimizer(
            disc_with_loss.discriminator.trainable_params(),
            betas=args.betas,
            name=args.optim,
            lr=lr,  # since lr is a shared list
            group_strategy=args.group_strategy,
            weight_decay=args.weight_decay,
        )
        loss_scaler_disc = create_loss_scaler(args)
        scaling_sens_d = loss_scaler_disc.loss_scale_value

    assert args.ema_start_step == 0, "Now only support to update EMA from the first step"
    ema = EMA(ae_with_loss.autoencoder, ema_decay=args.ema_decay, offloading=args.ema_offload) if args.use_ema else None
    # resume training states
    # TODO: resume Discriminator if used
    ckpt_dir = os.path.join(args.output_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    start_epoch = 0
    if args.resume_from_checkpoint:
        resume_ckpt = (
            os.path.join(ckpt_dir, "train_resume.ckpt")
            if isinstance(args.resume_from_checkpoint, bool)
            else args.resume_from_checkpoint
        )

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            ae_with_loss, optim_ae, resume_ckpt
        )
        loss_scaler_ae.loss_scale_value = loss_scale
        loss_scaler_ae.cur_iter = cur_iter.to(ms.int32)
        loss_scaler_ae.last_overflow_iter = last_overflow_iter
        logger.info(f"Resume autoencoder training from {resume_ckpt}")
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

    if use_discriminator:
        if args.resume_from_checkpoint:
            resume_ckpt = (
                os.path.join(ckpt_dir, "train_resume_disc.ckpt")
                if isinstance(args.resume_from_checkpoint, bool)
                else args.resume_from_checkpoint
            )

            start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
                disc_with_loss, optim_disc, resume_ckpt
            )
            loss_scaler_disc.loss_scale_value = loss_scale
            loss_scaler_disc.cur_iter = cur_iter.to(ms.int32)
            loss_scaler_disc.last_overflow_iter = last_overflow_iter
            logger.info(f"Resume discriminator training from {resume_ckpt}")
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
                f"Jit level: {args.jit_level}",
                f"Distributed mode: {args.use_parallel}",
                f"Recompute: {args.use_recompute}",
                f"amp level: {amp_level}",
                f"dtype: {args.precision}",
                f"Use discriminator: {args.use_discriminator}",
                f"Learning rate: {learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Rescale size: {args.resolution}",
                f"Crop size: {args.resolution}",
                f"Number of frames: {args.video_num_frames}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num of training steps: {total_train_steps}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"EMA: {args.use_ema}",
                f"Dataset sink: {args.dataset_sink_mode}",
                f"Output dir: {args.output_dir}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

    # 6. training process
    if not use_discriminator:
        model = Model(training_step_ae)

        # callbacks
        callback = [TimeMonitor(args.log_interval)]
        ofm_cb = OverflowMonitor()
        callback.append(ofm_cb)

        if rank_id == 0:
            save_cb = EvalSaveCallback(
                network=ae_with_loss.autoencoder,
                rank_id=rank_id,
                ckpt_save_dir=ckpt_dir,
                ema=ema,
                ckpt_save_policy="latest_k",
                ckpt_max_keep=args.ckpt_max_keep,
                step_mode=step_mode,
                use_step_unit=use_step_unit,
                ckpt_save_interval=ckpt_save_interval,
                log_interval=args.log_interval,
                start_epoch=start_epoch,
                model_name="vae_3d",
                record_lr=False,
                save_training_resume=args.save_training_resume,
            )
            callback.append(save_cb)
            if args.profile:
                callback.append(ProfilerCallback())

            logger.info("Start training...")
            if args.config is not None and len(args.config) > 0:
                # backup config files
                shutil.copyfile(args.config, os.path.join(args.output_dir, os.path.basename(args.config)))

            with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
                yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)

        model.train(
            sink_epochs,
            train_loader,
            callbacks=callback,
            dataset_sink_mode=args.dataset_sink_mode,
            sink_size=args.sink_size,
            initial_epoch=start_epoch,
        )
    else:
        if not os.path.exists(f"{args.output_dir}/rank_{rank_id}"):
            os.makedirs(f"{args.output_dir}/rank_{rank_id}")
        loss_log_file = open(f"{args.output_dir}/rank_{rank_id}/result.log", "w")
        loss_log_file.write("step\tloss_ae\tloss_disc\ttrain_time(s)\n")
        loss_log_file.flush()
        if rank_id == 0:
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
        # output_numpy=True ?
        ds_iter = train_loader.create_dict_iterator(args.epochs - start_epoch)

        for epoch in range(start_epoch, args.epochs):
            start_time_e = time.time()
            for step, data in enumerate(ds_iter):
                start_time_s = time.time()
                x = data["video"]

                global_step = epoch * dataset_size + step
                global_step = ms.Tensor(global_step, dtype=ms.int64)

                # NOTE: inputs must match the order in GeneratorWithLoss.construct
                loss_ae_t, overflow, scaling_sens = training_step_ae(x, global_step)

                if global_step >= disc_start:
                    loss_disc_t, overflow_d, scaling_sens_d = training_step_disc(x, global_step)

                cur_global_step = epoch * dataset_size + step + 1  # starting from 1 for logging
                if overflow:
                    logger.warning(
                        f"Overflow occurs in step {cur_global_step} in autoencoder"
                        + (", drop update." if args.drop_overflow_update else ", still update.")
                    )
                if global_step >= disc_start and overflow_d:
                    logger.warning(
                        f"Overflow occurs in step {cur_global_step} in discriminator"
                        + (", drop update." if args.drop_overflow_update else ", still update.")
                    )

                # log
                step_time = time.time() - start_time_s
                if step % args.log_interval == 0:
                    loss_ae = float(loss_ae_t.asnumpy())
                    logger.info(
                        f"E: {epoch+1}, S: {step+1}, Loss ae: {loss_ae:.4f}, ae loss scaler {scaling_sens},"
                        + f" Step time: {step_time*1000:.2f}ms"
                    )
                    if global_step >= disc_start:
                        loss_disc = float(loss_disc_t.asnumpy())
                        logger.info(f"Loss disc: {loss_disc:.4f}, disc loss scaler {scaling_sens_d}")
                        loss_log_file.write(f"{cur_global_step}\t{loss_ae:.7f}\t{loss_disc:.7f}\t{step_time:.2f}\n")
                    else:
                        loss_log_file.write(f"{cur_global_step}\t{loss_ae:.7f}\t{0.0}\t{step_time:.2f}\n")
                    loss_log_file.flush()

                if rank_id == 0 and step_mode:
                    cur_epoch = epoch + 1
                    if (cur_global_step % ckpt_save_interval == 0) or (cur_global_step == total_train_steps):
                        ckpt_name = (
                            f"vae_3d-e{cur_epoch}.ckpt" if not use_step_unit else f"vae_3d-s{cur_global_step}.ckpt"
                        )
                        if ema is not None:
                            ema.swap_before_eval()
                        ae_with_loss.set_train(False)
                        disc_with_loss.set_train(False)
                        ckpt_manager.save(ae_with_loss.autoencoder, None, ckpt_name=ckpt_name, append_dict=None)
                        if args.save_training_resume:
                            ms.save_checkpoint(
                                training_step_ae,
                                os.path.join(ckpt_dir, "train_resume.ckpt"),
                                append_dict={
                                    "epoch_num": cur_epoch - 1,
                                    "loss_scale": scaling_sens,
                                },
                            )
                            ms.save_checkpoint(
                                training_step_disc,
                                os.path.join(ckpt_dir, "train_resume_disc.ckpt"),
                                append_dict={
                                    "epoch_num": cur_epoch - 1,
                                    "loss_scale": scaling_sens_d,
                                },
                            )
                        if ema is not None:
                            ema.swap_after_eval()
                        ae_with_loss.set_train(True)
                        disc_with_loss.set_train(True)

                if cur_global_step == total_train_steps:
                    break

            epoch_cost = time.time() - start_time_e
            per_step_time = epoch_cost / dataset_size
            cur_epoch = epoch + 1
            logger.info(
                f"Epoch:[{int(cur_epoch):>3d}/{int(args.epochs):>3d}], "
                f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time*1000:.2f}ms, "
            )

            if rank_id == 0 and not step_mode:
                if (cur_epoch % ckpt_save_interval == 0) or (cur_epoch == args.epochs):
                    ckpt_name = f"vae_3d-e{cur_epoch}.ckpt" if not use_step_unit else f"vae_3d-s{cur_global_step}.ckpt"
                    if ema is not None:
                        ema.swap_before_eval()
                    ae_with_loss.set_train(False)
                    disc_with_loss.set_train(False)
                    ckpt_manager.save(ae_with_loss.autoencoder, None, ckpt_name=ckpt_name, append_dict=None)
                    if args.save_training_resume:
                        ms.save_checkpoint(
                            training_step_ae,
                            os.path.join(ckpt_dir, "train_resume.ckpt"),
                            append_dict={
                                "epoch_num": cur_epoch - 1,
                                "loss_scale": scaling_sens,
                            },
                        )
                        ms.save_checkpoint(
                            training_step_disc,
                            os.path.join(ckpt_dir, "train_resume_disc.ckpt"),
                            append_dict={
                                "epoch_num": cur_epoch - 1,
                                "loss_scale": scaling_sens_d,
                            },
                        )
                    if ema is not None:
                        ema.swap_after_eval()
                    ae_with_loss.set_train(True)
                    disc_with_loss.set_train(True)

            if cur_global_step == total_train_steps:
                break
            # TODO: eval while training
        loss_log_file.close()


def parse_causalvae_train_args(parser):
    parser.add_argument(
        "--use_discriminator",
        default=False,
        type=str2bool,
        help="Whether to use the discriminator in the training process. "
        "Phase 1 training does not use discriminator, set False to reduce memory cost in graph mode.",
    )

    parser.add_argument(
        "--model_config",
        default="scripts/causalvae/release.json",
        help="the default model configuration file for the causalvae.",
    )
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=True,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to True in training mode.",
    )
    parser.add_argument(
        "--output_dir", default="results/causalvae", help="The directory where training results are saved."
    )
    parser.add_argument("--exp_name", default=None, help="The name of the experiment.")
    parser.add_argument("--max_steps", default=None, type=int, help="The maximum number of training steps.")
    parser.add_argument("--save_steps", default=None, type=int, help="The interval steps to save checkpoints.")

    parser.add_argument(
        "--video_path", default="/remote-home1/dataset/data_split_tt", help="The path where the video data is stored."
    )
    parser.add_argument(
        "--data_file_path",
        default=None,
        help="The data file path where the video paths are recorded. Now support json and csv file"
        "If not provided, will search all videos under `video_path` in a recursive manner.",
    )
    parser.add_argument(
        "--video_column",
        default="video",
        help="The column of video file path in `data_file_path`. Defaults to `video`.",
    )

    parser.add_argument("--video_num_frames", default=17, type=int, help="The number of frames per video.")

    parser.add_argument("--sample_rate", default=1, type=int, help="The sampling interval.")

    parser.add_argument("--dynamic_sample", default=False, type=str2bool, help="Whether to use dynamic sampling.")

    parser.add_argument("--ae", default="CausalVAEModel_4x8x8", help="The name of the causal vae model.")

    parser.add_argument("--resolution", default=256, type=int, help="The resolution of the videos.")

    parser.add_argument("--load_from_checkpoint", default=None, help="Load the model from a specified checkpoint.")
    parser.add_argument(
        "--random_crop", default=False, type=str2bool, help="Whether to use random crop. If False, use center crop"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument(
        "--save_training_resume", type=str2bool, default=True, help="Whether to save the training resume checkpoint."
    )
    parser.add_argument("--ema_start_step", type=int, default=0)

    parser.add_argument(
        "--disc_cls",
        type=str,
        default="causalvideovae.model.losses.LPIPSWithDiscriminator3D",
        help="",
    )
    parser.add_argument("--disc_start", type=int, default=5, help="")
    parser.add_argument("--disc_weight", type=float, default=0.5, help="")
    parser.add_argument("--kl_weight", type=float, default=1e-06, help="")
    parser.add_argument("--perceptual_weight", type=float, default=1.0, help="")
    parser.add_argument("--loss_type", type=str, default="l1", help="")
    parser.add_argument("--logvar_init", type=float, default=0.0, help="")
    return parser


if __name__ == "__main__":
    args = parse_args(additional_parse_args=parse_causalvae_train_args)
    if args.resume_from_checkpoint == "True":
        args.resume_from_checkpoint = True
    main(args)
