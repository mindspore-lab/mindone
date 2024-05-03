"""
Train AutoEncoders with GAN loss
"""
import logging
import os
import shutil
import sys
import time

import yaml
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Model
from mindspore.train.callback import TimeMonitor

sys.path.append(".")
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
from opensora.models.ae import getae_model_config
from opensora.models.ae.videobase.dataset_videobase import VideoDataset, create_dataloader
from opensora.models.ae.videobase.losses.net_with_loss import DiscriminatorWithLoss, GeneratorWithLoss
from opensora.train.commons import create_loss_scaler, init_env, parse_args
from opensora.utils.utils import get_precision

from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from mindone.trainers.checkpoint import CheckpointManager, resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config, str2bool
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
        enable_dvm=args.enable_dvm,
    )
    if args.exp_name is not None and len(args.exp_name) > 0:
        args.output_dir = os.path.join(args.output_dir, args.exp_name)
    set_logger(output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))

    # Load Config
    model_config = os.path.join("opensora/models/ae/videobase/causal_vae/", getae_model_config(args.ae))
    model_config = OmegaConf.load(model_config)
    ae = instantiate_from_config(model_config)
    if args.load_from_checkpoint is not None and len(args.load_from_chckpoint) > 0:
        ae.init_from_ckpt(args.load_from_checkpoint)
    else:
        logger.info("No pre-trained model is loaded.")

    # discriminator (D)
    use_discriminator = args.use_discriminator and (model_config.lossconfig.disc_weight > 0.0)

    if args.use_discriminator and (model_config.lossconfig.disc_weight <= 0.0):
        logging.warning("use_discriminator is True but disc_weight is 0.")

    if use_discriminator:
        disc = instantiate_from_config(model_config.discriminator)
    else:
        disc = None

    # mixed precision
    # TODO: set softmax, sigmoid computed in FP32. manually set inside network since they are ops, instead of layers whose precision will be set by AMP level.
    if args.precision != "fp32":
        amp_level = "O2"
        dtype = get_precision(args.precision)
        ae = auto_mixed_precision(ae, amp_level, dtype)
        if use_discriminator:
            disc = auto_mixed_precision(disc, amp_level, dtype)
        logger.info(f"Set mixed precision to O2 with dtype={args.precision}")
    else:
        amp_level = "O0"

    # 3. build net with loss (core)
    # G with loss
    ae_with_loss = GeneratorWithLoss(ae, discriminator=disc, **model_config.lossconfig)
    disc_start = model_config.lossconfig.disc_start

    # D with loss
    if use_discriminator:
        disc_with_loss = DiscriminatorWithLoss(ae, disc, disc_start)

    tot_params, trainable_params = count_params(ae_with_loss)
    logger.info("Total params {:,}; Trainable params {:,}".format(tot_params, trainable_params))

    # 4. build dataset
    ds_config = dict(
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
    assert not (
        args.video_num_frames % 2 == 0 and model_config.generator.params.ddconfig.split_time_upsample
    ), "num of frames must be odd if split_time_upsample is True"

    dataset = VideoDataset(**ds_config)
    train_loader = create_dataloader(
        dataset,
        shuffle=True,
        num_parallel_workers=args.num_parallel_workers,
        batch_size=args.batch_size,
        drop_remainder=True,
        device_num=device_num,
        rank_id=rank_id,
        ds_name="video",
    )
    num_batches = train_loader.get_dataset_size()

    # 5. build training utils
    # torch scale lr by: model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
    if args.scale_lr:
        learning_rate = args.start_learning_rate * args.batch_size * args.gradient_accumulation_steps * device_num
    else:
        learning_rate = args.start_learning_rate
    if args.max_steps is not None and args.max_steps > 0:
        args.epochs = args.max_steps // num_batches
        logger.info("max_steps is set, override epochs to {}".format(args.epochs))
    if args.save_steps is not None and args.save_steps > 0:
        args.step_mode = True  # use step mode to save ckpt
        args.ckpt_save_interval = args.save_steps
        logger.info("save_steps is set, override ckpt_save_interval to {}".format(args.ckpt_save_interval))

    if not args.lr_decay_steps:
        args.lr_decay_steps = max(1, args.epochs * num_batches - args.lr_warmup_steps)
    lr = create_scheduler(
        steps_per_epoch=num_batches,
        name=args.lr_scheduler,
        lr=learning_rate,
        end_lr=args.end_learning_rate,
        warmup_steps=args.lr_warmup_steps,
        decay_steps=args.lr_decay_steps,
        num_epochs=args.epochs,
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

    ema = (
        EMA(
            ae_with_loss.autoencoder,
            ema_decay=args.ema_decay,
        )
        if args.use_ema
        else None
    )

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
                f"dtype: {args.precision}",
                f"Data path: {args.video_path}",
                f"Learning rate: {learning_rate}",
                f"Batch size: {args.batch_size}",
                f"Rescale size: {args.resolution}",
                f"Crop size: {args.resolution}",
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
                step_mode=args.step_mode,
                ckpt_max_keep=args.ckpt_max_keep,
                ckpt_save_interval=args.ckpt_save_interval,
                log_interval=args.log_interval,
                start_epoch=start_epoch,
                model_name="vae_kl_f8",
                record_lr=False,
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
            args.epochs,
            train_loader,
            callbacks=callback,
            dataset_sink_mode=args.dataset_sink_mode,
            sink_size=args.sink_size,
            initial_epoch=start_epoch,
        )
    else:
        if rank_id == 0:
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
        # output_numpy=True ?
        ds_iter = train_loader.create_dict_iterator(args.epochs - start_epoch)

        for epoch in range(start_epoch, args.epochs):
            start_time_e = time.time()
            for step, data in enumerate(ds_iter):
                start_time_s = time.time()
                x = data["video"]

                global_step = epoch * num_batches + step
                global_step = ms.Tensor(global_step, dtype=ms.int64)

                # NOTE: inputs must match the order in GeneratorWithLoss.construct
                loss_ae_t, overflow, scaling_sens = training_step_ae(x, global_step)

                if global_step >= disc_start:
                    loss_disc_t, overflow_d, scaling_sens_d = training_step_disc(x, global_step)

                cur_global_step = epoch * num_batches + step + 1  # starting from 1 for logging
                if overflow:
                    logger.warning(f"Overflow occurs in step {cur_global_step}")

                # log
                step_time = time.time() - start_time_s
                if step % args.log_interval == 0:
                    loss_ae = float(loss_ae_t.asnumpy())
                    logger.info(f"E: {epoch+1}, S: {step+1}, Loss ae: {loss_ae:.4f}, Step time: {step_time*1000:.2f}ms")
                    if global_step >= disc_start:
                        loss_disc = float(loss_disc_t.asnumpy())
                        logger.info(f"Loss disc: {loss_disc:.4f}")

            epoch_cost = time.time() - start_time_e
            per_step_time = epoch_cost / num_batches
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


def parse_causalvae_train_args(parser):
    parser.add_argument(
        "--use_discriminator",
        default=False,
        type=str2bool,
        help="Whether to use the discriminator in the training process. "
        "Phase 1 training does not use discriminator, set False to reduce memory cost in graph mode.",
    )

    parser.add_argument(
        "--output_dir", default="results/causalvae", help="The directory where training results are saved."
    )
    parser.add_argument("--exp_name", default=None, help="The name of the experiment.")
    parser.add_argument("--max_steps", dafault=None, type=int, help="The maximum number of training steps.")
    parser.add_argument("--save_steps", default=None, type=int, help="The interval steps to save checkpoints.")

    parser.add_argument(
        "--video_path", default="/remote-home1/dataset/data_split_tt", help="The path where the video data is stored."
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
    return parser


if __name__ == "__main__":
    args = parse_args(additional_parse_args=parse_causalvae_train_args)
    main(args)
