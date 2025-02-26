import logging
import os
import sys
import time
from typing import Dict, Union

from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import path_type

import mindspore as ms
import mindspore.dataset as ds
from mindspore import GRAPH_MODE, Model, get_context, nn, set_context, set_seed

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, ".."))
from hyvideo.acceleration import create_parallel_group
from hyvideo.constants import PRECISION_TO_TYPE
from hyvideo.dataset import VideoDataset
from hyvideo.utils import EMA, init_model, resume_train_net
from hyvideo.utils.helpers import set_modules_requires_grad, set_train
from hyvideo.vae import load_vae_train
from hyvideo.vae.losses import DiscriminatorWithLoss, GeneratorWithLoss, NLayerDiscriminator3D

from mindone.data import create_dataloader
from mindone.trainers import create_optimizer, create_scheduler
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback, StopAtStepCallback
from mindone.trainers.checkpoint import CheckpointManager
from mindone.trainers.zero import prepare_train_network
from mindone.utils import count_params, init_train_env, set_logger

logger = logging.getLogger(__name__)


def initialize_dataset(dataset_args, dataloader_args, device_num: int, shard_rank_id: int):
    dataset = VideoDataset(**dataset_args)
    dataloader_args = dataloader_args.as_dict()

    dataloader = create_dataloader(
        dataset=dataset,
        device_num=device_num,
        rank_id=shard_rank_id,
        **dataloader_args,
    )

    return dataloader, len(dataset)


def create_train_network(vae, args, dtype):
    use_discriminator = args.train.losses.disc_weight > 0
    if use_discriminator:
        disc = NLayerDiscriminator3D()
        if dtype != ms.float32:
            amp_level = "O2"
            disc = ms.amp.auto_mixed_precision(disc, amp_level=amp_level, dtype=dtype)
            logger.info(f"Set discriminator mixed precision to {amp_level} with dtype={dtype}")
    else:
        disc = None
    ae_with_loss = GeneratorWithLoss(
        vae,
        discriminator=disc,
        **args.train.losses,
    )
    disc_start = args.train.losses.disc_start

    # D with loss
    if use_discriminator:
        disc_with_loss = DiscriminatorWithLoss(vae, disc, disc_start)
    else:
        disc_with_loss = None
    return ae_with_loss, disc_with_loss


def main(args):
    # 1. init env
    args.train.output_path = os.path.abspath(args.train.output_path)
    os.makedirs(args.train.output_path, exist_ok=True)
    device_id, rank_id, device_num = init_train_env(**args.env)
    mode = get_context("mode")  # `init_train_env()` may change the mode during debugging
    # if graph mode and vae tiling is ON, uise dfs exec order
    if mode == GRAPH_MODE and args.vae.tiling:
        set_context(exec_order="dfs")
    # 1.1 init model parallel
    shard_rank_id = rank_id
    if args.train.sequence_parallel.shards > 1:
        create_parallel_group(**args.train.sequence_parallel)
        device_num = device_num // args.train.sequence_parallel.shards
        shard_rank_id = rank_id // args.train.sequence_parallel.shards

    # FIXME: Improve seed setting
    set_seed(args.env.seed + shard_rank_id)  # set different seeds per NPU for sampling different timesteps
    ds.set_seed(args.env.seed)  # keep MS.dataset's seed consistent as datasets first shuffled and then distributed

    set_logger("", output_dir=args.train.output_path, rank=rank_id)

    # instantiate classes only after initializing training environment
    initializer = parser.instantiate_classes(cfg)

    # 2. model initialize and weight loading
    # 2.1 vae
    logger.info("Initializing vae...")
    assert args.vae.trainable, "Expect vae to be trainable"
    vae, _, s_ratio, t_ratio = load_vae_train(
        logger=logger,
        **args.vae,
    )
    use_recompute = args.vae.factor_kwargs.get("use_recompute", False)
    # vae_kwargs = {"s_ratio": s_ratio, "t_ratio": t_ratio}
    vae_dtype = PRECISION_TO_TYPE[args.vae.precision]
    sample_n_frames = args.dataset.sample_n_frames
    if (sample_n_frames - 1) % 4 != 0:
        raise ValueError(f"`sample_n_frames - 1` must be a multiple of 4, got {sample_n_frames}")

    # 3. build training network
    ae_with_loss, disc_with_loss = create_train_network(vae, args, dtype=vae_dtype)

    if disc_with_loss is not None and args.train.data_sink_mode:
        logger.info("Data sink is not supported when training with discriminator. `data_sink_mode` is set to False")
        args.train.data_sink_mode = False
    # 4. build train & val datasets
    if args.train.sequence_parallel.shards > 1:
        logger.info(
            f"Initializing the dataloader: assigning shard ID {shard_rank_id} out of {device_num} total shards."
        )
    dataloader, dataset_len = initialize_dataset(args.dataset, args.dataloader, device_num, shard_rank_id)

    dataset_size = dataloader.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")

    # 5. build training utils: lr, optim, callbacks, trainer
    # 5.1 LR
    lr = create_scheduler(steps_per_epoch=0, **args.train.lr_scheduler)
    # build optimizer
    learn_logvar = args.train.losses.learn_logvar
    if learn_logvar:
        ae_params_to_update = [ae_with_loss.autoencoder.trainable_params(), ae_with_loss.logvar]
        ae_modules_to_update = [ae_with_loss.autoencoder, ae_with_loss.logvar]
    else:
        ae_params_to_update = ae_with_loss.autoencoder.trainable_params()
        ae_modules_to_update = [ae_with_loss.autoencoder]
    optim_ae = create_optimizer(ae_params_to_update, lr=lr, **args.train.optimizer_ae)
    loss_scaler_ae = initializer.train.loss_scaler_ae
    scaling_sens = loss_scaler_ae.loss_scale_value

    if disc_with_loss is not None:
        optim_disc = create_optimizer(
            disc_with_loss.discriminator.trainable_params(), lr=lr, **args.train.optimizer_disc
        )
        loss_scaler_disc = initializer.train.loss_scaler_disc
        scaling_sens_d = loss_scaler_disc.loss_scale_value

    # 5.3 trainer (standalone and distributed)
    ema = EMA(ae_with_loss.autoencoder, **args.train.ema) if args.train.ema else None

    training_step_ae = prepare_train_network(
        ae_with_loss, optimizer=optim_ae, scale_sense=loss_scaler_ae, ema=ema, **args.train.settings
    )

    if disc_with_loss is not None:
        training_step_disc = prepare_train_network(
            disc_with_loss,
            optimizer=optim_disc,
            scale_sense=loss_scaler_disc,
            ema=None,  # No ema for disriminator
            **args.train.settings,
        )

    start_epoch, global_step = 0, 0
    if args.train.resume_ckpt is not None:
        print("Not support resume now")
        start_epoch, global_step = resume_train_net(
            training_step_ae, resume_ckpt=os.path.abspath(args.train.resume_ckpt)
        )

    # 5.4 print out key info and save config
    if rank_id == 0:
        num_params_vae, num_params_trainable_vae = count_params(vae)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {mode}",
                f"Debug mode: {args.env.debug}",
                f"JIT level: {args.env.jit_level}",
                f"Distributed mode: {args.env.distributed}",
                f"Data path: {args.dataset.data_file_path}",
                f"Number of samples: {dataset_len}",
                f"vae dtype: {vae_dtype}",
                f"Num trainable params (Num params): {num_params_trainable_vae} ({num_params_vae})",
                f"Learning rate: {args.train.lr_scheduler.lr:.0e}",
                f"Batch size: {args.dataloader.batch_size}",
                f"Image size: {args.dataset.crop_size}",
                f"Frames: {args.dataset.sample_n_frames}",
                f"Use discriminator: {disc_with_loss is not None}",
                f"Weight decay ae: {args.train.optimizer_ae.weight_decay}"
                + (
                    f"\nWeight decay disc: {args.train.optimizer_disc.weight_decay}"
                    if disc_with_loss is not None
                    else ""
                ),
                f"Grad accumulation steps: {args.train.settings.gradient_accumulation_steps}",
                f"Number of training steps: {args.train.steps}",
                f"Loss scaler ae: {args.train.loss_scaler_ae.class_path}"
                + (
                    f"\nLoss scaler disc: {args.train.loss_scaler_disc.class_path}"
                    if disc_with_loss is not None
                    else ""
                ),
                f"Init loss scale ae: {args.train.loss_scaler_ae.init_args.loss_scale_value}"
                + (
                    f"\nInit loss scale disc: {args.train.loss_scaler_disc.init_args.loss_scale_value}"
                    if disc_with_loss is not None
                    else ""
                ),
                f"Grad clipping: {args.train.settings.clip_grad}",
                f"Max grad norm: {args.train.settings.clip_norm}",
                f"EMA: {ema is not None}",
                f"Use recompute: {use_recompute}",
            ]
        )
        key_info += "\n" + "=" * 50
        print(key_info)
        parser.save(args, args.train.output_path + "/config.yaml", format="yaml", overwrite=True)

    # 5.5 callbacks
    callbacks = [OverflowMonitor()]

    if args.train.settings.zero_stage == 3 or rank_id == 0:
        ckpt_save_dir = (
            os.path.join(args.train.output_path, f"rank_{rank_id}/ckpt")
            if args.train.settings.zero_stage == 3
            else os.path.join(args.train.output_path, "ckpt")
        )
        save_kwargs = args.train.save.as_dict()
        log_interval = save_kwargs.get("log_interval", 1)
        if args.train.data_sink_mode:
            if args.train.data_sink_size == -1:
                sink_size = len(dataloader)
            else:
                sink_size = args.train.data_sink_size
            new_log_interval = sink_size * log_interval
            if new_log_interval != log_interval:
                logger.info(
                    f"Because of data sink mode ON and sink size {sink_size}, log_interval is changed from {log_interval} to {new_log_interval}"
                )
            log_interval = new_log_interval
        save_kwargs["log_interval"] = log_interval

        callbacks.append(
            EvalSaveCallback(
                network=ae_with_loss.autoencoder,
                model_name=args.vae.type.replace("/", "-"),
                rank_id=0 if args.train.settings.zero_stage == 3 else rank_id,  # ZeRO-3 shards across all ranks
                ckpt_save_dir=ckpt_save_dir,
                ema=ema,
                step_mode=True,
                use_step_unit=True,
                start_epoch=start_epoch,
                resume_prefix_blacklist=("vae.", "swap."),
                train_steps=args.train.steps,
                **save_kwargs,
            )
        )

    callbacks.append(StopAtStepCallback(train_steps=args.train.steps, global_step=global_step))
    if args.profile:
        callbacks.append(ProfilerCallback(start_step=2, end_step=3, out_dir="./profile_data"))
    # 6. train
    logger.info("Start training...")
    # 6. training process
    if disc_with_loss is None:
        model = Model(training_step_ae)
        model.train(
            args.train.steps,
            dataloader,
            callbacks=callbacks,
            initial_epoch=start_epoch,
            dataset_sink_mode=args.train.data_sink_mode,
            sink_size=args.train.data_sink_size,
        )
    else:
        step_mode = True
        disc_start = args.train.losses.disc_start
        if not os.path.exists(f"{args.train.output_path}/rank_{rank_id}"):
            os.makedirs(f"{args.train.output_path}/rank_{rank_id}")
        if args.train.resume_ckpt is not None and os.path.exists(f"{args.train.output_path}/rank_{rank_id}/result.log"):
            # resume the loss log if it exists
            loss_log_file = open(f"{args.train.output_path}/rank_{rank_id}/result.log", "a")
        else:
            loss_log_file = open(f"{args.train.output_path}/rank_{rank_id}/result.log", "w")
            loss_log_file.write("step\tloss_ae\tloss_disc\ttrain_time(s)\n")
            loss_log_file.flush()
        ckpt_dir = os.path.join(args.train.output_path, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)
        if rank_id == 0:
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.train.save.ckpt_max_keep)
        total_epochs = args.train.steps // dataset_size + 1
        ds_iter = dataloader.create_dict_iterator(total_epochs - start_epoch)

        for epoch in range(start_epoch, total_epochs):
            start_time_e = time.time()
            set_train(ae_modules_to_update)
            for step, data in enumerate(ds_iter):
                start_time_s = time.time()
                x = data["video"]

                global_step = epoch * dataset_size + step

                if global_step % 2 == 1 and global_step >= disc_start:
                    set_modules_requires_grad(ae_modules_to_update, False)
                    step_gen = False
                    step_dis = True
                else:
                    set_modules_requires_grad(ae_modules_to_update, True)
                    step_gen = True
                    step_dis = False
                assert step_gen or step_dis, "You should backward either Gen or Dis in a step."

                global_step = ms.Tensor(global_step, dtype=ms.int64)
                cur_global_step = epoch * dataset_size + step + 1  # starting from 1 for logging
                # Generator Step
                if step_gen:
                    # NOTE: inputs must match the order in GeneratorWithLoss.construct
                    loss_ae_t, overflow, scaling_sens = training_step_ae(x, global_step)
                    if isinstance(scaling_sens, ms.Parameter):
                        scaling_sens = scaling_sens.value()

                    if overflow:
                        logger.warning(
                            f"Overflow occurs in step {cur_global_step} in autoencoder"
                            + (", drop update." if args.train.settings.drop_overflow_update else ", still update.")
                        )
                # Discriminator Step
                if step_dis:
                    if global_step >= disc_start:
                        loss_disc_t, overflow_d, scaling_sens_d = training_step_disc(x, global_step)
                        if isinstance(scaling_sens_d, ms.Parameter):
                            scaling_sens_d = scaling_sens_d.value()
                        if overflow_d:
                            logger.warning(
                                f"Overflow occurs in step {cur_global_step} in discriminator"
                                + (", drop update." if args.train.settings.drop_overflow_update else ", still update.")
                            )
                # log
                step_time = time.time() - start_time_s
                if step % args.train.save.log_interval == 0:
                    if step_gen:
                        loss_ae = float(loss_ae_t.asnumpy())
                        logger.info(
                            f"E: {epoch+1}, S: {step+1}, Loss ae: {loss_ae:.4f}, ae loss scaler {scaling_sens},"
                            + f" Step time: {step_time*1000:.2f}ms"
                        )
                        loss_disc = -1  # no discriminator loss, dummy value
                    if step_dis and global_step >= disc_start:
                        loss_disc = float(loss_disc_t.asnumpy())
                        logger.info(
                            f"E: {epoch+1}, S: {step+1}, Loss disc: {loss_disc:.4f}, disc loss scaler {scaling_sens_d},"
                            + f" Step time: {step_time*1000:.2f}ms"
                        )
                        loss_ae = -1  # no generator loss, dummy value

                    loss_log_file.write(f"{cur_global_step}\t{loss_ae:.7f}\t{loss_disc:.7f}\t{step_time:.2f}\n")
                    loss_log_file.flush()

                if rank_id == 0 and step_mode:
                    cur_epoch = epoch + 1
                    if (cur_global_step % args.train.save.ckpt_save_interval == 0) or (
                        cur_global_step == args.train.steps
                    ):
                        ae_with_loss.set_train(False)
                        disc_with_loss.set_train(False)
                        ckpt_name = f"{args.vae.type}-s{cur_global_step}.ckpt"
                        if not args.train.save.save_ema_only and ema is not None:
                            ckpt_manager.save(
                                ae_with_loss.autoencoder,
                                None,
                                ckpt_name=ckpt_name.replace(".ckpt", "_nonema.ckpt"),
                                append_dict=None,
                            )

                        if ema is not None:
                            ema.swap_before_eval()

                        ckpt_manager.save(ae_with_loss.autoencoder, None, ckpt_name=ckpt_name, append_dict=None)
                        if args.train.save.save_training_resume:
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

                if cur_global_step == args.train.steps:
                    break

            epoch_cost = time.time() - start_time_e
            per_step_time = epoch_cost / dataset_size
            cur_epoch = epoch + 1
            logger.info(
                f"Epoch:[{int(cur_epoch):>3d}/{int(total_epochs):>3d}], "
                f"epoch time:{epoch_cost:.2f}s, per step time:{per_step_time*1000:.2f}ms, "
            )

            if rank_id == 0 and not step_mode:
                if (cur_epoch % args.train.save.ckpt_save_interval == 0) or (cur_epoch == total_epochs):
                    ae_with_loss.set_train(False)
                    disc_with_loss.set_train(False)
                    ckpt_name = f"{args.vae.type}-s{cur_global_step}.ckpt"
                    if not args.train.save.save_ema_only and ema is not None:
                        ckpt_manager.save(
                            ae_with_loss.autoencoder,
                            None,
                            ckpt_name=ckpt_name.replace(".ckpt", "_nonema.ckpt"),
                            append_dict=None,
                        )
                    if ema is not None:
                        ema.swap_before_eval()
                    ckpt_manager.save(ae_with_loss.autoencoder, None, ckpt_name=ckpt_name, append_dict=None)
                    if args.train.save.save_training_resume:
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

            if cur_global_step == args.train.steps:
                break
            # TODO: eval while training
        loss_log_file.close()


if __name__ == "__main__":
    parser = ArgumentParser(description="Hunyuan Video training script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_train_env, "env")
    parser.add_function_arguments(init_model, "model", skip={"resume"})
    parser.add_function_arguments(load_vae_train, "vae", skip={"logger"})
    parser.add_class_arguments(VideoDataset, "dataset", instantiate=False)
    parser.add_function_arguments(
        create_dataloader,
        "dataloader",
        skip={"dataset", "batch_size", "transforms", "batch_transforms", "device_num", "rank_id"},
    )
    parser.add_argument(  # FIXME: support bucketing
        "--dataloader.batch_size", default=1, type=Union[int, Dict[str, int]], help="Number of samples per batch"
    )
    parser.add_argument("--profile", default=False, type=bool, help="Profile time analysis or not")
    parser.link_arguments("env.debug", "dataloader.debug", apply_on="parse")
    parser.add_function_arguments(create_parallel_group, "train.sequence_parallel")
    parser.add_function_arguments(create_scheduler, "train.lr_scheduler", skip={"steps_per_epoch", "num_epochs"})

    parser.add_function_arguments(create_optimizer, "train.optimizer_ae", skip={"params", "lr"})
    parser.add_function_arguments(create_optimizer, "train.optimizer_disc", skip={"params", "lr"})
    parser.add_subclass_arguments(
        nn.Cell,
        "train.loss_scaler_ae",
        fail_untyped=False,  # no typing in mindspore
        help="mindspore.nn.FixedLossScaleUpdateCell or mindspore.nn.DynamicLossScaleUpdateCell",
    )
    parser.add_subclass_arguments(
        nn.Cell,
        "train.loss_scaler_disc",
        fail_untyped=False,  # no typing in mindspore
        help="mindspore.nn.FixedLossScaleUpdateCell or mindspore.nn.DynamicLossScaleUpdateCell",
    )
    parser.add_function_arguments(
        prepare_train_network, "train.settings", skip={"network", "optimizer", "scale_sense", "ema"}
    )
    parser.add_subclass_arguments(EMA, "train.ema", skip={"network"}, required=False, instantiate=False)
    parser.add_class_arguments(
        GeneratorWithLoss, "train.losses", skip={"autoencoder", "discriminator"}, instantiate=False
    )
    parser.add_function_arguments(resume_train_net, "train", skip={"train_net"})
    parser.add_argument(
        "--train.output_path",
        default="output/",
        type=path_type("dcc"),  # path to a directory that can be created if it does not exist
        help="Output directory to save training results.",
    )
    parser.add_argument("--train.steps", default=100, type=int, help="Number of steps to train. Default: 100.")
    parser.link_arguments("train.steps", "train.lr_scheduler.total_steps", apply_on="parse")
    parser.add_argument("--train.data_sink_mode", default=False, type=bool, help="Whether to turn on data sink mode.")
    parser.add_argument("--train.data_sink_size", default=-1, type=int, help="The data sink size when sink mode is ON.")
    parser.add_class_arguments(
        EvalSaveCallback,
        "train.save",
        skip={
            "network",
            "rank_id",
            "shard_rank_id",
            "ckpt_save_dir",
            "output_dir",
            "ema",
            "start_epoch",
            "model_name",
            "step_mode",
            "use_step_unit",
            "train_steps",
            "resume_prefix_blacklist",
        },
        instantiate=False,
    )

    cfg = parser.parse_args()
    main(cfg)
