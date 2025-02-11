import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "../../../")))
import time
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

import mindspore as ms
from mindspore import nn

import mindone.models.threestudio as threestudio
from mindone.models.threestudio.systems.base import BaseSystem
from mindone.trainers.checkpoint import CheckpointManager
from mindone.trainers.recorder import PerfRecorder
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params, load_param_into_net_with_filter
from mindone.utils.seed import set_random_seed


def launch(args, extras) -> None:
    # step 1: init env & model, setup log dir by whether you resume or not
    set_random_seed(args.seed)
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(extras)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    train_cfg = cfg.train_cfg

    _, rank_id, device_num = init_train_env(mode=args.mode, seed=args.seed, debug=args.debug)

    if not args.debug:
        if cfg.resume is not None:
            # assume that the ckpt under xx/ckpt/xx.ckpt
            output_dir = Path("/".join(cfg.resume.split("/")[:-2]))
        else:
            output_dir = Path(cfg.exp_root_dir) / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + cfg.run_suffix)
            # only new training mkdir
            output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = cfg.exp_root_dir

    if args.train_highres:
        cfg.data.update({"batch_size": 4})
        cfg.data.update({"width": 256})
        cfg.data.update({"height": 256})
        global_step = 5000
    else:
        train_cfg.params.update({"max_steps": 5000})  # for lowres only train to s5k

    # put name as ""
    logger = set_logger(name="", output_dir=str(output_dir) if not args.debug else None, rank=rank_id)
    system: BaseSystem = threestudio.find(cfg.system_type)(
        cfg.system, resumed=cfg.resume is not None, train_highres=args.train_highres
    )
    system.set_save_dir(os.path.join(output_dir, cfg.trial_dir, "save"))
    if train_cfg.params.precision == "16-mixed":
        system = auto_mixed_precision(system, amp_level=train_cfg.params.amp_level)

    # resume state if needed
    if cfg.resume:
        _mode = "train" if args.train else "test"
        global_step, global_epoch, loss_scale, state_dict = get_resume_states(cfg.resume)
        logger.info(f"Resumed loss_scaler, prev epoch: {global_epoch}, global step {global_step}")
        m1, u1 = load_param_into_net_with_filter(system.renderer.geometry, state_dict)  # missing and unexpected keys
        m2, u2 = load_param_into_net_with_filter(system.renderer.background, state_dict)
        del state_dict
        m = set(m1).union(set(m2))
        u = set(u1).intersection(set(u2))
        logger.info(f"Resumed ckpt {cfg.resume} in {_mode} mode")
        logger.info(f"missing keys {m}")
        logger.info(f"unexpected keys {u}")
    else:
        global_step = 0
        global_epoch = 0

    # step 2: prepare dataloader opt/sche/trainer
    dataset = threestudio.find(cfg.data_type)(cfg.data)

    # step 3.a: launch training
    if args.train:
        dataset.setup("train")
        dataset.train_dataset.update_step(global_epoch, global_step)  # update the train set into the correct one
        train_loader = ms.dataset.GeneratorDataset(
            dataset.train_dataset,
            column_names=dataset.train_dataset.output_columns,
            shuffle=False,
        )
        val_loader = ms.dataset.GeneratorDataset(
            dataset.val_dataset, column_names=dataset.val_dataset.output_columns, shuffle=False
        )

        # optim w/ group
        _weight_decay = cfg.system.optimizer.args.weight_decay
        _lr_params = cfg.system.optimizer.lr_params
        group_params = [
            {
                "params": system.renderer.geometry.encoding.trainable_params(),
                "weight_decay": _weight_decay,
                "lr": _lr_params.geometry_encoding,
            },
            {
                "params": system.renderer.geometry.density_network.trainable_params(),
                "weight_decay": _weight_decay,
                "lr": _lr_params.geometry_density_network,
            },
            {
                "params": system.renderer.geometry.feature_network.trainable_params(),
                "weight_decay": _weight_decay,
                "lr": _lr_params.geometry_feature_network,
            },
            {
                "params": system.renderer.background.trainable_params(),
                "weight_decay": _weight_decay,
                "lr": _lr_params.background,
            },
        ]

        optimizer = nn.optim.AdamWeightDecay(
            group_params,
            beta1=cfg.system.optimizer.args.betas[0],
            beta2=cfg.system.optimizer.args.betas[1],
            eps=cfg.system.optimizer.args.eps,
        )

        if args.loss_scaler_type == "dynamic":  # for the case when there is an overflow during training
            loss_scaler = nn.DynamicLossScaleUpdateCell(
                loss_scale_value=cfg.train_cfg.loss_scale.loss_scale_value,
                scale_factor=cfg.train_cfg.loss_scale.loss_scale_factor,
                scale_window=cfg.train_cfg.loss_scale.scale_window,
            )
        elif args.loss_scaler_type == "static":
            loss_scaler = nn.FixedLossScaleUpdateCell(cfg.train_cfg.loss_scale.loss_scale_value)
        else:
            loss_scaler = ms.Tensor([1.0], dtype=ms.float32)
        if cfg.resume:
            loss_scaler.loss_scale_value = loss_scale

        system.on_train_start()  # log the loss configs
        net_with_grads = TrainOneStepWrapper(
            system,
            optimizer=optimizer,
            scale_sense=loss_scaler,
            **cfg.train_cfg.settings,  # alignment: no clip grap & overflow handling related for now, but if amp not ok then needs to clip it and loss-scale it
        )

        if rank_id == 0:
            num_params, num_trainable_params = count_params(system)
            key_info = "Key Settings:\n" + "=" * 50 + "\n"
            key_info += "\n".join(
                [
                    f"Debugging: {args.debug}",
                    f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                    f"Num params mvdream-3D: {num_params,}",
                    f"Num trainable params: {num_trainable_params,}",
                    f"Num training steps: {train_cfg.params.max_steps}",
                    f"Batch size: {cfg.data.batch_size}",
                    f"Number of frames: {cfg.data.n_view}",
                    f"Weight decay: {cfg.system.optimizer.args.weight_decay}",
                ]
            )
            key_info += "\n" + "=" * 50
            logger.info(key_info)
            logger.info("Start training...")

        ckpt_dir = (
            os.path.join(output_dir, "ckpt") if args.assign_output_ckpt_path is None else args.assign_output_ckpt_path
        )
        ds_iter = train_loader.create_tuple_iterator(do_copy=False)  # this must use with batch
        if rank_id == 0:
            ckpt_manager = CheckpointManager(ckpt_dir, "latest_k", k=args.ckpt_max_keep)
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            perf_columns = ["step", "loss", "train_time(s)"]
            output_dir = ckpt_dir.replace("/ckpt", "")
            if global_step == 0:
                record = PerfRecorder(output_dir, metric_names=perf_columns)
            else:
                record = PerfRecorder(output_dir, resume=True)

        net_with_grads.set_train()  # set mode for training

        threestudio.info(f"start training, targetting max step {train_cfg.params.max_steps}")
        threestudio.info(f"start with global step {global_step + 1}")
        for global_step in range(global_step + 1, train_cfg.params.max_steps + 1):
            data = next(ds_iter)
            start_time_s = time.time()
            loss, overflow, scaling_sens = net_with_grads(*data)
            step_time = time.time() - start_time_s

            # update_step for each module before fetching data
            net_with_grads.network.global_step = global_step
            step_remainder = global_step % 4
            if step_remainder == 0:
                global_epoch += 1
            net_with_grads.network.current_epoch = global_epoch
            net_with_grads.network.on_train_batch_start([], global_step, dataset.train_dataset)

            loss_train = float(loss.asnumpy())
            logger.info(
                f"Global step {global_step}, loss {loss_train:.5f}. Epoch {global_epoch}, epoch step {step_remainder}"
                + f" Step time {step_time*1000:.2f}ms"
            )
            # logger.info('')
            if overflow:
                logger.warning("overflow detected")

            # save train state and ckpts when reaches the global step milestone
            if global_step % train_cfg.params.save_interval == 0 or global_step == train_cfg.params.max_steps:
                # validate during training
                for view_idx, batch in enumerate(val_loader):
                    net_with_grads.network.validation_step(batch, view_idx)

                # saving resume states & ckpts
                save_train_net_states(net_with_grads, ckpt_dir, global_epoch - 1, global_step)
                ckpt_name = f"step{global_step}.ckpt"
                save_ckpts(
                    list(
                        itertools.chain.from_iterable(
                            [
                                system.renderer.geometry.trainable_params(),
                                system.renderer.background.trainable_params(),
                            ]
                        )
                    ),
                    ckpt_manager,
                    ckpt_name,
                    append_dict={
                        "epoch_num": global_epoch - 1,
                        "cur_step": global_step,
                    },
                )

            # logs
            if rank_id == 0:
                step_pref_value = [global_step, loss_train, step_time]
                record.add(*step_pref_value)

    elif args.test:
        # step 3.b: launch testing
        # conduct testing: map-style dataloader
        system.set_resume_status_eval(1, global_step)  # it must have read ckpt to resume

        dataset.setup("test")
        test_loader = ms.dataset.GeneratorDataset(
            dataset.test_dataset, column_names=dataset.test_dataset.output_columns, shuffle=False
        )

        for view_idx, batch in enumerate(test_loader):  #
            start_time_s = time.time()
            system.test_step(batch, view_idx)
            step_time = time.time() - start_time_s
            logger.info(f"Testing idx {view_idx} done. Step time {step_time*1000:.2f}ms")
            system.on_test_batch_start(batch, global_step, test_loader)

        # save gif/mp4
        system.on_test_epoch_end()
    else:
        raise ValueError("mode not supported")


def save_train_net_states(train_net, ckpt_dir, epoch, global_step):
    """save train net & state for resuming"""
    # train_net: i.e. net_with_grads, contains optimizer, ema, sense_scale, etc.
    ms.save_checkpoint(
        train_net,
        os.path.join(ckpt_dir, "train_resume.ckpt"),
        append_dict={
            "epoch_num": epoch,
            "cur_step": global_step,
            "loss_scale": train_net.scale_sense.asnumpy().item(),
        },
    )


def get_resume_states(resume_ckpt):
    state_dict = ms.load_checkpoint(resume_ckpt)
    global_step = int(state_dict.pop("cur_step", ms.Tensor(0, ms.int32)).asnumpy().item())
    start_epoch = int(state_dict.pop("epoch_num", ms.Tensor(0, ms.int32)).asnumpy().item())
    loss_scale = float(state_dict.pop("loss_scale", ms.Tensor(0, ms.float32)).asnumpy().item())

    return global_step, start_epoch, loss_scale, state_dict


def save_ckpts(net, ckpt_manager, ckpt_name, append_dict):
    ckpt_manager.save(net, None, ckpt_name=ckpt_name, append_dict=append_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mvdream-sd21.yaml", help="path to config file")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When debugging, set it true. Dumping files will overlap to avoid trashing your storage.",
    )
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--seed", default=42, type=int, help="data path")
    parser.add_argument(
        "--loss_scaler_type", default=None, type=str, help="dynamic or static"  # loss scale only used in amp O1/O2
    )
    parser.add_argument("--train", action="store_true", help="mode")
    parser.add_argument("--test", action="store_true", help="mode")
    parser.add_argument("--assign_output_ckpt_path", default=None)
    parser.add_argument("--ckpt_max_keep", default=1, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument(
        "--train_highres", action="store_true", help="high for training with 256x256, original/low for 64x64"
    )
    args, extras = parser.parse_known_args()

    launch(args, extras)
