import argparse
import datetime
import logging
import math
import os

import yaml
from utils.train_util import str2bool

import mindspore as ms
from mindspore import Model, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

logger = logging.getLogger("")

import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))  # for mindone

from model_stage1 import InstantMeshStage1WithLoss
from omegaconf import OmegaConf

from mindone.data import create_dataloader
from mindone.trainers.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallbackEpoch
from mindone.trainers.checkpoint import resume_train_network
from mindone.trainers.ema import EMA
from mindone.trainers.lr_schedule import create_scheduler
from mindone.trainers.optim import create_optimizer
from mindone.trainers.train_step import TrainOneStepWrapper
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import instantiate_from_config
from mindone.utils.env import init_train_env
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed


def parse_args(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser = parse_train_args(parser)
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="resume from checkpoint with path",
    )
    parser.add_argument(
        "--base",
        type=str,
        default="configs/instant-nerf-large-train.yaml",
        help="path to base configs",
    )
    parser.add_argument(
        "--log_interval",
        default=1,
        type=int,
        help="log interval in the unit of data sink size.. E.g. if data sink size = 10, log_inteval=2, log every 20 steps",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="When debugging, set it true. Dumping files will overlap to avoid trashing your storage.",
    )
    args = parser.parse_args()
    return args


def parse_train_args(parser):
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument(
        "--dtype",
        default="fp32",  # if amp level O0/1, must pass fp32
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what computation data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--global_bf16",
        default=False,
        help="Experimental. If True, dtype will be overrided, operators will be computered in bf16 if they are supported by CANN",
    )
    parser.add_argument(
        "--amp_level",
        default="O0",  # cannot amp for InstantMesh training, easily grad nan
        type=str,
        help="mindspore amp level, O1: most fp32, only layers in whitelist compute in fp16 (dense, conv, etc); \
                        O2: most fp16, only layers in blacklist compute in fp32 (batch norm etc)",
    )
    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs")
    parser.add_argument(
        "--profile",
        default=False,  # deactivate as profiler says NOT supporting PyNative
        type=str2bool,
        help="Profile or not",
    )
    parser.add_argument(
        "--loss_scaler_type", default=None, type=str, help="dynamic or static"  # loss scale only used in amp O1/O2
    )
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--ckpt_max_keep", default=5, type=int, help="Maximum number of checkpoints to keep")
    parser.add_argument("--output_path", default="outputs/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    parser.add_argument("--num_parallel_workers", default=12, type=int, help="num workers for data loading")
    parser.add_argument(
        "--data_multiprocessing",
        default=False,
        help="If True, use multiprocessing for data processing. Default: multithreading.",
    )
    parser.add_argument("--max_rowsize", default=64, type=int, help="max rowsize for data loading")
    parser.add_argument(
        "--train_steps", default=-1, type=int, help="If not -1, limit the number of training steps to the set value"
    )
    parser.add_argument(
        "--epochs",
        default=7000,
        type=int,
        help="epochs. If dataset_sink_mode is on, epochs is with respect to dataset sink size. Otherwise, it's w.r.t the dataset size.",
    )
    parser.add_argument(
        "--ckpt_save_steps",
        default=-1,
        type=int,
        help="save checkpoint every this steps. If -1, use ckpt_save_interval will be used.",
    )
    parser.add_argument("--step_mode", default=False, help="whether save ckpt by steps. If False, save ckpt by epochs.")
    # optimizer param
    parser.add_argument("--use_ema", default=False, help="whether use EMA")
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--clip_grad", default=True, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )
    parser.add_argument("--start_learning_rate", default=4e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=4e-5, type=float, help="The end learning rate for Adam.")
    parser.add_argument(
        "--decay_steps", default=9e3, type=int, help="lr decay steps."
    )  # with dataset_size==3, it's 3k epochs
    parser.add_argument("--scheduler", default="cosine_annealing_warm_restarts_lr", type=str, help="scheduler.")
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.9, 0.95],
        help="Specify the [beta1, beta2] parameter for the AdamW optimizer.",
    )
    parser.add_argument(
        "--optim_eps", type=float, default=1e-8, help="Specify the eps parameter for the AdamW optimizer."
    )
    parser.add_argument(
        "--group_strategy",
        type=str,
        default="not_grouping",
        help="Grouping strategy for weight decay. If `norm_and_bias`, weight decay filter list is [beta, gamma, bias]. \
                                If None, filter list is [layernorm, bias]. Default: norm_and_bias",
    )
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=42, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    # dataloader param
    parser.add_argument("--dataset_sink_mode", default=False, help="sink mode")
    parser.add_argument("--sink_size", default=-1, type=int, help="dataset sink size. If -1, sink size = dataset size.")

    return parser


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    if args.resume:
        args.output_path = args.resume
    elif not args.debug:
        args.output_path = os.path.join(args.output_path, time_str)
    else:
        print("make sure you are debugging now, as no ckpt will be saved.")

    # 1. init
    did, rank_id, device_num = init_train_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
        debug=args.debug,
    )
    set_random_seed(42)
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # load config yaml file
    config = OmegaConf.load(args.base)
    ckpt_dir = os.path.join(args.output_path, "ckpt")

    # 2. model initiate
    # 2.1 instantmesh model stage 1
    img_size = config.model.params.input_size
    config.model.params.lrm_generator_config.params.dtype = args.dtype

    lrm_model_with_loss = InstantMeshStage1WithLoss(config.model.params.lrm_generator_config)
    lrm_model_with_loss.set_train(True)

    if not args.global_bf16:
        lrm_model_with_loss.lrm_generator = auto_mixed_precision(
            lrm_model_with_loss.lrm_generator,
            amp_level=args.amp_level,
        )

    # 3. create dataset
    dataset = instantiate_from_config(config.data.train)
    nw = config.data.num_workers
    dataloader = create_dataloader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        device_num=device_num,
        rank_id=rank_id,
        num_workers=nw,
        python_multiprocessing=args.data_multiprocessing,
        max_rowsize=args.max_rowsize,
        debug=False,  # ms240_sept4: THIS CANNOT BE TRUE, OTHERWISE loader error
    )

    dataset_size = dataloader.get_dataset_size()

    # compute total steps and data epochs (in unit of data sink size)
    if args.train_steps == -1:
        assert args.epochs != -1
        total_train_steps = args.epochs * dataset_size
    else:
        total_train_steps = args.train_steps

    if args.dataset_sink_mode and args.sink_size != -1:
        steps_per_sink = args.sink_size
    else:
        steps_per_sink = dataset_size
    sink_epochs = math.ceil(total_train_steps / steps_per_sink)

    if args.ckpt_save_steps == -1:
        ckpt_save_interval = args.ckpt_save_interval
        step_mode = False
    else:
        step_mode = not args.dataset_sink_mode
        if not args.dataset_sink_mode:
            ckpt_save_interval = args.ckpt_save_steps
        else:
            # still need to count interval in sink epochs
            ckpt_save_interval = max(1, args.ckpt_save_steps // steps_per_sink)
            if args.ckpt_save_steps % steps_per_sink != 0:
                logger.warning(
                    f"'ckpt_save_steps' must be times of sink size or dataset_size under dataset sink mode."
                    f"Checkpoint will be saved every {ckpt_save_interval * steps_per_sink} steps."
                )
    step_mode = step_mode if args.step_mode is None else args.step_mode

    logger.info(f"train_steps: {total_train_steps}, train_epochs: {args.epochs}, sink_size: {args.sink_size}")
    logger.info(f"total train steps: {total_train_steps}, sink epochs: {sink_epochs}")
    logger.info(
        "ckpt_save_interval: {} {}".format(
            ckpt_save_interval, "steps" if (not args.dataset_sink_mode and step_mode) else "sink epochs"
        )
    )

    # 4. build training utils: lr, optim, callbacks, trainer
    # build learning rate scheduler
    if not args.decay_steps:
        args.decay_steps = total_train_steps - args.warmup_steps  # fix lr scheduling
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

    # 4.1 build optimizer
    optimizer = create_optimizer(
        lrm_model_with_loss.trainable_params(),
        name=args.optim,
        betas=args.betas,
        eps=args.optim_eps,
        group_strategy=args.group_strategy,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    if args.loss_scaler_type == "dynamic":  # for the case when there is an overflow during training
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        loss_scaler = ms.Tensor([1.0], dtype=ms.float32)

    # 4.2 weight loading: load checkpoint when resume
    lrm_model = lrm_model_with_loss.lrm_generator
    if args.resume:
        logger.info(f"Loading Fred's own ckpt {args.resume}'s 'train_resume.ckpt'")
        resume_ckpt = os.path.join(args.resume, "train_resume.ckpt")
        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            lrm_model, optimizer, resume_ckpt
        )  # refer to hpcai train script about the input usage of this func
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter
    else:
        logger.info(
            f"Resuming is turned off, with args {args.resume}.\n\t"
            "Following original itmh implementation by initializing the model using the pretrained weights from openlrm,"
            "see Sec. 3.2 of the paper for details.\n"
        )
        start_epoch = 0
        resume_param = ms.load_checkpoint(config.model.params.lrm_generator_config.openlrm_ckpt)
        ms.load_param_into_net(lrm_model, resume_param)
        # logger.info("Use random initialization for lrm, NO ckpt loading")  # NOT converge

    ema = (
        EMA(
            lrm_model_with_loss,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        lrm_model_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=args.drop_overflow_update,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    if args.global_bf16:
        model = Model(net_with_grads, amp_level="O0")
    else:
        model = Model(net_with_grads)

    # 4.3 callbacks
    callback = [
        TimeMonitor(),
        OverflowMonitor(),
    ]

    if rank_id == 0:
        save_cb = EvalSaveCallback(
            network=lrm_model_with_loss,
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="top_k",
            ckpt_max_keep=args.ckpt_max_keep,
            step_mode=step_mode,
            use_step_unit=(args.ckpt_save_steps != -1),
            ckpt_save_interval=ckpt_save_interval,
            log_interval=args.log_interval,
            start_epoch=start_epoch,
            model_name="instantmesh_stage1",
            record_lr=True,
            prefer_low_perf=True,  # prefer low loss, this for top_k recording
        )
        callback.append(save_cb)

    if args.profile:
        callback.append(ProfilerCallbackEpoch(2, 3, "./profile_data"))

    # 5. log and save config
    if rank_id == 0:
        num_params_lrm, num_params_lrm_trainable = count_params(lrm_model_with_loss)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"\tMindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"\tDistributed mode: {args.use_parallel}",
                f"\tNum params: {num_params_lrm} (lrm: {num_params_lrm})",
                f"\tNum trainable params: {num_params_lrm_trainable}",
                f"\tLearning rate: {args.start_learning_rate}",
                f"\tBatch size: {config.data.batch_size}",
                f"\tImage size: {img_size}",
                f"\tWeight decay: {args.weight_decay}",
                f"\tGrad accumulation steps: {args.gradient_accumulation_steps}",
                f"\tNum epochs: {args.epochs}",
                f"\tUse model dtype: {args.dtype}",
                f"\tMixed precision level: {args.amp_level}",
                f"\tLoss scaler: {args.loss_scaler_type}",
                f"\tInit loss scale: {args.init_loss_scale}",
                f"\tGrad clipping: {args.clip_grad}",
                f"\tMax grad norm: {args.max_grad_norm}",
                f"\tEMA: {args.use_ema}",
                f"\tUse recompute: {config.model.params.lrm_generator_config.params.use_recompute}",
                f"\tDataset sink: {args.dataset_sink_mode}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)
        logger.info("Start training...")
        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)
        OmegaConf.save(config, os.path.join(args.output_path, "cfg.yaml"))

    # 6. train
    logger.info("using the standard fitting api")
    model.fit(
        sink_epochs,
        dataloader,
        callbacks=callback,
        dataset_sink_mode=args.dataset_sink_mode,
        sink_size=args.sink_size,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
