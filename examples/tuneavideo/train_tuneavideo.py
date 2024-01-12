"""
Tune A Video based on Stable Diffusion
"""
import argparse
import importlib
import logging
import os
import shutil
import sys

import yaml

sys.path.append("../stable_diffusion_v2")  # FIXME: loading modules from the stable_diffusion_v2 directory
from common import init_env
from data.dataset_tuneavideo import load_data
from ldm.modules.logger import set_logger
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.checkpoint import resume_train_network
from ldm.modules.train.ema import EMA
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params, is_old_ms_version, str2bool
from omegaconf import OmegaConf
from utils.download import download_checkpoint

from mindspore import Model, context, load_checkpoint, load_param_into_net
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)

_version_cfg = {
    "2.1": ("sd_v2-1_base-7c8d09ce.ckpt", "v2-inference.yaml", 512),
    "2.1-v": ("sd_v2-1_768_v-061732d1.ckpt", "v2-vpred-inference.yaml", 768),
    "2.0": ("sd_v2_base-57526ee4.ckpt", "v2-inference.yaml", 512),
    "2.0-v": ("sd_v2_768_v-e12e3a9b.ckpt", "v2-vpred-inference.yaml", 768),
    "1.5": ("sd_v1.5-d0ab7146.ckpt", "v1-inference.yaml", 512),
    "1.5-wukong": ("wukong-huahua-ms.ckpt", "v1-inference-chinese.yaml", 512),
}
_URL_PREFIX = "https://download.mindspore.cn/toolkits/mindone/stable_diffusion"
_MIN_CKPT_SIZE = 4.0 * 1e9


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args():
    parser = argparse.ArgumentParser(description="A training script for tune-a-video.")
    parser.add_argument(
        "--train_config",
        default="configs/train/train_config_tuneavideo_v2.yaml",
        type=str,
        help="train config path to load a yaml file that override the default arguments",
    )
    parser.add_argument("--model_config", default="", type=str, help="model config path")
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="resume training, can set True or path to resume checkpoint.(default=False)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        nargs="?",
        default="2.0",
        help="Stable diffusion version. Options: '2.1', , '2.0', '1.5', '1.5-wukong'",
    )
    parser.add_argument("--video_path", default="", type=str, help="video path")
    parser.add_argument("--prompt", default="", type=str, help="the text prompt of the input video")
    parser.add_argument("--num_frames", default=24, type=int, help="the number of sampled frames from the input video")
    parser.add_argument("--sample_start_idx", default=0, type=int, help="the sample start index of the frames")
    parser.add_argument(
        "--sample_interval",
        default=1,
        type=int,
        help="the sampling interval of frames. sample_interval=2 means to decrease the frame rate by 2.",
    )
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="Enable parallel processing")
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument(
        "--json_data_path",
        default="mindone/examples/stable_diffusion_v2/ldm/data/num_samples_64_part.json",
        type=str,
        help="the path of num_samples.json containing a dictionary with 64 parts. "
        "Each part is a large dictionary containing counts of samples of 533 tar packages.",
    )
    parser.add_argument("--custom_text_encoder", default="", type=str, help="use this to plug in custom clip model")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model directory")
    parser.add_argument("--pretrained_model_file", default=None, type=str, help="pretrained model file name")
    parser.add_argument(
        "--trainable_modules", nargs="+", default=[], help="a list of parameter names which will be trained"
    )
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay.")
    parser.add_argument(
        "--betas", type=float, default=[0.9, 0.98], help="Specify the [beta1, beta2] parameter for the Adam optimizer."
    )
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=0, type=int, help="warmup steps")
    parser.add_argument("--train_batch_size", default=4, type=int, help="batch size")
    parser.add_argument("--callback_size", default=1, type=int, help="callback size.")
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument(
        "--scheduler",
        default="constant",
        type=str,
        help="Scheduler: currently support 'constant', 'cosine_decay','polynomial_decay', 'multi_step_decay'",
    )
    parser.add_argument("--epochs", default=4, type=int, help="epochs")
    parser.add_argument(
        "--max_train_steps",
        default=None,
        type=int,
        help="the maximum training steps. If specified, it will overwrite `epochs`.",
    )
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="max gradient norm for clipping, effective when `clip_grad` enabled.",
    )

    parser.add_argument("--ckpt_save_interval", default=1, type=int, help="save checkpoint every this epochs or steps")
    parser.add_argument(
        "--step_mode",
        default=False,
        type=str2bool,
        help="whether save ckpt by steps. If False, save ckpt by epochs.",
    )
    parser.add_argument("--image_size", default=512, type=int, help="images size")

    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )

    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    default_args = parser.parse_args()
    if default_args.train_config:
        default_args.train_config = os.path.join(abs_path, default_args.train_config)
        with open(default_args.train_config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    args.model_config = os.path.join(abs_path, args.model_config)

    # check args
    if args.version:
        if args.version not in _version_cfg:
            raise ValueError(f"Unknown version: {args.version}. Supported SD versions are: {list(_version_cfg.keys())}")
    if args.pretrained_model_path is None:
        args.pretrained_model_path = "models/"
    if args.pretrained_model_file is None:
        ckpt_name = _version_cfg[args.version][0]
        args.pretrained_model_file = ckpt_name
    ckpt_path = os.path.join(args.pretrained_model_path, args.pretrained_model_file)
    desire_size = _version_cfg[args.version][2]
    if args.image_size != desire_size:
        logger.warning(
            f"The optimal H, W for SD {args.version} is ({desire_size}, {desire_size}) . But got ({args.image_size})."
        )
    # download if not exists or not complete
    if os.path.exists(ckpt_path):
        if os.path.getsize(ckpt_path) < _MIN_CKPT_SIZE:
            print(
                f"WARNING: The checkpoint size is too small {args.ckpt_path}. Please check and remove it if it is incomplete!"
            )
    if not os.path.exists(ckpt_path):
        print(f"Start downloading checkpoint {_version_cfg[args.version][0]} ...")
        ckpt_name = _version_cfg[args.version][0]
        download_checkpoint(os.path.join(_URL_PREFIX, ckpt_name), args.pretrained_model_path)
    logger.info(args)
    return args


def build_model_from_config(config):
    config = OmegaConf.load(config).model
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    return get_obj_from_str(config["target"])(**config_params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_pretrained_model(pretrained_ckpt, net):
    logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)
        param_dict = init_temporal_params(param_dict, net)
        if is_old_ms_version():
            param_not_load = load_param_into_net(net, param_dict)
        else:
            param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)

        logger.info("Params not load: {}".format(param_not_load))
    else:
        logger.warning("Checkpoint file {pretrained_ckpt} dose not exist!!!")


def load_pretrained_model_vae_unet_cnclip(pretrained_ckpt, cnclip_ckpt, net):
    new_param_dict = {}
    logger.info(f"Loading pretrained model from {pretrained_ckpt}, {cnclip_ckpt}")
    if os.path.exists(pretrained_ckpt) and os.path.exists(cnclip_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)
        param_dict = init_temporal_params(param_dict, net)
        cnclip_param_dict = load_checkpoint(pretrained_ckpt)
        for key in param_dict:
            if key.startswith("first") or key.startswith("model"):
                new_param_dict[key] = param_dict[key]
        for key in cnclip_param_dict:
            new_param_dict[key] = cnclip_param_dict[key]
        param_not_load = load_param_into_net(net, new_param_dict)

        logger.info("Params not load: {}".format(param_not_load))
    else:
        logger.warning("Checkpoint file {pretrained_ckpt}, {cnclip_ckpt} dose not exist!!!")


def init_temporal_params(param_dict, network):
    update_param_dict = {}
    # copy attn_temp from the UNetModel3D to the param_dict
    for param_name, param in network.parameters_and_names():
        if "attn_temp." in param_name or "norm_temp." in param_name:
            update_param_dict[param_name] = param
    # reload network's attn_temp to avoid warnings
    param_dict.update(update_param_dict)

    return param_dict


def main(args):
    # init
    _, rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        enable_modelarts=args.enable_modelarts,
        num_workers=args.num_workers,
        json_data_path=args.json_data_path,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # build model
    latent_diffusion_with_loss = build_model_from_config(args.model_config)

    pretrained_ckpt = os.path.join(args.pretrained_model_path, args.pretrained_model_file)

    if args.custom_text_encoder is not None and os.path.exists(args.custom_text_encoder):
        load_pretrained_model_vae_unet_cnclip(pretrained_ckpt, args.custom_text_encoder, latent_diffusion_with_loss)
    else:
        load_pretrained_model(pretrained_ckpt, latent_diffusion_with_loss)

    # build dataset
    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenizer
    dataset = load_data(
        args.video_path,
        args.prompt,
        tokenizer,
        args.train_batch_size,
        image_size=args.image_size,
        num_frames=args.num_frames,
        sample_start_idx=args.sample_start_idx,
        sample_interval=args.sample_interval,
        device_num=device_num,
        rank_id=rank_id,
    )
    if args.max_train_steps is not None:
        dataset_size = dataset.get_dataset_size()
        args.epochs = args.max_train_steps // dataset_size
        logger.info(
            f"max_train_steps is set to {args.max_train_steps}. It overwrites the number of epochs to {args.epochs}"
        )

    # freeze network
    for param_name, param in latent_diffusion_with_loss.parameters_and_names():
        if any([name in param_name for name in args.trainable_modules]):
            param.requires_grad = True
        else:
            param.requires_grad = False

    if not args.decay_steps and not args.warmup_steps:
        dataset_size = dataset.get_dataset_size()
        args.decay_steps = args.epochs * dataset_size - args.warmup_steps  # fix lr scheduling
        if args.decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.decay_steps = 1
    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        scheduler=args.scheduler,
        lr=args.start_learning_rate,
        min_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=args.optim,
        betas=args.betas,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    loss_scaler = DynamicLossScaleUpdateCell(
        loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
    )

    # resume ckpt
    if rank_id == 0:
        ckpt_dir = os.path.join(args.output_path, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    start_epoch = 0
    if args.resume:
        resume_ckpt = os.path.join(ckpt_dir, "train_resume.ckpt") if isinstance(args.resume, bool) else args.resume

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(
            latent_diffusion_with_loss, optimizer, resume_ckpt
        )
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss,  # .model, #TODO: remove .model if not only train UNet
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )
    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=True,  # TODO: allow config
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )
    model = Model(net_with_grads)

    # callbacks
    callback = [TimeMonitor(args.callback_size), LossMonitor(args.callback_size)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss,  # TODO: save unet/vae seperately
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=10,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            log_interval=args.callback_size,
            start_epoch=start_epoch,
        )

        callback.append(save_cb)

    # log
    if rank_id == 0:
        num_params_unet, _ = count_params(latent_diffusion_with_loss.model.diffusion_model)
        num_params_text_encoder, _ = count_params(latent_diffusion_with_loss.cond_stage_model)
        num_params_vae, _ = count_params(latent_diffusion_with_loss.first_stage_model)
        num_params, num_trainable_params = count_params(latent_diffusion_with_loss)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {context.get_context('mode')}",
                f"Distributed mode: {args.use_parallel}",
                f"Video path: {args.video_path}",
                f"Model: StableDiffusion v{args.version}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Number of Frames: {args.num_frames}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Init Loss Scale: {args.init_loss_scale}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(args.model_config, os.path.join(args.output_path, "model_config.yaml"))
        shutil.copyfile(args.train_config, os.path.join(args.output_path, "train_config.yaml"))

    # train
    model.train(args.epochs, dataset, callbacks=callback, dataset_sink_mode=False, initial_epoch=start_epoch)


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
