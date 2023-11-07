"""
Train Controlnet
"""
import argparse
import importlib
import logging
import os
import shutil

from ldm.data.dataset_controlnet import load_data
from ldm.modules.logger import set_logger
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.checkpoint import resume_train_network
from ldm.modules.train.ema import EMA
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.parallel_config import ParallelConfig
from ldm.modules.train.tools import set_random_seed
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params, is_old_ms_version, str2bool
from omegaconf import OmegaConf
from utils.download import download_checkpoint

import mindspore as ms
from mindspore import Model, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor
import yaml
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


def init_env(args):
    set_random_seed(args.seed)
    ms.set_context(mode=args.mode)  # needed for MS2.0
    if args.use_parallel:
        init()
        device_id = int(os.getenv("DEVICE_ID"))
        device_num = get_group_size()
        ParallelConfig.dp = device_num
        rank_id = get_rank()
        args.rank = rank_id
        logger.debug("Device_id: {}, rank_id: {}, device_num: {}".format(device_id, rank_id, device_num))
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL,
            # parallel_mode=context.ParallelMode.AUTO_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        args.rank = rank_id

    context.set_context(
        mode=args.mode,
        device_target="Ascend",
        device_id=device_id,
        max_device_memory="30GB",  # TODO: why limit?
        pynative_synchronize=False,  # for debug in pynative mode
    )

    return rank_id, device_id, device_num

def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")

def parse_args():
    parser = argparse.ArgumentParser(description="A training script for controlnet.")
    parser.add_argument(
        "--train_config",
        default="",
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
        default="1.5",
        help="Stable diffusion version. Options: '2.1', , '2.0', '1.5', '1.5-wukong'",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--control_type", default="canny", type=str, help="Specify the control image type: canny or seg")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="Enable parallel processing")
    parser.add_argument("--custom_text_encoder", default="", type=str, help="use this to plug in custom clip model")
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="pretrained model directory")
    parser.add_argument("--pretrained_model_file", default=None, type=str, help="pretrained model file name")

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
    parser.add_argument("--scheduler", default="constant", type=str, help="Scheduler: currently support 'constant', 'cosine_decay','polynomial_decay', 'multi_step_decay'")
    parser.add_argument("--epochs", default=4, type=int, help="epochs")
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    # parser.add_argument("--cond_stage_trainable", default=False, type=str2bool, help="whether text encoder is trainable")
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
    parser.add_argument("--random_crop", default=False, type=str2bool, help="random crop")
    parser.add_argument("--filter_small_size", default=True, type=str2bool, help="filter small images")
    parser.add_argument("--image_size", default=512, type=int, help="images size")
    parser.add_argument("--image_filter_size", default=256, type=int, help="image filter size")

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
        param_dict = init_controlnet_param(param_dict, net)
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
        param_dict = init_controlnet_param(param_dict, net)
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

def init_controlnet_param(param_dict, network):
    update_param_dict = {}
    # copy sd input blocks, middle block, and time embed
    for param_name in param_dict:
        if "model.diffusion_model.input_blocks" in param_name:
            update_param_dict.update({param_name.replace("model.diffusion_model.input_blocks", "model.diffusion_model.controlnet.input_blocks"): 
                            param_dict[param_name]})
        elif "model.diffusion_model.middle_block" in param_name:
            update_param_dict.update({param_name.replace("model.diffusion_model.middle_block", "model.diffusion_model.controlnet.middle_block"): 
                            param_dict[param_name]})
        elif "model.diffusion_model.time_embed" in param_name:
            update_param_dict.update({param_name.replace("model.diffusion_model.time_embed", "model.diffusion_model.controlnet.time_embed"): 
                            param_dict[param_name]})
    # reload network's input_hint_block, middle_block_out, to avoid load warnings
    param_dict.update(update_param_dict)
    update_param_dict = {}
    for param in network.get_parameters():
        if 'model.diffusion_model.controlnet.input_hint_block' in param.name or 'model.diffusion_model.controlnet.middle_block_out' in param.name or 'model.diffusion_model.controlnet.zero_convs' in param.name:
            update_param_dict.update({param.name: param})
    param_dict.update(update_param_dict)
    return param_dict
        

def main(args):
    # init
    rank_id, device_id, device_num = init_env(args)
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
        args.data_path,
        args.train_batch_size,
        tokenizer,
        args.control_type,
        image_size=args.image_size,
        image_filter_size=args.image_filter_size,
        device_num=device_num,
        rank_id=rank_id,
        random_crop=args.random_crop,
        filter_small_size=args.filter_small_size,
    )
    
    # freeze network
    for param in latent_diffusion_with_loss.get_parameters():
        param.requires_grad = False
    SD_LOCKED = latent_diffusion_with_loss.model.diffusion_model.sd_locked
    logging.info(
        f"[Controlnet] sd_locked is {SD_LOCKED}"
    )  # set in args.model_config file
    for param in latent_diffusion_with_loss.get_parameters():
        if param.name.startswith("model.diffusion_model.controlnet."):
            param.requires_grad = True
        elif not SD_LOCKED:
            if param.name.startswith("model.diffusion_model.output_blocks") or param.name.startswith(
                "model.diffusion_model.out"
            ):
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False
    logging.info(f"[Controlnet] Num of trainable params: {len(latent_diffusion_with_loss.trainable_params())}")
    
    
    if not args.decay_steps:
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
    # decay_filter: not apply weight decay to layernorm, bias terms, and the controlnet.input_hint_block
    def decay_filter_controlnet(x):
        return "layernorm" not in x.name.lower() and "bias" not in x.name.lower() and "input_hint_block" not in x.name.lower()
    
    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=args.optim,
        betas=args.betas,
        weight_decay=args.weight_decay,
        lr=lr,
        decay_filter=decay_filter_controlnet,
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
        num_params_control, _ = count_params(latent_diffusion_with_loss.model.diffusion_model.controlnet)
        num_params_unet = num_params_unet - num_params_control
        num_params, num_trainable_params = count_params(latent_diffusion_with_loss)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {context.get_context('mode')}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Model: StableDiffusion v{args.version}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, controlnet:{num_params_control:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.controlnet.dtype}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
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