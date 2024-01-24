"""
Stable diffusion model training/finetuning
"""
import argparse
import importlib
import logging
import os
import shutil

import yaml
from common import init_env
from ldm.data.dataset import build_dataset
from ldm.modules.logger import set_logger
from ldm.modules.lora import inject_trainable_lora, inject_trainable_lora_to_textencoder
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.checkpoint import resume_train_network
from ldm.modules.train.ema import EMA
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params, is_old_ms_version, str2bool
from omegaconf import OmegaConf

from mindspore import Model, Profiler, load_checkpoint, load_param_into_net, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def build_model_from_config(config, enable_flash_attention=None):
    config = OmegaConf.load(config).model
    if args is not None:
        if enable_flash_attention is not None:
            config["params"]["unet_config"]["params"]["enable_flash_attention"] = enable_flash_attention
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    # config_params['cond_stage_trainable'] = cond_stage_trainable # TODO: easy config
    return get_obj_from_str(config["target"])(**config_params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_pretrained_model(pretrained_ckpt, net, unet_initialize_random=False):
    logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)

        if unet_initialize_random:
            pnames = list(param_dict.keys())
            # pop unet params from pretrained weight
            for pname in pnames:
                if pname.startswith("model.diffusion_model"):
                    param_dict.pop(pname)
            logger.warning("UNet will be initialized randomly")

        if is_old_ms_version():
            param_not_load = load_param_into_net(net, param_dict)
        else:
            param_not_load, ckpt_not_load = load_param_into_net(net, param_dict)
        logger.info("Params not load: {}".format(param_not_load))
    else:
        logger.warning(f"Checkpoint file {pretrained_ckpt} dose not exist!!!")


def load_pretrained_model_vae_unet_cnclip(pretrained_ckpt, cnclip_ckpt, net):
    new_param_dict = {}
    logger.info(f"Loading pretrained model from {pretrained_ckpt}, {cnclip_ckpt}")
    if os.path.exists(pretrained_ckpt) and os.path.exists(cnclip_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)
        cnclip_param_dict = load_checkpoint(pretrained_ckpt)
        for key in param_dict:
            if key.startswith("first") or key.startswith("model"):
                new_param_dict[key] = param_dict[key]
        for key in cnclip_param_dict:
            new_param_dict[key] = cnclip_param_dict[key]
        param_not_load = load_param_into_net(net, new_param_dict)
        logger.info("Params not load: {}".format(param_not_load))
    else:
        logger.warning(f"Checkpoint file {pretrained_ckpt}, {cnclip_ckpt} dose not exist!!!")


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        default="",
        type=str,
        help="train config path to load a yaml file that override the default arguments",
    )
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--replace_small_images",
        default=True,
        type=str2bool,
        help="replace the small-size images with other training samples",
    )
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument(
        "--json_data_path",
        default="mindone/examples/stable_diffusion_v2/ldm/data/num_samples_64_part.json",
        type=str,
        help="the path of num_samples.json containing a dictionary with 64 parts. "
        "Each part is a large dictionary containing counts of samples of 533 tar packages.",
    )
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="resume training, can set True or path to resume checkpoint.(default=False)",
    )
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
    parser.add_argument("--model_config", default="configs/v1-train-chinese.yaml", type=str, help="model config path")
    parser.add_argument("--custom_text_encoder", default="", type=str, help="use this to plug in custom clip model")
    parser.add_argument(
        "--pretrained_model_path", default="", type=str, help="Specify the pretrained model from this checkpoint"
    )
    parser.add_argument("--use_lora", default=False, type=str2bool, help="use lora finetuning")
    parser.add_argument("--lora_ft_unet", default=True, type=str2bool, help="whether to apply lora finetune to unet")
    parser.add_argument(
        "--lora_ft_text_encoder", default=False, type=str2bool, help="whether to apply lora finetune to text encoder"
    )
    parser.add_argument(
        "--lora_rank",
        default=4,
        type=int,
        help="lora rank. The bigger, the larger the LoRA model will be, but usually gives better generation quality.",
    )
    parser.add_argument("--lora_fp16", default=True, type=str2bool, help="Whether use fp16 for LoRA params.")
    parser.add_argument(
        "--lora_scale",
        default=1.0,
        type=float,
        help="scale, the higher, the more LoRA weights will affect orignal SD. If 0, LoRA has no effect.",
    )

    parser.add_argument("--unet_initialize_random", default=False, type=str2bool, help="initialize unet randomly")
    parser.add_argument("--dataset_sink_mode", default=False, type=str2bool, help="sink mode")
    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument(
        "--betas", type=float, default=[0.9, 0.999], help="Specify the [beta1, beta2] parameter for the Adam optimizer."
    )
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--train_batch_size", default=10, type=int, help="batch size")
    parser.add_argument("--callback_size", default=1, type=int, help="callback size.")
    parser.add_argument("--start_learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler.")
    parser.add_argument("--epochs", default=10, type=int, help="epochs")
    parser.add_argument("--init_loss_scale", default=65536, type=float, help="loss scale")
    parser.add_argument("--loss_scale_factor", default=2, type=float, help="loss scale factor")
    parser.add_argument("--scale_window", default=1000, type=float, help="scale window")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="gradient accumulation steps")
    # parser.add_argument("--cond_stage_trainable", default=False, type=str2bool, help="whether text encoder is trainable")
    parser.add_argument("--use_ema", default=False, type=str2bool, help="whether use EMA")
    parser.add_argument("--clip_grad", default=False, type=str2bool, help="whether apply gradient clipping")
    # parser.add_argument("--use_recompute", default=None, type=str2bool, help="whether use recompute")
    parser.add_argument(
        "--enable_flash_attention",
        default=None,
        type=str2bool,
        help="whether enable flash attention. If not None, it will overwrite the value in model config yaml.",
    )
    parser.add_argument("--drop_overflow_update", default=True, type=str2bool, help="drop overflow update")
    parser.add_argument("--loss_scaler_type", default="dynamic", type=str, help="dynamic or static")
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

    logger.info(args)
    return args


def main(args):
    if args.profile:
        profiler = Profiler(output_path="./profiler_data")
        args.epochs = 3

    # init
    device_id, rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        enable_modelarts=args.enable_modelarts,
        num_workers=args.num_workers,
        json_data_path=args.json_data_path,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # build model
    latent_diffusion_with_loss = build_model_from_config(args.model_config, args.enable_flash_attention)
    if args.custom_text_encoder is not None and os.path.exists(args.custom_text_encoder):
        load_pretrained_model_vae_unet_cnclip(
            args.pretrained_model_path, args.custom_text_encoder, latent_diffusion_with_loss
        )
    else:
        load_pretrained_model(
            args.pretrained_model_path, latent_diffusion_with_loss, unet_initialize_random=args.unet_initialize_random
        )

    # build dataset
    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenizer
    dataset = build_dataset(
        data_path=args.data_path,
        train_batch_size=args.train_batch_size,
        tokenizer=tokenizer,
        image_size=args.image_size,
        image_filter_size=args.image_filter_size,
        device_num=device_num,
        rank_id=rank_id,
        random_crop=args.random_crop,
        filter_small_size=args.filter_small_size,
        replace=args.replace_small_images,
        enable_modelarts=args.enable_modelarts,
    )

    # lora injection
    if args.use_lora:
        # freeze network
        for param in latent_diffusion_with_loss.get_parameters():
            param.requires_grad = False

        # inject lora params
        num_injected_params = 0
        if args.lora_ft_unet:
            unet_lora_layers, unet_lora_params = inject_trainable_lora(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
                scale=args.lora_scale,
            )
            num_injected_params += len(unet_lora_params)
        if args.lora_ft_text_encoder:
            text_encoder_lora_layers, text_encoder_lora_params = inject_trainable_lora_to_textencoder(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
                scale=args.lora_scale,
            )
            num_injected_params += len(text_encoder_lora_params)

        assert (
            len(latent_diffusion_with_loss.trainable_params()) == num_injected_params
        ), "Only lora params {} should be trainable. but got {} trainable params".format(
            num_injected_params, len(latent_diffusion_with_loss.trainable_params())
        )
        # print('Trainable params: ', latent_diffusion_with_loss.model.trainable_params())
    dataset_size = dataset.get_dataset_size()
    if not args.decay_steps:
        args.decay_steps = args.epochs * dataset_size - args.warmup_steps  # fix lr scheduling
        if args.decay_steps <= 0:
            logger.warning(
                f"decay_steps is {args.decay_steps}, please check epochs, dataset_size and warmup_steps. "
                f"Will force decay_steps to be set to 1."
            )
            args.decay_steps = 1

    # build learning rate scheduler
    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        scheduler=args.scheduler,
        lr=args.start_learning_rate,
        min_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )

    # build optimizer
    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=args.optim,
        betas=args.betas,
        weight_decay=args.weight_decay,
        lr=lr,
    )

    if args.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
        )
    elif args.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(args.init_loss_scale)
    else:
        raise ValueError

    # resume ckpt
    if rank_id == 0:
        ckpt_dir = os.path.join(args.output_path, "ckpt")
        os.makedirs(ckpt_dir, exist_ok=True)

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
        drop_overflow_update=args.drop_overflow_update,  # TODO: allow config
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        clip_grad=args.clip_grad,
        clip_norm=args.max_grad_norm,
        ema=ema,
    )

    model = Model(net_with_grads)

    # callbacks
    callback = [TimeMonitor(args.callback_size)]
    ofm_cb = OverflowMonitor()
    callback.append(ofm_cb)

    if rank_id == 0:
        save_cb = EvalSaveCallback(
            network=latent_diffusion_with_loss,  # TODO: save unet/vae seperately
            use_lora=args.use_lora,
            rank_id=rank_id,
            ckpt_save_dir=ckpt_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=10,
            step_mode=args.step_mode,
            ckpt_save_interval=args.ckpt_save_interval,
            lora_rank=args.lora_rank,
            log_interval=args.callback_size,
            start_epoch=start_epoch,
            record_lr=False,  # LR retrival is not supportted on 910b currently
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
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
                f"Use LoRA: {args.use_lora}",
                f"LoRA rank: {args.lora_rank}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Loss scaler: {args.loss_scaler_type}",
                f"Init loss scale: {args.init_loss_scale}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
                f"Enable flash attention: {args.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(args.model_config, os.path.join(args.output_path, "model_config.yaml"))

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)
    # train
    model.train(
        args.epochs, dataset, callbacks=callback, dataset_sink_mode=args.dataset_sink_mode, initial_epoch=start_epoch
    )

    if args.profile:
        profiler.analyse()


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
