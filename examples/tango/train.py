import argparse
import importlib
import logging
import os
import shutil

from audioldm.latent_diffusion.ldm_util import count_params, str2bool
from data.dataset import build_dataset
from modules.logger import set_logger
from modules.lora import inject_trainable_lora, inject_trainable_lora_to_textencoder
from modules.train.callback import EvalSaveCallback, OverflowMonitor
from modules.train.checkpoint import resume_train_network
from modules.train.ema import EMA
from modules.train.lr_schedule import create_scheduler
from modules.train.optim import build_optimizer
from modules.train.parallel_config import ParallelConfig
from modules.train.tools import parse_with_config, set_random_seed
from modules.train.trainer import TrainOneStepWrapper
from omegaconf import OmegaConf
from tango import Tango

import mindspore as ms
from mindspore import Model, context
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"
SD_VERSION = os.getenv("SD_VERSION", default="2.1")

logger = logging.getLogger(__name__)


def init_env(args):
    set_random_seed(args.seed)

    ms.set_context(mode=context.GRAPH_MODE)  # needed for MS2.0
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
        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        print(dict(zip(var_info, var_value)), flush=True)

        # if args.enable_modelarts:
        #     split_and_sync_data(args, device_num, rank_id)
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        args.rank = rank_id

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=device_id,
        max_device_memory="30GB",
    )
    ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})  # Only effective on Ascend 901B

    return rank_id, device_id, device_num


def build_model_from_config(config):
    config = OmegaConf.load(config).model
    config_params = config.get("params", dict())
    # config_params['cond_stage_trainable'] = cond_stage_trainable # TODO: easy config
    return get_obj_from_str(config["target"])(**config_params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def main(args):
    # init
    rank_id, device_id, device_num = init_env(args)
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # build model
    tango = Tango(args.model_config)
    latent_diffusion_with_loss = tango.model

    # build dataset
    tokenizer = latent_diffusion_with_loss.tokenizer
    dataset = build_dataset(args, device_num, rank_id, tokenizer)

    # lora injection
    if args.use_lora:
        # freeze network
        for param in tango.get_parameters():
            param.requires_grad = False

        # inject lora params
        num_injected_params = 0
        if args.lora_ft_unet:
            unet_lora_layers, unet_lora_params = inject_trainable_lora(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
            )
            num_injected_params += len(unet_lora_params)
        if args.lora_ft_text_encoder:
            text_encoder_lora_layers, text_encoder_lora_params = inject_trainable_lora_to_textencoder(
                latent_diffusion_with_loss,
                rank=args.lora_rank,
                use_fp16=args.lora_fp16,
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
    lr = create_scheduler(
        steps_per_epoch=dataset_size,
        scheduler=args.scheduler,
        lr=args.start_learning_rate,
        min_lr=args.end_learning_rate,
        warmup_steps=args.warmup_steps,
        decay_steps=args.decay_steps,
        num_epochs=args.epochs,
    )
    optimizer = build_optimizer(latent_diffusion_with_loss, args, lr)

    loss_scaler = DynamicLossScaleUpdateCell(
        loss_scale_value=args.init_loss_scale, scale_factor=args.loss_scale_factor, scale_window=args.scale_window
    )

    if rank_id == 0:
        os.makedirs(os.path.join(args.output_path, "model_config"), exist_ok=True)
        ckpt_dir = os.path.join(args.output_path, "ckpt")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)
    start_epoch = 0

    pretrained_ckpt = os.path.join(args.pretrained_model_path, args.pretrained_model_file)
    if os.path.isfile(pretrained_ckpt):
        resume_ckpt = pretrained_ckpt

        start_epoch, loss_scale, cur_iter, last_overflow_iter = resume_train_network(tango, optimizer, resume_ckpt)
        loss_scaler.loss_scale_value = loss_scale
        loss_scaler.cur_iter = cur_iter
        loss_scaler.last_overflow_iter = last_overflow_iter

    # trainer (standalone and distributed)
    ema = (
        EMA(
            latent_diffusion_with_loss,
            ema_decay=0.9999,
        )
        if args.use_ema
        else None
    )

    net_with_grads = TrainOneStepWrapper(
        tango,
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
            network=tango,  # TODO: save unet/vae seperately
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
        )

        callback.append(save_cb)

    # log
    if rank_id == 0:
        num_params_unet, _ = count_params(tango.model.unet)
        num_params_text_encoder, _ = count_params(tango.model.text_encoder)
        num_params_vae, _ = count_params(tango.vae)
        num_params, num_trainable_params = count_params(latent_diffusion_with_loss)

        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                "MindSpore mode[GRAPH(0)/PYNATIVE(1)]: 0",
                f"Distributed mode: {args.use_parallel}",
                f"Data path: {args.data_path}",
                f"Model: StableDiffusion v{SD_VERSION}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.unet.diffusion_model.dtype}",
                f"Use LoRA: {args.use_lora}",
                f"LoRA rank: {args.lora_rank}",
                f"Learning rate: {args.start_learning_rate}",
                f"Batch size: {args.train_batch_size}",
                f"Weight decay: {args.weight_decay}",
                f"Grad accumulation steps: {args.gradient_accumulation_steps}",
                f"Num epochs: {args.epochs}",
                f"Grad clipping: {args.clip_grad}",
                f"Max grad norm: {args.max_grad_norm}",
                f"EMA: {args.use_ema}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(os.path.join(args.model_config, "config.json"), os.path.join(args.output_path, "config.json"))
        shutil.copyfile(
            os.path.join(args.model_config, "diffusion_model_config.json"),
            os.path.join(args.output_path, "diffusion_model_config.json"),
        )
        shutil.copyfile(
            os.path.join(args.model_config, "stft_config.json"), os.path.join(args.output_path, "stft_config.json")
        )
        shutil.copyfile(
            os.path.join(args.model_config, "train_config_v.json"),
            os.path.join(args.output_path, "train_config_v.json"),
        )
        shutil.copyfile(
            os.path.join(args.model_config, "vae_config.json"), os.path.join(args.output_path, "vae_config.json")
        )

    # train
    model.train(args.epochs, dataset, callbacks=callback, dataset_sink_mode=False, initial_epoch=start_epoch)


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--enable_modelarts", default=False, type=str2bool, help="run codes in ModelArts platform")
    parser.add_argument("--num_workers", default=1, type=int, help="the number of modelarts workers")
    parser.add_argument("--data_path", default="dataset", type=str, help="data path")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument("--train_config", default="configs/train_config.json", type=str, help="train config path")
    parser.add_argument("--model_config", default="configs", type=str, help="model config folder")
    parser.add_argument("--custom_text_encoder", default="", type=str, help="use this to plug in custom clip model")
    parser.add_argument("--pretrained_model_path", default="", type=str, help="pretrained model directory")
    parser.add_argument("--pretrained_model_file", default="", type=str, help="pretrained model file name")
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

    parser.add_argument("--optim", default="adamw", type=str, help="optimizer")
    parser.add_argument("--weight_decay", default=1e-6, type=float, help="Weight decay.")
    parser.add_argument("--seed", default=3407, type=int, help="data path")
    parser.add_argument("--warmup_steps", default=1000, type=int, help="warmup steps")
    parser.add_argument("--train_batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--callback_size", default=1, type=int, help="callback size.")
    parser.add_argument("--start_learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--end_learning_rate", default=1e-7, type=float, help="The end learning rate for Adam.")
    parser.add_argument("--decay_steps", default=0, type=int, help="lr decay steps.")
    parser.add_argument("--scheduler", default="cosine_decay", type=str, help="scheduler.")
    parser.add_argument("--epochs", default=40, type=int, help="epochs")
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

    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )

    args = parser.parse_args()
    args = parse_with_config(args)
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    args.model_config = os.path.join(abs_path, args.model_config)

    logger.info(args)
    main(args)
