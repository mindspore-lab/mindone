"""
Marigold model training
"""
import argparse
import importlib
import logging
import os
import shutil

import yaml
from common import init_env
from ldm.modules.logger import set_logger
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.checkpoint import resume_train_network
from ldm.modules.train.ema import EMA
from ldm.modules.train.lr_schedule import create_scheduler
from ldm.modules.train.optim import build_optimizer
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params, is_old_ms_version
from omegaconf import OmegaConf
from src.dataset import DatasetMode, get_dataset
from src.util.config_util import recursive_load_config
from src.util.depth_transform import get_depth_normalizer
from src.util.msckpt_utils import replace_unet_conv_in

import mindspore
import mindspore.dataset as ds
from mindspore import Model, Profiler, load_checkpoint, load_param_into_net, nn
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import TimeMonitor

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def build_model_from_config(config, enable_flash_attention=None, cfg=None):
    config = OmegaConf.load(config).model
    if cfg is not None:
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


def main(args):
    # load config first
    cfg = recursive_load_config(args.train_config)

    if cfg.profile:
        profiler = Profiler(output_path="./profiler_data")

    # init
    device_id, rank_id, device_num = init_env(
        args.mode,
        seed=cfg.seed,
        distributed=cfg.use_parallel,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(cfg.log_level))

    # build model
    latent_diffusion_with_loss = build_model_from_config(cfg.model_config, cfg.enable_flash_attention)
    load_pretrained_model(
        args.pretrained_model_path, latent_diffusion_with_loss, unet_initialize_random=cfg.unet_initialize_random
    )

    # change input channel of UNet from 4 to 8
    replace_unet_conv_in(latent_diffusion_with_loss, cfg.use_fp16)

    # build dataset
    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenizer
    depth_transform = get_depth_normalizer(cfg_normalizer=cfg.depth_normalization)
    train_dataset = get_dataset(
        cfg.dataset.train,
        base_data_dir="marigold-data",
        mode=DatasetMode.TRAIN,
        augmentation_args=cfg.augmentation,
        depth_transform=depth_transform,
        tokenizer=tokenizer,
        dtype=mindspore.float16 if cfg.use_fp16 else mindspore.float32,
    )
    dataset = ds.GeneratorDataset(
        source=train_dataset,
        column_names=[
            "rgb_int",
            "rgb_norm",
            "depth_raw_linear",
            "depth_filled_linear",
            "valid_mask_raw",
            "valid_mask_filled",
            "depth_raw_norm",
            "depth_filled_norm",
            "c",
        ],
        shuffle=False,
    ).batch(
        batch_size=cfg.dataloader.max_train_batch_size,
        drop_remainder=True,
        num_parallel_workers=cfg.dataloader.num_workers,
    )

    # change total_iter to epochs
    steps_per_epoch = dataset.get_dataset_size()
    epochs = (cfg.lr_scheduler.kwargs.total_iter * cfg.dataloader.effective_batch_size) // steps_per_epoch
    if cfg.profile:
        epochs = 3
    cfg.lr_scheduler.kwargs.total_iter = steps_per_epoch * epochs

    # build learning rate scheduler
    lr = create_scheduler(
        steps_per_epoch=steps_per_epoch,
        scheduler=cfg.lr_scheduler.name,
        lr=cfg.lr,
        decay_rate=cfg.lr_scheduler.kwargs.final_ratio,
        warmup_steps=cfg.lr_scheduler.kwargs.warmup_steps,
        total_iters=cfg.lr_scheduler.kwargs.total_iter,
    )

    # build optimizer
    optimizer = build_optimizer(
        model=latent_diffusion_with_loss,
        name=cfg.optim,
        lr=lr,
    )

    # build loss scaler
    if cfg.loss_scaler_type == "dynamic":
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=cfg.init_loss_scale, scale_factor=cfg.loss_scale_factor, scale_window=cfg.scale_window
        )
    elif cfg.loss_scaler_type == "static":
        loss_scaler = nn.FixedLossScaleUpdateCell(cfg.init_loss_scale)
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
        if cfg.use_ema
        else None
    )

    gradient_accumulation_steps = int(cfg.dataloader.effective_batch_size // cfg.dataloader.max_train_batch_size)

    net_with_grads = TrainOneStepWrapper(
        latent_diffusion_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=cfg.drop_overflow_update,  # TODO: allow config
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=cfg.clip_grad,
        clip_norm=cfg.max_grad_norm,
        ema=ema,
    )

    model = Model(net_with_grads)

    # callbacks
    callback = [TimeMonitor(cfg.callback_size)]
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
            step_mode=cfg.step_mode,
            ckpt_save_interval=cfg.ckpt_save_interval,
            log_interval=cfg.callback_size,
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
                f"Distributed mode: {cfg.use_parallel}",
                f"Data path: {cfg.dataset.train.dir}",
                f"Num params: {num_params:,} (unet: {num_params_unet:,}, text encoder: {num_params_text_encoder:,}, vae: {num_params_vae:,})",
                f"Num trainable params: {num_trainable_params:,}",
                f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
                f"Learning rate: {cfg.lr}",
                f"Batch size: {cfg.dataloader.max_train_batch_size}",
                f"Weight decay: {cfg.weight_decay}",
                f"Grad accumulation steps: {gradient_accumulation_steps}",
                f"Num epochs: {epochs}",
                f"Loss scaler: {cfg.loss_scaler_type}",
                f"Init loss scale: {cfg.init_loss_scale}",
                f"Grad clipping: {cfg.clip_grad}",
                f"Max grad norm: {cfg.max_grad_norm}",
                f"EMA: {cfg.use_ema}",
                f"Enable flash attention: {cfg.enable_flash_attention}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)

        logger.info("Start training...")
        # backup config files
        shutil.copyfile(cfg.model_config, os.path.join(args.output_path, "model_config.yaml"))

        with open(os.path.join(args.output_path, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), stream=f, default_flow_style=False, sort_keys=False)
    # train
    model.train(epochs, dataset, callbacks=callback, dataset_sink_mode=cfg.dataset_sink_mode, initial_epoch=start_epoch)

    if cfg.profile:
        profiler.analyse()


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    # parse arguments only take some often changed arguments in, configs about training should record in yaml file specified by `--train_config`
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="train config path to load a yaml file that override the default arguments",
    )
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--output_path", default="output/", type=str, help="output directory to save training results")
    parser.add_argument(
        "--resume",
        default=False,
        type=str,
        help="resume training, can set True or path to resume checkpoint.(default=False)",
    )
    parser.add_argument(
        "--pretrained_model_path", type=str, required=True, help="Specify the pretrained model from this checkpoint"
    )

    args = parser.parse_args()

    logger.info(args)
    main(args)
