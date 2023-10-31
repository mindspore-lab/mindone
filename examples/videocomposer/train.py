"""
VC training/finetuning
"""
import logging
import os

# import datetime
import shutil
import sys

import numpy as np

# from omegaconf import OmegaConf
from vc.annotator.depth import midas_v3_dpt_large
from vc.annotator.sketch import pidinet_bsd, sketch_simplification_gan
from vc.config import Config
from vc.data.dataset_train import build_dataset
from vc.diffusion.latent_diffusion import LatentDiffusion
from vc.models import AutoencoderKL, FrozenOpenCLIPEmbedder, FrozenOpenCLIPVisualEmbedder, UNetSD_temporal
from vc.trainer.lr_scheduler import build_lr_scheduler
from vc.trainer.optim import build_optimizer
from vc.utils import CUSTOM_BLACK_LIST, convert_to_abspath, get_abspath_of_weights, setup_logger

import mindspore as ms
from mindspore import Model, context
from mindspore.amp import custom_mixed_precision
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../stable_diffusion_v2/")))

from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor, ProfilerCallback
from ldm.modules.train.ema import EMA
from ldm.modules.train.parallel_config import ParallelConfig
from ldm.modules.train.tools import set_random_seed
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.util import count_params

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"  # large value may block the log of error info on modelarts

logger = logging.getLogger(__name__)


def init_env(args):
    # rank_id - global card id, device_num - num of cards
    set_random_seed(args.seed)

    ms.set_context(mode=args.ms_mode)  # needed for MS2.0
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
            gradients_mean=True,
            device_num=device_num,
        )
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        args.rank = rank_id

    context.set_context(
        mode=args.ms_mode,
        device_target="Ascend",
        device_id=device_id,
        # max_device_memory="30GB", # adapt for 910b
    )
    ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})  # Only effective on Ascend 901B
    # ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_bf16"})  # TODO: testing bf16

    # logger
    # ct = datetime.datetime.now().strftime("_%y%m%d_%H_%M")
    # args.output_dir += ct
    setup_logger(output_dir=args.output_dir, rank=args.rank)

    return rank_id, device_id, device_num


def check_config(cfg):
    # prev_cond_idx = -1
    for cond in cfg.conditions_for_train:
        if cond not in cfg.video_compositions:
            raise ValueError(f"Unknown condition: {cond}. Available conditions are: {cfg.video_compositions}")
            # idx = cfg.video_compositions.index(cond)
    print("===> Conditions used for training: ", cfg.conditions_for_train)

    # turn to abs path if it's relative path, for modelarts running
    cfg.root_dir = convert_to_abspath(cfg.root_dir, __dir__)
    cfg.cfg_file = convert_to_abspath(cfg.cfg_file, __dir__)
    cfg.resume_checkpoint = convert_to_abspath(cfg.resume_checkpoint, __dir__)

    # TODO: set sink_size and epochs to solve it
    assert not (
        cfg.step_mode and cfg.dataset_sink_mode
    ), f"step_mode is enabled, dataset_sink_mode should be set to False, but got {cfg.dataset_sink_mode})"


def main(cfg):
    check_config(cfg)

    # 1. init
    rank_id, device_id, device_num = init_env(cfg)

    # 2. build model components for ldm
    # 2.1 clip - text encoder, and image encoder (optional)
    clip_text_encoder = FrozenOpenCLIPEmbedder(
        layer="penultimate",
        pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint),
        tokenizer_path=get_abspath_of_weights(cfg.clip_tokenizer),
        use_fp16=cfg.use_fp16,
    )
    logger.info("clip text encoder init.")
    tokenizer = clip_text_encoder.tokenizer
    clip_text_encoder.set_train(False)

    if "image" in cfg.conditions_for_train:
        clip_image_encoder = FrozenOpenCLIPVisualEmbedder(
            layer="penultimate", pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint), use_fp16=cfg.use_fp16
        )
        clip_image_encoder.set_train(False)
        logger.info("clip image encoder init.")
    else:
        clip_image_encoder = None

    # 2.2 vae
    vae = AutoencoderKL(
        cfg.sd_config,
        4,
        ckpt_path=get_abspath_of_weights(cfg.sd_checkpoint),
        use_fp16=cfg.use_fp16,
        version="2.1",
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False
    logger.info("vae init")

    # 2.3 unet3d with STC encoders
    unet = UNetSD_temporal(
        cfg=cfg,
        in_dim=cfg.unet_in_dim,
        concat_dim=cfg.unet_concat_dim,
        dim=cfg.unet_dim,
        context_dim=cfg.unet_context_dim,
        out_dim=cfg.unet_out_dim,
        dim_mult=cfg.unet_dim_mult,
        num_heads=cfg.unet_num_heads,
        head_dim=cfg.unet_head_dim,
        num_res_blocks=cfg.unet_res_blocks,
        attn_scales=cfg.unet_attn_scales,
        dropout=cfg.unet_dropout,
        temporal_attention=cfg.temporal_attention,
        temporal_attn_times=cfg.temporal_attn_times,
        use_checkpoint=cfg.use_checkpoint,
        use_fps_condition=cfg.use_fps_condition,
        use_sim_mask=cfg.use_sim_mask,
        video_compositions=cfg.video_compositions,
        misc_dropout=cfg.misc_dropout,
        p_all_zero=cfg.p_all_zero,
        p_all_keep=cfg.p_all_zero,
        use_fp16=cfg.use_fp16,
        use_adaptive_pool=cfg.use_adaptive_pool,
        use_recompute=cfg.use_recompute,
    )
    # TODO: use common checkpoiont download, mapping, and loading
    if cfg.resume_checkpoint.endswith(".ckpt") and os.path.exists(cfg.resume_checkpoint):
        unet.load_state_dict(cfg.resume_checkpoint)
        logger.warning(f"UNet loaded from {cfg.resume_checkpoint}")
    else:
        logger.warning("UNet checkpoint is not given or not exists. UNet will be trained from scratch!!!")
    unet = unet.set_train(True)

    # 2.4 other NN-based condition extractors
    cfg.dtype = ms.float16 if cfg.use_fp16 else ms.float32
    extra_conds = {}
    # 2.4 1) sketch - pidinet and cleaner
    if ("single_sketch" in cfg.conditions_for_train) or ("sketch" in cfg.conditions_for_train):
        # sketch extractor
        pidinet = pidinet_bsd(
            pretrained=True, vanilla_cnn=True, ckpt_path=get_abspath_of_weights(cfg.pidinet_checkpoint)
        )
        custom_mixed_precision(
            pidinet, black_list=CUSTOM_BLACK_LIST
        )  # similar to O2, cast to fp16 except for those in black_list
        pidinet = pidinet.set_train(False)
        for _, param in pidinet.parameters_and_names():
            param.requires_grad = False
        # cleaner
        cleaner = sketch_simplification_gan(
            pretrained=True, ckpt_path=get_abspath_of_weights(cfg.sketch_simplification_checkpoint)
        )
        custom_mixed_precision(
            cleaner, black_list=CUSTOM_BLACK_LIST
        )  # similar to O2, cast to fp16 except for those in black_list
        cleaner = cleaner.set_train(False)
        for _, param in cleaner.parameters_and_names():
            param.requires_grad = False
        extra_conds["sketch"] = {
            "pidinet": pidinet,
            "sketch_mean": np.array(cfg.sketch_mean).reshape((1, -1, 1, 1)),
            "sketch_std": np.array(cfg.sketch_std).reshape((1, -1, 1, 1)),
            "cleaner": cleaner,
        }

    # 2.4 2) depth - midas
    if "depthmap" in cfg.conditions_for_train:
        midas = midas_v3_dpt_large(pretrained=True, ckpt_path=get_abspath_of_weights(cfg.midas_checkpoint))
        custom_mixed_precision(
            midas, black_list=CUSTOM_BLACK_LIST
        )  # similar to O2, cast to fp16 except for those in black_list
        midas = midas.set_train(False)
        for _, param in midas.parameters_and_names():
            param.requires_grad = False
        extra_conds["depthmap"] = {"midas": midas, "depth_clamp": cfg.depth_clamp, "depth_std": cfg.depth_std}

    # count num params for each network
    param_nums = {
        "vae": count_params(vae)[0],
        "clip_text_encoder": count_params(clip_text_encoder)[0],
        "unet with stc encoders": count_params(unet)[0],
    }
    if clip_image_encoder is not None:
        param_nums["clip_image_encoder"] = count_params(clip_image_encoder)[0]
    for cond in extra_conds:
        for _net in extra_conds[cond]:
            print()
            if isinstance(extra_conds[cond][_net], ms.nn.Cell):
                param_nums[cond + "-" + _net] = count_params(extra_conds[cond][_net])[0]
    logger.info("Param numbers: {}".format(param_nums))
    tot_params = sum([param_nums[module] for module in param_nums])
    logger.info("Total parameters: {:,}".format(tot_params))

    # 3. build latent diffusion with loss cell (core)
    ldm_with_loss = LatentDiffusion(
        unet,
        vae,
        clip_text_encoder,
        clip_image_encoder=clip_image_encoder,
        conditions=cfg.conditions_for_train,
        extra_conds=extra_conds,
        use_fp16=cfg.use_fp16,
        timesteps=cfg.num_timesteps,
        parameterization=cfg.mean_type,
        scale_factor=cfg.scale_factor,
        cond_stage_trainable=cfg.cond_stage_trainable,
        linear_start=cfg.linear_start,
        linear_end=cfg.linear_end,
    )

    # auto_mixed_precision(ldm_with_loss, amp_level="O3") # Note: O3 will lead to gradient overflow

    # 4. build training dataset
    dataloader = build_dataset(cfg, device_num, rank_id, tokenizer)
    num_batches = dataloader.get_dataset_size()

    # 5. build training utils
    learning_rate = build_lr_scheduler(
        steps_per_epoch=num_batches,
        scheduler=cfg.scheduler,
        lr=cfg.learning_rate,
        min_lr=cfg.end_learning_rate,
        warmup_steps=cfg.warmup_steps,
        decay_steps=cfg.decay_steps,
        num_epochs=cfg.epochs,
    )
    optimizer = build_optimizer(ldm_with_loss, cfg, learning_rate, eps=cfg.optim_eps)
    loss_scaler = DynamicLossScaleUpdateCell(loss_scale_value=cfg.loss_scale, scale_factor=2, scale_window=1000)
    ema = (
        EMA(
            ldm_with_loss.unet,
            ema_decay=0.9999,
        )
        if cfg.use_ema
        else None
    )
    net_with_grads = TrainOneStepWrapper(
        ldm_with_loss,
        optimizer=optimizer,
        scale_sense=loss_scaler,
        drop_overflow_update=True,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        clip_grad=False,  # args.clip_grad,
        clip_norm=1.0,  # args.max_grad_norm,
        ema=ema,
    )

    model = Model(net_with_grads)

    callbacks = [TimeMonitor(cfg.log_interval), LossMonitor(cfg.log_interval), OverflowMonitor()]
    if cfg.profile:
        callbacks.append(ProfilerCallback())

    start_epoch = 0
    if rank_id == 0:
        net_to_save = ldm_with_loss.unet if cfg.save_unet_only else ldm_with_loss
        model_name = "vc_unet" if cfg.save_unet_only else "vc"
        save_cb = EvalSaveCallback(
            network=net_to_save,  # TODO: save unet seperately
            use_lora=False,
            rank_id=rank_id,
            ckpt_save_dir=cfg.output_dir,
            ema=ema,
            ckpt_save_policy="latest_k",
            ckpt_max_keep=cfg.ckpt_max_keep,
            step_mode=cfg.step_mode,
            ckpt_save_interval=cfg.ckpt_save_interval,
            log_interval=cfg.log_interval,
            start_epoch=start_epoch,
            record_lr=False,  # LR retrival is not supportted on 910b currently
            model_name=model_name,
        )
        callbacks.append(save_cb)

    # - log and save training configs
    if rank_id == 0:
        _, num_trainable_params = count_params(ldm_with_loss)
        key_info = "Key Settings:\n" + "=" * 50 + "\n"
        key_info += "\n".join(
            [
                f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {cfg.ms_mode}",
                f"Distributed mode: {cfg.use_parallel}",
                f"Dataset sink mode: {cfg.dataset_sink_mode}",
                f"Data path: {cfg.root_dir}",
                f"Num batches per card: {num_batches}",
                "Model: VideoComposer",
                f"Conditions for training: {cfg.conditions_for_train}",
                f"Num params: {param_nums}",
                f"Num trainable params: {num_trainable_params:,}",
                f"Optimizer: {cfg.optim}",
                f"Learning rate: {cfg.learning_rate}",
                f"Batch size: {cfg.batch_size}",
                f"Max frames: {cfg.max_frames}",
                f"Weight decay: {cfg.weight_decay}",
                f"Num epochs: {cfg.epochs}",
                f"Use fp16: {cfg.use_fp16}",
                f"Use recompute: {cfg.use_recompute}",
                f"Use EMA: {cfg.use_ema}",
            ]
        )
        key_info += "\n" + "=" * 50
        logger.info(key_info)
        shutil.copyfile(os.path.join(__dir__, "configs/train_base.py"), os.path.join(cfg.output_dir, "train_base.py"))
        shutil.copyfile(cfg.cfg_file, os.path.join(cfg.output_dir, "train.yaml"))

    # 6. train
    logger.info("Start training. Please wait for graph compilation (~15 mins depending on processor)")
    model.train(
        cfg.epochs, dataloader, callbacks=callbacks, dataset_sink_mode=cfg.dataset_sink_mode, initial_epoch=start_epoch
    )


if __name__ == "__main__":
    # 0. parse config
    from configs.train_base import cfg  # base config from train_base.py

    args_for_update = Config(load=True).cfg_dict  # config args from CLI (arg parser) and yaml files

    # update base config
    for k, v in args_for_update.items():
        cfg[k] = v

    print(cfg)
    main(cfg)
