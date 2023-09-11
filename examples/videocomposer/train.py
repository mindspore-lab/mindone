"""
VC training/finetuning
"""
from vc.config import Config
import importlib
import datetime
import logging
import os
import shutil

from omegaconf import OmegaConf

import mindspore as ms
from mindspore import Model, context, load_checkpoint, load_param_into_net
from mindspore.communication.management import get_group_size, get_rank, init
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.train.callback import LossMonitor, TimeMonitor
from mindspore.amp import auto_mixed_precision

from vc.utils import get_abspath_of_weights, save_video_multiple_conditions, setup_logger, setup_seed
from vc.models import (
    AutoencoderKL,
    FrozenOpenCLIPEmbedder,
    FrozenOpenCLIPVisualEmbedder,
    UNetSD_temporal,
    get_first_stage_encoding,
)
from vc.annotator.depth import midas_v3_dpt_large
from vc.annotator.sketch import pidinet_bsd, sketch_simplification_gan

from vc.diffusion.latent_diffusion import LatentDiffusion
from vc.data.dataset_train import build_dataset
from vc.trainer.optim import build_optimizer
from vc.trainer.lr_scheduler import build_lr_scheduler

import sys
sys.path.append("../stable_diffusion_v2/")
from ldm.modules.train.tools import set_random_seed
from ldm.util import count_params
from ldm.modules.train.trainer import TrainOneStepWrapper
from ldm.modules.train.callback import EvalSaveCallback, OverflowMonitor
from ldm.modules.train.parallel_config import ParallelConfig
from mindspore.train.callback import LossMonitor, TimeMonitor


os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

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
        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        print(dict(zip(var_info, var_value)), flush=True)

        #if args.enable_modelarts:
        #    split_and_sync_data(args, device_num, rank_id)
    else:
        device_num = 1
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        args.rank = rank_id

    context.set_context(
        mode=args.ms_mode,
        device_target="Ascend",
        device_id=device_id,
        max_device_memory="30GB",
    )
    ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})  # Only effective on Ascend 901B
    
    # logger
    ct = datetime.datetime.now().strftime("_%y%m%d_%H_%M") # TODO: 
    #args.output_dir += ct
    setup_logger(output_dir=args.output_dir, rank=args.rank)

    return rank_id, device_id, device_num

def check_config(cfg):
    prev_cond_idx = -1
    for cond in cfg.conditions_for_train:
        if cond not in cfg.video_compositions:
            raise ValueError(f"Unknown condition: {cond}. Available conditions are: {cfg.video_compositions}")
            idx = cfg.video_compositions.index(cond) 


def main(cfg):
    check_config(cfg)

    # 1. init
    rank_id, device_id, device_num = init_env(cfg)

    # 2. build model components for ldm
    # create vae, clip, unet, and load pretrained weights  

    # 2.1 clip - text encoder and image encoder
    clip_text_encoder = FrozenOpenCLIPEmbedder(
        layer="penultimate",
        pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint),
        tokenizer_path=get_abspath_of_weights(cfg.clip_tokenizer),
        use_fp16=cfg.use_fp16,
    )
    logger.info("clip text encoder init.")
    tokenizer = clip_text_encoder.tokenizer
    clip_text_encoder.set_train(False)

    clip_image_encoder= FrozenOpenCLIPVisualEmbedder(
        layer="penultimate", 
        pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint), 
        use_fp16=cfg.use_fp16
    )
    clip_image_encoder.set_train(False)
    logger.info("clip image encoder init.")

    #print("D--: ", tokenizer("a dog", padding="max_length", max_length=77)["input_ids"])

    # 2.2 vae
    vae = AutoencoderKL(
        cfg.sd_config,
        4,
        ckpt_path=get_abspath_of_weights(cfg.sd_checkpoint),
        use_fp16=cfg.use_fp16,
        version="2.1",
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters(): # freeze vae
        param.requires_grad = False
    logger.info("vae init")

    # 2.3 unet3d with STC encoders
    # TODO: optimize the args, consider isoldate stc encoder from unet
    black_image_feature = ms.ops.zeros([1, 1, cfg.vit_dim]) # img feature vector of vit-h is fxed to len of 1024 
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
        zero_y=None, # assume we always use text prompts  (y, even y="")
        black_image_feature=black_image_feature,
        use_fp16=cfg.use_fp16,
        use_adaptive_pool=cfg.use_adaptive_pool, # TODO: AdaptivePool2DGrad is not stable on ms2.0 
    )
    # TODO: use common checkpoiont download, mapping, and loading 
    unet.load_state_dict(get_abspath_of_weights(cfg.resume_checkpoint))
    unet = unet.set_train(True)

    # 2.4 other NN-based condition extractors
    # TODO: add for depth/sketch to video training
    cfg.dtype = ms.float16 if cfg.use_fp16 else ms.float32
    extra_conds = {}
    # 2.4 1) sktech - pidinet and cleaner
    if "single_sketch" in cfg.conditions_for_train or "sketch" in cfg.conditions_for_train:
        # sketch extractor
        pidinet = pidinet_bsd(
                pretrained=True, vanilla_cnn=True, ckpt_path=get_abspath_of_weights(cfg.pidinet_checkpoint)
            )
        pidinet = pidinet.set_train(False).to_float(cfg.dtype)
        for _, param in pidinet.parameters_and_names():
            param.requires_grad = False
        # cleaner
        cleaner = sketch_simplification_gan(
                pretrained=True, ckpt_path=get_abspath_of_weights(cfg.sketch_simplification_checkpoint)
            )
        cleaner = cleaner.set_train(False).to_float(cfg.dtype)
        for _, param in cleaner.parameters_and_names():
            param.requires_grad = False
        extra_conds['sketch'] = {'pidinet': pidinet, 'sketch_mean': cfg.sketch_mean, 'sketch_std': cfg.sketch_std, 'cleaner': cleaner}

        # TODO: set pidinet and cleaner as O3 amp

    # 2.4 2) depth - midas 
    if "depthmap" in cfg.conditions_for_train:
        midas = midas_v3_dpt_large(pretrained=True, ckpt_path=get_abspath_of_weights(cfg.midas_checkpoint))
        midas = midas.set_train(False).to_float(cfg.dtype)
        for _, param in midas.parameters_and_names():
            param.requires_grad = False
        extra_conds['depthmap'] = {'midas': midas, 'depth_clamp': cfg.depth_clamp, 'depth_std':cfg.depth_std}

        auto_mixed_precision(midas, amp_level="O3") # TODO: check output type fp16 or fp32
    
    # count num params for each network
    param_nums = {
            "vae": count_params(vae)[0],
            "clip_text_encoder": count_params(clip_text_encoder)[0],
            "clip_image_encoder": count_params(clip_image_encoder)[0],
            "unet with stc encoders": count_params(unet)[0],
            }
    for cond in extra_conds:
        for _net in extra_conds[cond]:
            print()
            if isinstance(extra_conds[cond][_net], ms.nn.Cell):
                param_nums[cond + '-' + _net] = count_params(extra_conds[cond][_net])[0] 
    logger.info("Param numbers: {}".format(param_nums))
    tot_params = sum([param_nums[module] for module in param_nums])
    logger.info("Total parameters: {:,}".format(tot_params))

    # 3. build latent diffusion with loss cell    
    ldm_with_loss = LatentDiffusion(
            unet,
            vae,
            clip_text_encoder,
            clip_image_encoder,
            extra_conds=extra_conds,
            use_fp16=cfg.use_fp16,
            timesteps=cfg.num_timesteps,
            parameterization=cfg.mean_type,
            scale_factor=cfg.scale_factor,
            cond_stage_trainable=cfg.cond_stage_trainable,
            linear_start=cfg.linear_start,
            linear_end=cfg.linear_end,
            ) 

    #auto_mixed_precision(ldm_with_loss, amp_level="O3") # TODO: debugging

    # 4. build training dataset
    dataloader = build_dataset(cfg, device_num, rank_id, tokenizer)
    num_batches = dataloader.get_dataset_size()

    # test match btw dataloader and diffusion model
    debug = False 
    if debug: 
        iterator = dataloader.create_dict_iterator()
        for i, batch in enumerate(iterator):
            for k in batch:
                print(k, batch[k].shape, batch[k].min(), batch[k].max())
            loss = ldm_with_loss(*list(batch.values()))
            print(loss)
    
    else:
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
        optimizer = build_optimizer(ldm_with_loss, cfg, learning_rate)
        loss_scaler = DynamicLossScaleUpdateCell(
            loss_scale_value=65536, scale_factor=2, scale_window=1000
            ) # TODO: tune

        net_with_grads = TrainOneStepWrapper(
            ldm_with_loss,
            optimizer=optimizer,
            scale_sense=loss_scaler,
            drop_overflow_update=True, #TODO: orignally True in SD, but here overflow keep happening, debug True
            gradient_accumulation_steps=1, # #args.gradient_accumulation_steps,
            clip_grad=False, #args.clip_grad,
            clip_norm=1.0, #args.max_grad_norm,
            ema=None, #ema,
        )

        model = Model(net_with_grads)

        # callbacks
        callback = [TimeMonitor(cfg.log_interval), LossMonitor(cfg.log_interval)]
        ofm_cb = OverflowMonitor()
        callback.append(ofm_cb)
        
        start_epoch = 0
        if rank_id == 0:
            save_cb = EvalSaveCallback(
                network=ldm_with_loss,  # TODO: save unet/vae seperately
                use_lora=False,
                rank_id=rank_id,
                ckpt_save_dir=cfg.output_dir,
                ema=None,
                ckpt_save_policy="latest_k",
                ckpt_max_keep=3,
                step_mode=False,
                ckpt_save_interval=cfg.ckpt_save_interval,
                log_interval=cfg.log_interval,
                start_epoch=start_epoch,
            )

            callback.append(save_cb)
    
        # train
        print("Start training. Please wait for graph compile (~20min)")
        model.train(cfg.epochs, dataloader, callbacks=callback, dataset_sink_mode=False, initial_epoch=start_epoch)

    
if __name__ == "__main__":
    # 0. parse config 
    from configs.train_config import cfg # base config from train_config.py

    args_for_update = Config(load=True).cfg_dict # config args from CLI (arg parser) and yaml files

    # update base config
    for k, v in args_for_update.items():
        cfg[k] = v
    
    print(cfg)
    
    main(cfg)
