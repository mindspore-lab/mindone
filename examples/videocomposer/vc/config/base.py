import os

from easydict import EasyDict

cfg = EasyDict(__name__="Config: VideoComposer")

cfg.video_compositions = ["text", "mask", "depthmap", "sketch", "motion", "image", "local_image", "single_sketch"]
cfg.midas_checkpoint = "midas_v3_dpt_large-c8fd1049.ckpt"
cfg.pidinet_checkpoint = "table5_pidinet-37904a63.ckpt"
cfg.sketch_simplification_checkpoint = "sketch_simplification_gan-b928fdfa.ckpt"

# dataset
cfg.root_dir = "webvid10m/"
cfg.alpha = 0.7
cfg.misc_size = 384
cfg.depth_std = 20.0
cfg.depth_clamp = 10.0
cfg.hist_sigma = 10.0
cfg.use_image_dataset = False
cfg.alpha_img = 0.7
cfg.resolution = 256
cfg.mean = [0.5, 0.5, 0.5]
cfg.std = [0.5, 0.5, 0.5]

# sketch
cfg.sketch_mean = [0.485, 0.456, 0.406]
cfg.sketch_std = [0.229, 0.224, 0.225]

# dataloader
cfg.max_words = 1000
cfg.max_frames = 16
cfg.feature_framerate = 4
cfg.batch_size = 1
cfg.chunk_size = 64
cfg.num_workers = 8  # not used yet
cfg.prefetch_factor = 2
cfg.seed = 8888

# diffusion
cfg.num_timesteps = 1000
cfg.mean_type = "eps"
cfg.var_type = "fixed_small"  # NOTE: to stabilize training and avoid NaN
cfg.loss_type = "mse"
cfg.ddim_eta = 0.0
cfg.clamp = 1.0
cfg.share_noise = False
cfg.use_div_loss = False

# classifier-free guidance
cfg.p_zero = 0.9

# stable diffusion
cfg.sd_checkpoint = "sd_v2-1_base-7c8d09ce.ckpt"
cfg.ddconfig = {
    "double_z": True,
    "z_channels": 4,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
}

# clip vision encoder
cfg.vit_image_size = 336
cfg.vit_patch_size = 14
cfg.vit_dim = 1024
cfg.vit_out_dim = 768
cfg.vit_heads = 16
cfg.vit_layers = 24
cfg.vit_mean = [0.48145466, 0.4578275, 0.40821073]
cfg.vit_std = [0.26862954, 0.26130258, 0.27577711]
cfg.clip_checkpoint = "open_clip_vit_h_14-9bb07a10.ckpt"
cfg.clip_tokenizer = "bpe_simple_vocab_16e6.txt.gz"
cfg.mvs_visual = False

# unet
cfg.unet_in_dim = 4
cfg.unet_concat_dim = 8
cfg.unet_y_dim = cfg.vit_out_dim
cfg.unet_context_dim = 1024
cfg.unet_out_dim = 8 if cfg.var_type.startswith("learned") else 4
cfg.unet_dim = 320
cfg.unet_dim_mult = [1, 2, 4, 4]
cfg.unet_res_blocks = 2
cfg.unet_num_heads = 8
cfg.unet_head_dim = 64
cfg.unet_attn_scales = [1 / 1, 1 / 2, 1 / 4]
cfg.unet_dropout = 0.1
cfg.misc_dropout = 0.5
cfg.p_all_zero = 0.1
cfg.p_all_keep = 0.1
cfg.temporal_conv = False
cfg.temporal_attn_times = 1
cfg.temporal_attention = True
cfg.use_fps_condition = False
cfg.use_sim_mask = False

# load 2d pretrain
cfg.pretrained = False
cfg.fix_weight = False

# resume
cfg.resume = True
cfg.resume_step = 148000
cfg.resume_check_dir = "."
cfg.resume_checkpoint = os.path.join(cfg.resume_check_dir, f"step_{cfg.resume_step}/non_ema_{cfg.resume_step}.pth")
cfg.resume_optimizer = False
if cfg.resume_optimizer:
    cfg.resume_optimizer = os.path.join(cfg.resume_check_dir, f"optimizer_step_{cfg.resume_step}.pt")

# acceleration
cfg.use_ema = True
cfg.load_from = None
cfg.use_checkpoint = False
cfg.use_sharded_ddp = False
cfg.use_fsdp = False
cfg.use_fp16 = True

# training
cfg.ema_decay = 0.9999
cfg.viz_interval = 1000
cfg.save_ckp_interval = 1000

# logging
cfg.log_interval = 100
cfg.log_dir = "outputs/"
