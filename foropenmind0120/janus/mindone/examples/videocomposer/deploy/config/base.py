from easydict import EasyDict

cfg = EasyDict(__name__="Config: VideoComposer")

cfg.video_compositions = ["text", "mask", "depthmap", "sketch", "motion", "image", "local_image", "single_sketch"]


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

# dataloader
cfg.max_words = 1000
cfg.max_frames = 16
cfg.feature_framerate = 4
cfg.batch_size = 1
cfg.chunk_size = 64
cfg.num_workers = 8  # not used yet
cfg.prefetch_factor = 2
cfg.seed = 8888

# clip
cfg.clip_tokenizer = "bpe_simple_vocab_16e6.txt.gz"

# unet
cfg.use_fps_condition = False
cfg.unet_in_dim = 4
cfg.unet_dim_mult = [1, 2, 4, 4]

# diffusion
cfg.num_timesteps = 1000
cfg.share_noise = False
cfg.ddim_eta = 0.0

# clip vision encoder
cfg.vit_mean = [0.48145466, 0.4578275, 0.40821073]
cfg.vit_std = [0.26862954, 0.26130258, 0.27577711]

# logging
cfg.log_interval = 100
cfg.log_dir = "outputs/"
