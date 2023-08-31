import logging
import os

import mindspore as ms
from mindspore import dataset as ds
from mindspore.dataset import transforms, vision

from ..annotator.canny import CannyDetector
from ..annotator.depth import midas_v3
from ..annotator.sketch import pidinet_bsd, sketch_simplification_gan
from ..config import cfg
from ..data import CenterCrop, RandomResize, VideoDataset, make_masked_images
from ..diffusion import GaussianDiffusion, beta_schedule
from ..models import (
    AutoencoderKL,
    FrozenOpenCLIPEmbedder,
    FrozenOpenCLIPVisualEmbedder,
    UNetSD_temporal,
    get_first_stage_encoding,
)
from ..utils import get_abspath_of_weights, save_video_multiple_conditions, setup_logger, setup_seed

__all__ = [
    "inference_multi",
]

_logger = logging.getLogger(__name__)


def inference_multi(cfg_update, **kwargs):
    cfg.update(**kwargs)

    # Copy update input parameter to current task
    for k, v in cfg_update.items():
        cfg[k] = v

    cfg.pmi_rank = int(os.getenv("RANK", 0))  # rank_of_node?
    cfg.pmi_world_size = int(os.getenv("WORLD_SIZE", 1))  # n_nodes?
    setup_seed(cfg.seed)

    cfg.gpus_per_machine = 1
    cfg.world_size = 1
    # if cfg.debug:
    #     cfg.gpus_per_machine = 1
    #     cfg.world_size = 1
    # else:
    #     # LOCAL_WORLD_SIZE - The local world size (e.g. number of workers running locally); (nproc-per-node)
    #     cfg.gpus_per_machine = ms.communication.get_group_size()
    #     # WORLD_SIZE - The world size (total number of workers in the job).
    #     cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine

    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        # mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg,))
        raise NotImplementedError("Distributed inference is not supported yet!")
    return cfg


def worker(gpu, cfg):
    # LOCAL_RANK - The local rank.
    cfg.gpu = gpu
    # RANK - The global rank.
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu

    if cfg.world_size > 1:
        # init distributed processes
        ms.communication.init()
        rank_id, device_num = ms.communication.get_rank(), ms.communication.get_group_size()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
        )
    else:
        rank_id, device_num = 0, 1
    assert rank_id == cfg.rank and device_num == cfg.world_size

    # logging
    input_video_name = os.path.basename(cfg.input_video).split(".")[0]
    # todo: need `all_gather` to get consistent `log_dir` between each proc if running in distributed mode.
    log_dir = cfg.log_dir
    exp_name = os.path.basename(cfg.cfg_file).split(".")[0] + f"-{input_video_name}" + "-S%05d" % (cfg.seed)
    log_dir = os.path.join(log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir
    setup_logger(output_dir=cfg.log_dir, rank=cfg.rank)
    _logger.info(cfg)

    # rank-wise params
    l1 = len(cfg.frame_lens)
    l2 = len(cfg.feature_framerates)
    cfg.max_frames = cfg.frame_lens[cfg.rank % (l1 * l2) // l2]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]

    # [Transform] Transforms for different inputs
    infer_transforms = transforms.Compose(
        [
            vision.CenterCrop(size=cfg.resolution),
            vision.ToTensor(),
            vision.Normalize(mean=cfg.mean, std=cfg.std, is_hwc=False),
        ]
    )
    misc_transforms = transforms.Compose(
        [
            RandomResize(size=cfg.misc_size),
            vision.CenterCrop(cfg.misc_size),
            vision.ToTensor(),
        ]
    )
    mv_transforms = transforms.Compose(
        [
            vision.Resize(size=cfg.resolution),
            vision.CenterCrop(cfg.resolution),
        ],
    )
    vit_transforms = transforms.Compose(
        [
            CenterCrop(cfg.vit_image_size),
            vision.ToTensor(),
            vision.Normalize(mean=cfg.vit_mean, std=cfg.vit_std, is_hwc=False),
        ]
    )

    dataset = VideoDataset(
        cfg=cfg,
        max_words=cfg.max_words,
        feature_framerate=cfg.feature_framerate,
        max_frames=cfg.max_frames,
        image_resolution=cfg.resolution,
        transforms=infer_transforms,
        mv_transforms=mv_transforms,
        misc_transforms=misc_transforms,
        vit_transforms=vit_transforms,
        vit_image_size=cfg.vit_image_size,
        misc_size=cfg.misc_size,
    )
    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=["ref_frame", "cap_txt", "video_data", "misc_data", "feature_framerate", "mask", "mv_data"],
    )
    dataloader = dataloader.batch(1)

    clip_encoder = FrozenOpenCLIPEmbedder(
        layer="penultimate",
        pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint),
        tokenizer_path=get_abspath_of_weights(cfg.clip_tokenizer),
    )
    zero_y = ms.ops.stop_gradient(clip_encoder("")).to(ms.float32)  # [1, 77, 1024]
    clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(
        layer="penultimate", pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint)
    )
    black_image_feature = (
        clip_encoder_visual(clip_encoder_visual.black_image).unsqueeze(1).to(ms.float32)
    )  # [1, 1, 1024]
    black_image_feature = ms.ops.zeros_like(black_image_feature)  # for old

    # [Conditions] Generators for various conditions
    if "depthmap" in cfg.video_compositions:
        midas = midas_v3(pretrained=True, ckpt_path=get_abspath_of_weights(cfg.midas_checkpoint))
        midas = midas.set_train(False).to_float(ms.float32)
    if "canny" in cfg.video_compositions:
        canny_detector = CannyDetector()
    if "sketch" in cfg.video_compositions:
        pidinet = pidinet_bsd(
            pretrained=True, vanilla_cnn=True, ckpt_path=get_abspath_of_weights(cfg.pidinet_checkpoint)
        )
        pidinet = pidinet.set_train(False).to_float(ms.float32)
        cleaner = sketch_simplification_gan(
            pretrained=True, ckpt_path=get_abspath_of_weights(cfg.sketch_simplification_checkpoint)
        )
        cleaner = cleaner.set_train(False).to_float(ms.float32)
        pidi_mean = ms.Tensor(cfg.sketch_mean).view(1, -1, 1, 1)
        pidi_std = ms.Tensor(cfg.sketch_std).view(1, -1, 1, 1)
    # Placeholder for color inference
    palette = None

    # [Model] autoencoder & unet
    ddconfig = {
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
    autoencoder = AutoencoderKL(ddconfig, 4, ckpt_path=get_abspath_of_weights(cfg.sd_checkpoint))
    autoencoder = autoencoder.set_train(False).to_float(ms.float32)
    for param in autoencoder.get_parameters():
        param.requires_grad = False

    if hasattr(cfg, "network_name") and cfg.network_name == "UNetSD_temporal":
        model = UNetSD_temporal(
            cfg=cfg,
            in_dim=cfg.unet_in_dim,
            concat_dim=cfg.unet_concat_dim,
            dim=cfg.unet_dim,
            y_dim=cfg.unet_y_dim,
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
            zero_y=zero_y,
            black_image_feature=black_image_feature,
        )
        model = model.set_train(False).to_float(ms.float32)
    else:
        raise NotImplementedError(f"The model {cfg.network_name} not implement")

    # load checkpoint
    resume_step = 1
    if cfg.resume and cfg.resume_checkpoint:
        if hasattr(cfg, "text_to_video_pretrain") and cfg.text_to_video_pretrain:
            model.load_state_dict(get_abspath_of_weights(cfg.resume_checkpoint), text_to_video_pretrain=True)
        else:
            model.load_state_dict(get_abspath_of_weights(cfg.resume_checkpoint), text_to_video_pretrain=False)
        if cfg.resume_step:
            resume_step = cfg.resume_step
        _logger.info(f"Successfully load step {resume_step} model from {cfg.resume_checkpoint}")
    else:
        raise ValueError(f"The checkpoint file {cfg.resume_checkpoint} is wrong ")
    _logger.info(
        f"Created a model with {int(sum(p.numel() for p in model.get_parameters()) / (1024 ** 2))}M parameters"
    )

    # diffusion
    betas = beta_schedule("linear_sd", cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
    diffusion = GaussianDiffusion(
        betas=betas, mean_type=cfg.mean_type, var_type=cfg.var_type, loss_type=cfg.loss_type, rescale_timesteps=False
    )

    # global variables
    viz_num = cfg.batch_size
    for step, batch in enumerate(dataloader.create_tuple_iterator()):
        model.set_train(False)

        caps = batch[1].numpy().tolist()
        del batch[1]
        if cfg.max_frames == 1 and cfg.use_image_dataset:
            ref_imgs, video_data, misc_data, mask, mv_data = batch
            fps = ms.Tensor([cfg.feature_framerate] * cfg.batch_size, dtype=ms.int64)
        else:
            ref_imgs, video_data, misc_data, fps, mask, mv_data = batch

        # save for visualization
        misc_backups = misc_data.copy()
        misc_backups = ms.ops.transpose(misc_backups, (0, 2, 1, 3, 4))
        mv_data_video = []
        if "motion" in cfg.video_compositions:
            mv_data_video = ms.ops.transpose(mv_data, (0, 2, 1, 3, 4))
        # mask images
        masked_video = []
        if "mask" in cfg.video_compositions:
            masked_video = make_masked_images((misc_data - 0.5) / 0.5, mask)
            masked_video = ms.ops.transpose(masked_video, (0, 2, 1, 3, 4))
        image_local = []
        if "local_image" in cfg.video_compositions:
            frames_num = misc_data.shape[1]
            bs_vd_local = misc_data.shape[0]  # noqa
            image_local = misc_data[:, :1].copy().tile((1, frames_num, 1, 1, 1))
            image_local = ms.ops.transpose(image_local, (0, 2, 1, 3, 4))

        # encode the video_data
        bs_vd = video_data.shape[0]
        video_data_origin = video_data.copy()  # noqa
        video_data = ms.ops.reshape(video_data, (video_data.shape[0] * video_data.shape[1], *video_data.shape[2:]))
        misc_data = ms.ops.reshape(misc_data, (misc_data.shape[0] * misc_data.shape[1], *misc_data.shape[2:]))

        video_data_list = ms.ops.chunk(video_data, video_data.shape[0] // cfg.chunk_size, axis=0)
        misc_data_list = ms.ops.chunk(misc_data, misc_data.shape[0] // cfg.chunk_size, axis=0)

        # with torch.no_grad() start
        decode_data = []
        for vd_data in video_data_list:
            encoder_posterior = autoencoder.encode(vd_data)
            tmp = get_first_stage_encoding(encoder_posterior)
            decode_data.append(tmp)
        video_data = ms.ops.cat(decode_data, axis=0)
        # (b f) c h w -> b f c h w -> b c f h w
        video_data = ms.ops.reshape(video_data, (bs_vd, video_data.shape[0] // bs_vd, *video_data.shape[1:]))
        video_data = ms.ops.transpose(video_data, (0, 2, 1, 3, 4))

        depth_data = []
        if "depthmap" in cfg.video_compositions:
            for misc_imgs in misc_data_list:
                depth = midas((misc_imgs - 0.5) / 0.5)
                depth = (depth / cfg.depth_std).clamp(0, cfg.depth_clamp)
                depth_data.append(depth)
            depth_data = ms.ops.cat(depth_data, axis=0)
            # (b f) c h w -> b f c h w -> b c f h w
            depth_data = ms.ops.reshape(depth_data, (bs_vd, depth_data.shape[0] // bs_vd, *depth_data.shape[1:]))
            depth_data = ms.ops.transpose(depth_data, (0, 2, 1, 3, 4))

        canny_data = []
        if "canny" in cfg.video_compositions:
            for misc_imgs in misc_data_list:
                # print(misc_imgs.shape)
                misc_imgs = ms.ops.transpose(misc_imgs.copy(), (0, 2, 3, 1))  # 'k' means 'chunk'.
                canny_condition = ms.ops.stack([canny_detector(misc_img) for misc_img in misc_imgs])
                canny_condition = ms.ops.transpose(canny_condition, (0, 3, 1, 2))
                canny_data.append(canny_condition)
            canny_data = ms.ops.cat(canny_data, axis=0)
            # (b f) c h w -> b f c h w -> b c f h w
            canny_data = ms.ops.reshape(canny_data, (bs_vd, canny_data.shape[0] // bs_vd, *canny_data.shape[1:]))
            canny_data = ms.ops.transpose(canny_data, (0, 2, 1, 3, 4))

        sketch_data = []
        if "sketch" in cfg.video_compositions:
            for misc_imgs in misc_data_list:
                sketch = pidinet((misc_imgs - pidi_mean) / pidi_std)
                sketch = 1.0 - cleaner(1.0 - sketch)
                sketch_data.append(sketch)
            sketch_data = ms.ops.cat(sketch_data, axis=0)
            # (b f) c h w -> b f c h w -> b c f h w
            sketch_data = ms.ops.reshape(sketch_data, (bs_vd, sketch_data.shape[0] // bs_vd, *sketch_data.shape[1:]))
            sketch_data = ms.ops.transpose(sketch_data, (0, 2, 1, 3, 4))

        single_sketch_data = []
        if "single_sketch" in cfg.video_compositions:
            single_sketch_data = sketch_data.copy()[:, :, :1].tile((1, 1, frames_num, 1, 1))
        # with torch.no_grad() end

        # preprocess for input text descripts
        y = clip_encoder(caps).to(ms.float32)  # [1, 77, 1024]
        y0 = y.copy()
        y_visual = []
        if "image" in cfg.video_compositions:
            # with torch.no_grad():
            # ref_imgs = ref_imgs.squeeze(1)  # (1, 3, 224, 224) todo: torch.squeeze does nothing?
            y_visual = clip_encoder_visual(ref_imgs).unsqueeze(1).to(ms.float32)  # [1, 1, 1024]
            y_visual0 = y_visual.copy()

        # with torch.no_grad() start
        # Log memory
        # pynvml.nvmlInit()
        # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # logging.info(f"GPU Memory used {meminfo.used / (1024 ** 3):.2f} GB")
        # Sample images (DDIM)
        # with amp.autocast(enabled=cfg.use_fp16) start
        if cfg.share_noise:
            b, c, f, h, w = video_data.shape
            noise = ms.ops.randn((viz_num, c, h, w))
            noise = noise.repeat_interleave(repeats=f, dim=0)
            # (b f) c h w -> b c f h w
            noise = ms.ops.reshape(noise, (viz_num, noise.shape[0] // viz_num, *noise.shape[1:]))
            noise = ms.ops.transpose(noise, (0, 2, 1, 3, 4))
        else:
            noise = ms.ops.randn_like(video_data[:viz_num])

        full_model_kwargs = [
            {
                "y": y0[:viz_num],
                "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                "image": None if len(y_visual) == 0 else y_visual0[:viz_num],
                "depth": None if len(depth_data) == 0 else depth_data[:viz_num],
                "canny": None if len(canny_data) == 0 else canny_data[:viz_num],
                "sketch": None if len(sketch_data) == 0 else sketch_data[:viz_num],
                "masked": None if len(masked_video) == 0 else masked_video[:viz_num],
                "motion": None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                "single_sketch": None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                "fps": fps[:viz_num],
            },
            {
                "y": zero_y.tile((viz_num, 1, 1)) if not cfg.use_fps_condition else ms.ops.zeros_like(y0)[:viz_num],
                "local_image": None if len(image_local) == 0 else image_local[:viz_num],
                "image": None if len(y_visual) == 0 else ms.ops.zeros_like(y_visual0[:viz_num]),
                "depth": None if len(depth_data) == 0 else depth_data[:viz_num],
                "canny": None if len(canny_data) == 0 else canny_data[:viz_num],
                "sketch": None if len(sketch_data) == 0 else sketch_data[:viz_num],
                "masked": None if len(masked_video) == 0 else masked_video[:viz_num],
                "motion": None if len(mv_data_video) == 0 else mv_data_video[:viz_num],
                "single_sketch": None if len(single_sketch_data) == 0 else single_sketch_data[:viz_num],
                "fps": fps[:viz_num],
            },
        ]

        # Save generated videos
        # ---------------- txt + Motion -----------
        partial_keys_motion = ["y", "motion"]
        noise_motion = noise.copy()
        model_kwargs_motion = prepare_model_kwargs(
            partial_keys=partial_keys_motion,
            full_model_kwargs=full_model_kwargs,
            use_fps_condition=cfg.use_fps_condition,
        )
        video_data_motion = diffusion.ddim_sample_loop(
            noise=noise_motion,
            model=model.set_train(False),
            model_kwargs=model_kwargs_motion,
            guide_scale=9.0,
            ddim_timesteps=cfg.ddim_timesteps,
            eta=0.0,
        )
        visualize_with_model_kwargs(
            model_kwargs=model_kwargs_motion,
            video_data=video_data_motion,
            autoencoder=autoencoder,
            ori_video=misc_backups,
            viz_num=viz_num,
            step=step,
            caps=caps,
            palette=palette,
            cfg=cfg,
        )
        # --------------------------------------

        # ---------------- txt + Sketch --------
        partial_keys_1 = ["y", "sketch"]
        noise_1 = noise.copy()
        model_kwargs_1 = prepare_model_kwargs(
            partial_keys=partial_keys_1,
            full_model_kwargs=full_model_kwargs,
            use_fps_condition=cfg.use_fps_condition,
        )
        video_data_1 = diffusion.ddim_sample_loop(
            noise=noise_1,
            model=model.set_train(False),
            model_kwargs=model_kwargs_1,
            guide_scale=9.0,
            ddim_timesteps=cfg.ddim_timesteps,
            eta=0.0,
        )
        visualize_with_model_kwargs(
            model_kwargs=model_kwargs_1,
            video_data=video_data_1,
            autoencoder=autoencoder,
            ori_video=misc_backups,
            viz_num=viz_num,
            step=step,
            caps=caps,
            palette=palette,
            cfg=cfg,
        )
        # --------------------------------------

        # ---------------- txt + Depth --------
        partial_keys_2 = ["y", "depth"]
        noise_2 = noise.copy()
        model_kwargs_2 = prepare_model_kwargs(
            partial_keys=partial_keys_2,
            full_model_kwargs=full_model_kwargs,
            use_fps_condition=cfg.use_fps_condition,
        )
        video_data_2 = diffusion.ddim_sample_loop(
            noise=noise_2,
            model=model.set_train(False),
            model_kwargs=model_kwargs_2,
            guide_scale=9.0,
            ddim_timesteps=cfg.ddim_timesteps,
            eta=0.0,
        )
        visualize_with_model_kwargs(
            model_kwargs=model_kwargs_2,
            video_data=video_data_2,
            autoencoder=autoencoder,
            ori_video=misc_backups,
            viz_num=viz_num,
            step=step,
            caps=caps,
            palette=palette,
            cfg=cfg,
        )
        # --------------------------------------

        # ---------------- txt + local_image --------
        partial_keys_2_local_image = ["y", "local_image"]
        noise_2_local_image = noise.copy()
        model_kwargs_2_local_image = prepare_model_kwargs(
            partial_keys=partial_keys_2_local_image,
            full_model_kwargs=full_model_kwargs,
            use_fps_condition=cfg.use_fps_condition,
        )
        video_data_2_local_image = diffusion.ddim_sample_loop(
            noise=noise_2_local_image,
            model=model.set_train(False),
            model_kwargs=model_kwargs_2_local_image,
            guide_scale=9.0,
            ddim_timesteps=cfg.ddim_timesteps,
            eta=0.0,
        )
        visualize_with_model_kwargs(
            model_kwargs=model_kwargs_2_local_image,
            video_data=video_data_2_local_image,
            autoencoder=autoencoder,
            ori_video=misc_backups,
            viz_num=viz_num,
            step=step,
            caps=caps,
            palette=palette,
            cfg=cfg,
        )
        # --------------------------------------

        # ---------------- image + depth --------
        partial_keys_2_image = ["image", "depth"]
        noise_2_image = noise.copy()
        model_kwargs_2_image = prepare_model_kwargs(
            partial_keys=partial_keys_2_image,
            full_model_kwargs=full_model_kwargs,
            use_fps_condition=cfg.use_fps_condition,
        )
        video_data_2_image = diffusion.ddim_sample_loop(
            noise=noise_2_image,
            model=model.set_train(False),
            model_kwargs=model_kwargs_2_image,
            guide_scale=9.0,
            ddim_timesteps=cfg.ddim_timesteps,
            eta=0.0,
        )
        visualize_with_model_kwargs(
            model_kwargs=model_kwargs_2_image,
            video_data=video_data_2_image,
            autoencoder=autoencoder,
            ori_video=misc_backups,
            viz_num=viz_num,
            step=step,
            caps=caps,
            palette=palette,
            cfg=cfg,
        )
        # --------------------------------------

        # ---------------- text + mask --------
        partial_keys_3 = ["y", "masked"]
        noise_3 = noise.copy()
        model_kwargs_3 = prepare_model_kwargs(
            partial_keys=partial_keys_3,
            full_model_kwargs=full_model_kwargs,
            use_fps_condition=cfg.use_fps_condition,
        )
        video_data_3 = diffusion.ddim_sample_loop(
            noise=noise_3,
            model=model.set_train(False),
            model_kwargs=model_kwargs_3,
            guide_scale=9.0,
            ddim_timesteps=cfg.ddim_timesteps,
            eta=0.0,
        )
        visualize_with_model_kwargs(
            model_kwargs=model_kwargs_3,
            video_data=video_data_3,
            autoencoder=autoencoder,
            ori_video=misc_backups,
            viz_num=viz_num,
            step=step,
            caps=caps,
            palette=palette,
            cfg=cfg,
        )
        # --------------------------------------
        # with amp.autocast(enabled=cfg.use_fp16) end
        # with torch.no_grad() end

    _logger.info("Congratulations! The inference is completed!")


def prepare_model_kwargs(partial_keys, full_model_kwargs, use_fps_condition):
    for partial_key in partial_keys:
        assert partial_key in [
            "y",
            "depth",
            "canny",
            "masked",
            "sketch",
            "image",
            "motion",
            "local_image",
        ]

    if use_fps_condition is True:
        partial_keys.append("fps")

    partial_model_kwargs = [{}, {}]
    for partial_key in partial_keys:
        partial_model_kwargs[0][partial_key] = full_model_kwargs[0][partial_key]
        partial_model_kwargs[1][partial_key] = full_model_kwargs[1][partial_key]

    return partial_model_kwargs


def visualize_with_model_kwargs(
    model_kwargs,
    video_data,
    autoencoder,
    # ref_imgs,
    ori_video,
    viz_num,
    step,
    caps,
    palette,
    cfg,
):
    scale_factor = 0.18215
    video_data = 1.0 / scale_factor * video_data

    bs_vd = video_data.shape[0]
    b, c, f, h, w = video_data.shape
    video_data = ms.ops.transpose(video_data, (0, 2, 1, 3, 4))
    video_data = ms.ops.reshape(video_data, (b * f, c, h, w))
    chunk_size = min(16, video_data.shape[0])
    video_data_list = ms.ops.chunk(video_data, video_data.shape[0] // chunk_size, axis=0)
    decode_data = []
    for vd_data in video_data_list:
        tmp = autoencoder.decode(vd_data)
        decode_data.append(tmp)
    video_data = ms.ops.cat(decode_data, axis=0)
    bf, c, h, w = video_data.shape
    b, f = bs_vd, bf // bs_vd
    video_data = ms.ops.reshape(video_data, (b, f, c, h, w))
    video_data = ms.ops.transpose(video_data, (0, 2, 1, 3, 4))
    ori_video = ori_video[:viz_num]

    # upload conditional texts and videos
    oss_key_dir = os.path.join(cfg.log_dir, f"step_{step}" + "-" + f"{'_'.join(model_kwargs[0].keys())}")
    oss_key = os.path.join(oss_key_dir, f"rank_{cfg.world_size}-{cfg.rank}.gif")
    text_key = os.path.join(cfg.log_dir, "text_description.txt")
    if not os.path.exists(oss_key_dir):
        os.makedirs(oss_key_dir, exist_ok=True)

    # Save videos and text inputs.
    try:
        del model_kwargs[0][list(model_kwargs[0].keys())[0]]
        del model_kwargs[1][list(model_kwargs[1].keys())[0]]
        save_video_multiple_conditions(oss_key, video_data, model_kwargs, ori_video, palette, cfg.mean, cfg.std, nrow=1)
        if cfg.rank == 0:
            texts = "\n".join(caps[:viz_num])
            open(text_key, "w").writelines(texts)
    except Exception as e:
        _logger.warning(f"Got an error when saving text or video: {e}")

    _logger.info(f"Successfully saved videos to {oss_key}")
