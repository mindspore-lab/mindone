import datetime
import logging
import os
import time

from PIL import Image

import mindspore as ms
from mindspore import dataset as ds
from mindspore.dataset import transforms, vision

from ..annotator.canny import CannyDetector
from ..annotator.depth import midas_v3_dpt_large
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
    "inference_single",
]

_logger = logging.getLogger(__name__)


def inference_single(cfg_update, **kwargs):
    cfg.update(**kwargs)

    # Copy update input parameter to current task
    for k, v in cfg_update.items():
        cfg[k] = v

    cfg.read_image = getattr(cfg, "read_image", False)
    cfg.read_sketch = getattr(cfg, "read_sketch", False)
    cfg.read_style = getattr(cfg, "read_style", False)
    cfg.save_origin_video = getattr(cfg, "save_origin_video", True)

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


def init(gpu, cfg, video_name=None):
    ms.set_context(mode=cfg.mode)
    ms.set_context(ascend_config={"precision_mode": "allow_fp32_to_fp16"})  # Only effective on Ascend 901B
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
    # todo: need `all_gather` to get consistent `log_dir` between each proc if running in distributed mode.
    log_dir = cfg.log_dir
    if video_name is None:
        ct = datetime.datetime.now().strftime("-%y%m%d%H%M")
        exp_name = os.path.basename(cfg.cfg_file).split(".")[0] + "-S%05d" % (cfg.seed) + ct
    else:
        exp_name = os.path.basename(cfg.cfg_file).split(".")[0] + f"-{video_name}" + "-S%05d" % (cfg.seed)
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

    cfg.dtype = ms.float16 if cfg.use_fp16 else ms.float32


def prepare_transforms(cfg):
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
        ]
    )
    vit_transforms = transforms.Compose(
        [
            CenterCrop(cfg.vit_image_size),
            vision.ToTensor(),
            vision.Normalize(mean=cfg.vit_mean, std=cfg.vit_std, is_hwc=False),
        ]
    )
    return infer_transforms, misc_transforms, mv_transforms, vit_transforms


def prepare_dataloader(cfg, transforms_list):
    infer_transforms, misc_transforms, mv_transforms, vit_transforms = transforms_list
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
        mvs_visual=cfg.mvs_visual,
    )
    dataloader = ds.GeneratorDataset(
        source=dataset,
        column_names=["ref_frame", "cap_txt", "video_data", "misc_data", "feature_framerate", "mask", "mv_data"],
    )
    dataloader = dataloader.batch(1)
    return dataloader


def prepare_clip_encoders(cfg):
    clip_encoder = FrozenOpenCLIPEmbedder(
        layer="penultimate",
        pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint),
        tokenizer_path=get_abspath_of_weights(cfg.clip_tokenizer),
        use_fp16=cfg.use_fp16,
    )
    clip_encoder_visual = FrozenOpenCLIPVisualEmbedder(
        layer="penultimate", pretrained_ckpt_path=get_abspath_of_weights(cfg.clip_checkpoint), use_fp16=cfg.use_fp16
    )
    return clip_encoder, clip_encoder_visual


def read_image_if_provided(flag, path, transform=None, return_tensor=True, dtype=ms.float32):
    "read image `path` if `flag` is True, else return None"
    if not flag:
        return None
    img = Image.open(open(path, mode="rb")).convert("RGB")
    if transform is not None:
        img = transform(img)
    if return_tensor:
        img = ms.Tensor(img, dtype=dtype)
    return img


def prepare_condition_models(cfg):
    # [Conditions] Generators for various conditions
    if "depthmap" in cfg.video_compositions and "depth" in cfg.guidances:
        midas = midas_v3_dpt_large(pretrained=True, ckpt_path=get_abspath_of_weights(cfg.midas_checkpoint))
        midas = midas.set_train(False).to_float(cfg.dtype)
        for _, param in midas.parameters_and_names():
            param.requires_grad = False

        def depth_extractor(misc_imgs):
            depth = midas((misc_imgs - 0.5) / 0.5)
            depth = (depth / cfg.depth_std).clamp(0, cfg.depth_clamp)
            return depth

    else:
        depth_extractor = None

    if "canny" in cfg.video_compositions and "canny" in cfg.guidances:
        canny_detector = CannyDetector()

        def canny_extractor(misc_imgs):
            misc_imgs = ms.ops.transpose(misc_imgs.copy(), (0, 2, 3, 1))  # 'k' means 'chunk'.
            canny_condition = ms.ops.stack([canny_detector(misc_img) for misc_img in misc_imgs])
            canny_condition = ms.ops.transpose(canny_condition, (0, 3, 1, 2))
            return canny_condition

    else:
        canny_extractor = None

    if "sketch" in cfg.video_compositions and ("single_sketch" in cfg.guidances or "sketch" in cfg.guidances):
        pidinet = pidinet_bsd(
            pretrained=True, vanilla_cnn=True, ckpt_path=get_abspath_of_weights(cfg.pidinet_checkpoint)
        )
        pidinet = pidinet.set_train(False).to_float(cfg.dtype)
        for _, param in pidinet.parameters_and_names():
            param.requires_grad = False
        cleaner = sketch_simplification_gan(
            pretrained=True, ckpt_path=get_abspath_of_weights(cfg.sketch_simplification_checkpoint)
        )
        cleaner = cleaner.set_train(False).to_float(cfg.dtype)
        for _, param in cleaner.parameters_and_names():
            param.requires_grad = False
        pidi_mean = ms.Tensor(cfg.sketch_mean).view(1, -1, 1, 1)
        pidi_std = ms.Tensor(cfg.sketch_std).view(1, -1, 1, 1)

        def sketch_extractor(misc_imgs):
            sketch = pidinet((misc_imgs - pidi_mean) / pidi_std)
            sketch = 1.0 - cleaner(1.0 - sketch)
            return sketch

    else:
        sketch_extractor = None
    return depth_extractor, canny_extractor, sketch_extractor


def extract_conditions(batch_size, condition_extractor, data_list):
    cond_data = []
    for imgs in data_list:
        cond = condition_extractor(imgs)
        cond_data.append(cond)
    cond_data = ms.ops.cat(cond_data, axis=0)
    # (b f) c h w -> b f c h w -> b c f h w
    cond_data = ms.ops.reshape(cond_data, (batch_size, cond_data.shape[0] // batch_size, *cond_data.shape[1:]))
    cond_data = ms.ops.transpose(cond_data, (0, 2, 1, 3, 4))
    return cond_data


def prepare_autoencoder_unet(cfg, zero_y=None, black_image_feature=None, version="2.1"):
    # [Model] autoencoder & unet
    autoencoder = AutoencoderKL(
        cfg.ddconfig,
        4,
        ckpt_path=get_abspath_of_weights(cfg.sd_checkpoint),
        use_fp16=cfg.use_fp16,
        version=version,
    )
    autoencoder = autoencoder.set_train(False)
    for param in autoencoder.get_parameters():
        param.requires_grad = False

    if hasattr(cfg, "network_name") and cfg.network_name == "UNetSD_temporal":
        model = UNetSD_temporal(
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
            zero_y=zero_y,
            black_image_feature=black_image_feature,
            use_fp16=cfg.use_fp16,
            use_adaptive_pool=False,
        )
        model = model.set_train(False)
        for _, param in model.parameters_and_names():
            param.requires_grad = False
    else:
        raise NotImplementedError(f"The model {cfg.network_name} not implement")

    # load checkpoint
    if cfg.resume and cfg.resume_checkpoint:
        model.load_state_dict(cfg.resume_checkpoint)
        _logger.info(f"Successfully load unet from {cfg.resume_checkpoint}")
    else:
        raise ValueError(f"The checkpoint file {cfg.resume_checkpoint} is wrong ")
    _logger.info(
        f"Created a model with {int(sum(p.numel() for p in model.get_parameters()) / (1024 ** 2))}M parameters"
    )
    return autoencoder, model


def worker(gpu, cfg):
    init(gpu, cfg)
    transforms_list = prepare_transforms(cfg)
    misc_transforms = transforms_list[1]
    dataloader = prepare_dataloader(cfg, transforms_list)
    clip_encoder, clip_encoder_visual = prepare_clip_encoders(cfg)
    zero_y = ms.ops.stop_gradient(clip_encoder.encode([""]))  # [1, 77, 1024]
    # zero_y = None
    black_image_feature = clip_encoder_visual.encode(clip_encoder_visual.black_image).unsqueeze(1)  # [1, 1, 1024]
    black_image_feature = ms.ops.zeros_like(black_image_feature)  # for old

    frame_in = read_image_if_provided(cfg.read_image, cfg.image_path, misc_transforms, return_tensor=True)
    frame_sketch = read_image_if_provided(cfg.read_sketch, cfg.sketch_path, misc_transforms, return_tensor=True)
    frame_style = read_image_if_provided(cfg.read_style, cfg.style_image, None, return_tensor=False)

    depth_extractor, canny_extractor, sketch_extractor = prepare_condition_models(cfg)
    # Placeholder for color inference
    palette = None

    autoencoder, model = prepare_autoencoder_unet(cfg, zero_y, black_image_feature)
    # diffusion
    betas = beta_schedule("linear_sd", cfg.num_timesteps, init_beta=0.00085, last_beta=0.0120)
    diffusion = GaussianDiffusion(
        betas=betas, mean_type=cfg.mean_type, var_type=cfg.var_type, loss_type=cfg.loss_type, rescale_timesteps=False
    )

    # global variables
    viz_num = cfg.batch_size
    time_cost = []
    n_trials = 1  # set 4 for testing inference speed
    for tidx in range(n_trials):
        for step, batch in enumerate(dataloader.create_tuple_iterator()):
            start = time.time()
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
            if "motion" in cfg.video_compositions and "motion" in cfg.guidances:
                mv_data_video = ms.ops.transpose(mv_data, (0, 2, 1, 3, 4))
            # mask images
            masked_video = []
            if "mask" in cfg.video_compositions and "masked" in cfg.guidances:  # TODO: fix the name
                masked_video = make_masked_images((misc_data - 0.5) / 0.5, mask)
                masked_video = ms.ops.transpose(masked_video, (0, 2, 1, 3, 4))
            # Single Image
            image_local = []
            frames_num = misc_data.shape[1]
            if "local_image" in cfg.video_compositions and "local_image" in cfg.guidances:
                bs_vd_local = misc_data.shape[0]
                if cfg.read_image:
                    image_local = frame_in.unsqueeze(0).tile((bs_vd_local, frames_num, 1, 1, 1))
                else:
                    image_local = misc_data[:, :1].copy().tile((1, frames_num, 1, 1, 1))
                image_local = ms.ops.transpose(image_local, (0, 2, 1, 3, 4))  # (1, 3, 16, 384, 384)

            # encode the video_data
            bs_vd = video_data.shape[0]
            video_data_origin = video_data.copy()  # noqa
            # [bs, F, 3, 256, 256] -> (bs*f 3 256 256)
            video_data = ms.ops.reshape(video_data, (video_data.shape[0] * video_data.shape[1], *video_data.shape[2:]))
            video_data_list = ms.ops.chunk(video_data, video_data.shape[0] // cfg.chunk_size, axis=0)
            # print("D--: frames shape after chunk: ", video_data_list.shape)

            # [bs, F, 3, 384, 384] -> (bs*f 3 384 384)
            misc_data = ms.ops.reshape(misc_data, (misc_data.shape[0] * misc_data.shape[1], *misc_data.shape[2:]))

            misc_data_list = ms.ops.chunk(misc_data, misc_data.shape[0] // cfg.chunk_size, axis=0)

            decode_data = []
            # TODO:
            for vd_data in video_data_list:
                encoder_posterior = autoencoder.encode(vd_data)
                tmp = get_first_stage_encoding(encoder_posterior)
                decode_data.append(tmp)
            video_data = ms.ops.cat(decode_data, axis=0)
            # (b f) c h w -> b f c h w -> b c f h w
            video_data = ms.ops.reshape(video_data, (bs_vd, video_data.shape[0] // bs_vd, *video_data.shape[1:]))
            video_data = ms.ops.transpose(video_data, (0, 2, 1, 3, 4))

            depth_data = []
            if "depthmap" in cfg.video_compositions and "depth" in cfg.guidances:
                depth_data = extract_conditions(bs_vd, depth_extractor, misc_data_list)
            canny_data = []
            if "canny" in cfg.video_compositions and "canny" in cfg.guidances:
                canny_data = extract_conditions(bs_vd, canny_extractor, misc_data_list)
            sketch_data = []
            if "sketch" in cfg.video_compositions and ("single_sketch" in cfg.guidances or "sketch" in cfg.guidances):
                sketch_list = misc_data_list
                if cfg.read_sketch:
                    sketch_repeat = frame_sketch.tile((frames_num, 1, 1, 1))
                    sketch_list = [sketch_repeat]
                sketch_data = extract_conditions(bs_vd, sketch_extractor, sketch_list)
            single_sketch_data = []
            if "single_sketch" in cfg.video_compositions and "single_sketch" in cfg.guidances:
                single_sketch_data = sketch_data.copy()[:, :, :1].tile((1, 1, frames_num, 1, 1))

            # preprocess for input text descripts
            y = clip_encoder.encode(caps)  # [1, 77, 1024]
            y0 = y.copy()
            y_visual = []
            if "image" in cfg.video_compositions and "image" in cfg.guidances:  # TODO: check
                # with torch.no_grad():
                if cfg.read_style:
                    y_visual = clip_encoder_visual.encode(frame_style).unsqueeze(0)  # from style is Image raw
                    y_visual0 = y_visual.copy()
                else:
                    y_visual = clip_encoder_visual(ref_imgs).unsqueeze(
                        1
                    )  # [1, 1, 1024], since ref_imgs has been processed via vit_transform to nchw normalized tensor
                    y_visual0 = y_visual.copy()

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
            # --------------------------------------
            partial_keys = cfg.guidances
            noise_motion = noise.copy()
            model_kwargs = prepare_model_kwargs(
                partial_keys=partial_keys,
                full_model_kwargs=full_model_kwargs,
                use_fps_condition=cfg.use_fps_condition,
            )
            video_output = diffusion.ddim_sample_loop(
                noise=noise_motion,
                model=model.set_train(False),
                model_kwargs=model_kwargs,
                guide_scale=9.0,
                ddim_timesteps=cfg.ddim_timesteps,
                eta=0.0,
            )

            visualize_with_model_kwargs(
                model_kwargs=model_kwargs,
                video_data=video_output,
                autoencoder=autoencoder,
                ori_video=misc_backups,
                viz_num=viz_num,
                step=step,
                caps=caps,
                palette=palette,
                cfg=cfg,
                sample_idx=tidx,
            )
            # --------------------------------------

            time_cost.append(time.time() - start)
            print("D--: ", time_cost)
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
            "single_sketch",
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
    sample_idx=0,
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

    oss_key = os.path.join(cfg.log_dir, f"Sample_{sample_idx}.gif")
    text_key = os.path.join(cfg.log_dir, "text_description.txt")

    # Save videos and text inputs.
    # try:
    del model_kwargs[0][list(model_kwargs[0].keys())[0]]
    del model_kwargs[1][list(model_kwargs[1].keys())[0]]
    save_video_multiple_conditions(
        oss_key,
        video_data,
        model_kwargs,
        ori_video,
        palette,
        cfg.mean,
        cfg.std,
        nrow=1,
        save_origin_video=cfg.save_origin_video,
    )
    if cfg.rank == 0:
        texts = "\n".join(caps[:viz_num])
        open(text_key, "w").writelines(texts)

    _logger.info(f"Successfully saved videos to {oss_key}")
    # except Exception as e:
    #    _logger.warning(f"Got an error when saving text or video: {e}")
