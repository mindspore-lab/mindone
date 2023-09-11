import logging
import os

import mindspore as ms

from ..config import cfg
from ..data import make_masked_images
from ..diffusion import GaussianDiffusion, beta_schedule
from ..models import get_first_stage_encoding
from ..utils import save_video_multiple_conditions, setup_seed
from .inference_single import (
    extract_conditions,
    init,
    prepare_autoencoder_unet,
    prepare_clip_encoders,
    prepare_condition_models,
    prepare_dataloader,
    prepare_transforms,
)

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
    # logging
    input_video_name = os.path.basename(cfg.input_video).split(".")[0]
    init(gpu, cfg, video_name=input_video_name)
    transforms_list = prepare_transforms(cfg)
    dataloader = prepare_dataloader(cfg, transforms_list)

    clip_encoder, clip_encoder_visual = prepare_clip_encoders(cfg)
    zero_y = ms.ops.stop_gradient(clip_encoder.encode(""))  # [1, 77, 1024]
    black_image_feature = clip_encoder_visual.encode(clip_encoder_visual.black_image).unsqueeze(1)  # [1, 1, 1024]
    black_image_feature = ms.ops.zeros_like(black_image_feature)  # for old

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
            depth_data = extract_conditions(bs_vd, depth_extractor, misc_data_list)
        canny_data = []
        if "canny" in cfg.video_compositions:
            canny_data = extract_conditions(bs_vd, canny_extractor, misc_data_list)
        sketch_data = []
        if "sketch" in cfg.video_compositions:
            sketch_data = extract_conditions(bs_vd, sketch_extractor, misc_data_list)
        single_sketch_data = []
        if "single_sketch" in cfg.video_compositions:
            single_sketch_data = sketch_data.copy()[:, :, :1].tile((1, 1, frames_num, 1, 1))

        # preprocess for input text descripts
        y = clip_encoder.encode(caps)  # [1, 77, 1024]
        y0 = y.copy()
        y_visual = []
        if "image" in cfg.video_compositions:
            # with torch.no_grad():
            # ref_imgs = ref_imgs.squeeze(1)  # (1, 3, 224, 224) todo: torch.squeeze does nothing?
            y_visual = clip_encoder_visual.encode(ref_imgs).unsqueeze(1)  # [1, 1, 1024]
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
