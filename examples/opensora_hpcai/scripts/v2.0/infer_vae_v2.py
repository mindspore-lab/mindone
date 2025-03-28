import argparse
import logging
import os
import sys
import time
from pathlib import Path
import random
from typing import Optional

import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore import mint, nn
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../../")))

from opensora.datasets.video_dataset import create_dataloader
# from opensora.models.vae.vae import SD_CONFIG, AutoencoderKL
from opensora.models.hunyuan_vae import CausalVAE3D_HUNYUAN
from opensora.utils.amp import auto_mixed_precision
from opensora.utils.model_utils import str2bool  # _check_cfgs_in_parser

from mindone.utils.logger import set_logger
from mindone.utils.misc import to_abspath
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def prepare_visual_condition_uncausal(
    x: ms.tensor, condition_config: dict, model_ae: nn.Cell, pad: bool = False
) -> ms.tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (ms.tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (nn.Cell): The video encoder module.

    Returns:
        ms.tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = mint.zeros(B, 1, T, H, W).to(x.dtype)  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = mint.zeros(B, C, T, H, W).to(x.dtype)
    x_0 = mint.zeros(B, C, T, H, W).to(x.dtype)
    if T > 1:  # video
        # certain v2v conditions not are applicable for short videos
        if T <= 32 // model_ae.time_compression_ratio:
            condition_config.pop("v2v_head", None)  # given first 32 frames
            condition_config.pop("v2v_tail", None)  # given last 32 frames
            condition_config.pop("v2v_head_easy", None)  # given first 64 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 64 frames
        if T <= 64 // model_ae.time_compression_ratio:
            condition_config.pop("v2v_head_easy", None)  # given first 64 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 64 frames

        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities

        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor
            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                # padded video such that the first latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1  # 32 --> new video: 7 + (1+31-7)
                    padded_x = mint.cat([x[i, :, :1]] * pad_num + [x[i, :, :-pad_num]], dim=1).unsqueeze(0)
                    x_0[i] = model_ae.encoder(padded_x)[0]
                else:
                    x_0[i] = model_ae.encoder(x[i : i + 1])[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encoder(
                    x[i, :, :1, :, :].unsqueeze(0)
                ) 
            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1
                    padded_x = mint.cat(
                        [x[i, :, :1]] * pad_num
                        + [x[i, :, : -pad_num * 2]]
                        + [x[i, :, -pad_num * 2 - 1].unsqueeze(1)] * pad_num,
                        dim=1,
                    ).unsqueeze(
                        0
                    )  # remove the last pad_num * 2 frames from the end of the video
                    x_0[i] = model_ae.encoder(padded_x)[0]
                    # condition: encode the image only
                    latent[i, :, :1, :, :] = model_ae.encoder(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = model_ae.encoder(x[i, :, -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = model_ae.encoder(x[i : i + 1])[0]
                    latent[i, :, :1, :, :] = model_ae.encoder(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = model_ae.encoder(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = model_ae.time_compression_ratio - 1
                    padded_x = mint.cat([x[i, :, pad_num:]] + [x[i, :, -1:]] * pad_num, dim=1).unsqueeze(0)
                    x_0[i] = model_ae.encoder(padded_x)[0]
                    latent[i, :, -1:, :, :] = model_ae.encoder(x[i, :, -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = model_ae.encoder(x[i : i + 1])[0]
                    latent[i, :, -1:, :, :] = model_ae.encoder(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == "v2v_head":  # mask the first 32 video frames
                assert T > 32 // model_ae.time_compression_ratio
                conditioned_t = 32 // model_ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == "v2v_tail":  # mask the last 32 video frames
                assert T > 32 // model_ae.time_compression_ratio
                conditioned_t = 32 // model_ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            elif mask_cond == "v2v_head_easy":  # mask the first 64 video frames
                assert T > 64 // model_ae.time_compression_ratio
                conditioned_t = 64 // model_ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == "v2v_tail_easy":  # mask the last 64 video frames
                assert T > 64 // model_ae.time_compression_ratio
                conditioned_t = 64 // model_ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            # elif mask_cond == "v2v_head":  # mask from the beginning to a random point
            #     masks[i, :, : random.randint(1, T - 2), :, :] = 1
            # elif mask_cond == "v2v_tail":  # mask from a random point to the end
            #     masks[i, :, -random.randint(1, T - 2) :, :, :] = 1
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
    else:  # image
        x_0 = model_ae.encoder(x)  # latent video

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = mint.cat((masks, latent), dim=1)
    return x_0, cond


def prepare_visual_condition_causal(x: ms.tensor, condition_config: dict, model_ae: nn.Cell) -> ms.tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (ms.tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        model_ae (nn.Cell): The video encoder module.

    Returns:
        ms.tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    C = model_ae.cfg.latent_channels
    T, H, W = model_ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = mint.zeros(B, 1, T, H, W).to(x.dtype)  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = mint.zeros(B, C, T, H, W).to(x.dtype)
    x_0 = mint.zeros(B, C, T, H, W).to(x.dtype)
    if T > 1:  # video
        # certain v2v conditions not are applicable for short videos
        if T <= (32 // model_ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head", None)  # given first 33 frames
            condition_config.pop("v2v_tail", None)  # given last 33 frames
            condition_config.pop("v2v_head_easy", None)  # given first 65 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 65 frames
        if T <= (64 // model_ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head_easy", None)  # given first 65 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 65 frames

        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities

        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor

            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                masks[i, :, 0, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encoder(x[i, :, :1, :, :].unsqueeze(0))

            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = model_ae.encoder(x[i, :, :1, :, :].unsqueeze(0))
                latent[i, :, -1:, :, :] = model_ae.encoder(x[i, :, -1:, :, :].unsqueeze(0))

            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                # condition: encode the last image only
                latent[i, :, -1:, :, :] = model_ae.encoder(x[i, :, -1:, :, :].unsqueeze(0))

            elif "v2v_head" in mask_cond:  # mask the first 33 video frames
                ref_t = 33 if not "easy" in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                # encode the first ref_t frame video separately
                latent[i, :, :conditioned_t, :, :] = model_ae.encoder(x[i, :, :ref_t, :, :].unsqueeze(0))

            elif "v2v_tail" in mask_cond:  # mask the last 32 video frames
                ref_t = 33 if not "easy" in mask_cond else 65
                assert (ref_t - 1) % model_ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // model_ae.time_compression_ratio + 1
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
                # encode the first ref_t frame video separately
                latent[i, :, -conditioned_t:, :, :] = model_ae.encoder(x[i, :, -ref_t:, :, :].unsqueeze(0))
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"
                x_0[i] = model_ae.encoder(x[i].unsqueeze(0))[0]
    else:  # image
        x_0 = model_ae.encoder(x)  # latent video

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = mint.cat((masks, latent), dim=1)
    return x_0, cond


def save_output(output_folder: Path, video_name: Path, latent: np.array, conds: Optional[np.array] = None):
    fn = video_name.with_suffix(".npz")
    npz_fp = os.path.join(output_folder, fn)
    if not os.path.exists(os.path.dirname(npz_fp)):
        os.makedirs(os.path.dirname(npz_fp))
    if os.path.exists(npz_fp):
        if args.allow_overwrite:
            logger.info(f"Overwritting {npz_fp}")
    if conds is not None:
        np.savez(
            npz_fp,
            latent=latent.astype(np.float32),
            cond=conds.astype(np.float32),
        )
    else:
        np.savez(
            npz_fp,
            latent=latent.astype(np.float32),
        )


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
    max_device_memory: str = None,
    device_target: str = "Ascend",
    jit_level: str = "O0",
):
    """
    Initialize MindSpore environment.

    Args:
        mode: MindSpore execution mode. Default is 0 (ms.GRAPH_MODE).
        seed: The seed value for reproducibility. Default is 42.
        distributed: Whether to enable distributed training. Default is False.
    Returns:
        A tuple containing the device ID, rank ID and number of devices.
    """
    set_random_seed(seed)

    if max_device_memory is not None:
        ms.set_context(max_device_memory=max_device_memory)

    if distributed:
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"rank_id: {rank_id}, device_num: {device_num}")
        ms.reset_auto_parallel_context()

        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=device_num,
        )

        var_info = ["device_num", "rank_id", "device_num / 8", "rank_id / 8"]
        var_value = [device_num, rank_id, int(device_num / 8), int(rank_id / 8)]
        logger.info(dict(zip(var_info, var_value)))

    else:
        device_num = 1
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
        )

    try:
        if jit_level in ["O0", "O1", "O2"]:
            ms.set_context(jit_config={"jit_level": jit_level})
        else:
            logger.warning(
                f"Unsupport jit_level: {jit_level}. The framework automatically selects the execution method"
            )
    except Exception:
        logger.warning(
            "The current jit_level is not suitable because current MindSpore version or mode does not match,"
            "please ensure the MindSpore version >= ms2.3_0615, and use GRAPH_MODE."
        )

    # if global_bf16:
    #     ms.set_context(ascend_config={"precision_mode": "allow_mix_precision_bf16"})

    return rank_id, device_num


def main(args):
    set_logger(name="", output_dir="logs/infer_vae")
    rank_id, device_num = init_env(
        args.mode, args.seed, args.use_parallel, device_target=args.device_target,
    )
    print(f"rank_id {rank_id}, device_num {device_num}")

    if args.resize_by_max_value and args.batch_size != 1:
        raise ValueError(
            f"Batch size must be 1 when `resize_by_max_value=True`, but get `batch_size={args.batch_size}`."
        )

    # build dataloader for large amount of captions
    ds_config = dict(
        csv_path=args.csv_path,
        video_folder=args.video_folder,
        sample_size=args.image_size,
        sample_stride=args.frame_stride,
        micro_batch_size=args.vae_micro_batch_size,
        video_column=args.video_column,
        caption_column=args.caption_column,
        return_frame_data=args.dl_return_all_frames,
        resize_by_max_value=args.resize_by_max_value,
        transform_name=args.transform_name,  # TODO: check whether align with original repo
        filter_data=args.filter_data,
    )
    dataloader, ds = create_dataloader(
        ds_config,
        args.batch_size,
        ds_name="video",
        num_parallel_workers=16,
        max_rowsize=256,
        shuffle=False,  # be in order
        device_num=device_num,
        rank_id=rank_id,
        drop_remainder=False,
        return_dataset=True,
    )
    dataset_size = dataloader.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")

    # model initiate and weight loading
    logger.info("vae init")
    # with open(args.config, "r") as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)
    # ae_config = config["ae"]

    # temp solution, copied from configs/opensora-v2-0/train/image.py
    # FIXME: load config from yaml
    ae_config = dict(
        # type="hunyuan_vae",
        from_pretrained=args.vae_checkpoint,
        dtype=args.vae_precision,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        latent_channels=16,
        use_spatial_tiling=True,
        use_temporal_tiling=False,
    )
    # FIXME: load config from yaml
    cond_config = dict(
        is_causal_vae=True,
        condition_config = dict(
            t2v=1,
            i2v_head=5,  # train i2v (image as first frame) with weight 5
            i2v_loop=1,  # train image connection with weight 1
            i2v_tail=1,  # train i2v (image as last frame) with weight 1
        ),
    )
    model_ae = CausalVAE3D_HUNYUAN(**ae_config).set_train(False)
    del model_ae.decoder

    if cond_config.get("is_causal_vae", False):
        prepare_visual_condition = prepare_visual_condition_causal
    else:
        prepare_visual_condition = prepare_visual_condition_uncausal

    # TODO
    """
    def pack(x: Tensor, patch_size: int = 2) -> Tensor:
        return rearrange(
            x, "b c t (h ph) (w pw) -> b (t h w) (c ph pw)", ph=patch_size, pw=patch_size
        )
    """
    
    dtype_map = {"fp16": ms.float16, "bf16": ms.bfloat16}
    if args.vae_precision in ["fp16", "bf16"]:
        vae = auto_mixed_precision(
            vae,
            amp_level=args.amp_level,
            dtype=dtype_map[args.vae_precision],
            custom_fp32_cells=[nn.GroupNorm],
        )

    logger.info("Start VAE embedding...")

    # infer
    if args.csv_path is not None:
        if args.output_path in [None, ""]:
            output_folder = os.path.dirname(args.csv_path)
        else:
            output_folder = args.output_path
        os.makedirs(output_folder, exist_ok=True)

        logger.info(f"Output embeddings will be saved: {output_folder}")

        ds_iter = dataloader.create_dict_iterator(1, output_numpy=True)
        for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
            start_time = time.time()
            # caption = data["caption"]

            if args.dl_return_all_frames:
                frame_data = data["frame_data"]
                num_videos = frame_data.shape[0]
                # fps = data["fps"][0]
                # ori_size = data["ori_size"][0]
                assert args.batch_size == 1, "batch size > 1 is not supported due to dynamic frame numbers among videos"
                for i in range(num_videos):
                    abs_video_path = data["video_path"][i]
                    video_path = Path(abs_video_path).relative_to(args.video_folder)
                    fn = video_path.with_suffix(".npz")
                    npz_fp = os.path.join(output_folder, fn)
                    if os.path.exists(npz_fp) and not args.allow_overwrite:
                        logger.info(f"{npz_fp} exists, skip vae encoding")
                        continue

                    latent = []
                    conds = []
                    x = frame_data[i]
                    x = np.expand_dims(np.transpose(x, (1, 0, 2, 3)), axis=0)  # [f, c, h, w] -> [b, c, f, h, w], b must be 1
                    bs = args.vae_micro_batch_size

                    for j in range(0, x.shape[0], bs):
                        x_bs = x[:, :, j : min(j + bs, x.shape[2]), :, :]
                        
                        if cond_config.get("condition_config", None) is not None:
                            # condition for i2v & v2v
                            x_0, cond = ms.ops.stop_gradient(prepare_visual_condition(x_bs, cond_config.condition_config, model_ae))
                            # TODO: pack function
                            # cond = pack(cond, patch_size=ae_config.get("patch_size", 2))  # FIXME: general config, not ae_config
                            conds.append(cond.asnumpy())
                        else:
                            x_0 = ms.ops.stop_gradient(model_ae.encoder(ms.Tensor(x_bs, ms.float32)))
                        latent.append(x_0.asnumpy())

                    latent = np.concatenate(latent, axis=0)
                    conds = np.concatenate(conds, axis=0) if len(conds) > 0 else None
                    save_output(output_folder, video_path, latent, conds)
            else:
                num_videos = data["video_path"].shape[0]
                for i in range(num_videos):
                    abs_video_path = data["video_path"][i]
                    video_path = Path(abs_video_path).relative_to(args.video_folder)
                    fn = video_path.with_suffix(".npz")
                    npz_fp = os.path.join(output_folder, fn)
                    if os.path.exists(npz_fp) and not args.allow_overwrite:
                        logger.info(f"{npz_fp} exists, skip vae encoding")
                        continue

                    latent = []
                    conds = []
                    for x_bs, fps, ori_size in ds.get_video_frames_in_batch(
                        abs_video_path, micro_batch_size=args.vae_micro_batch_size, sample_stride=args.frame_stride
                    ):
                        x_bs = np.expand_dims(np.transpose(x_bs, (1, 0, 2, 3)), axis=0)  # [f, c, h, w] -> [b, c, f, h, w]
                        
                        if cond_config.get("condition_config", None) is not None:
                            # condition for i2v & v2v
                            x_0, cond = ms.ops.stop_gradient(prepare_visual_condition(x_bs, cond_config.condition_config, model_ae))
                            # TODO: pack function
                            # cond = pack(cond, patch_size=ae_config.get("patch_size", 2))  # FIXME: general config, not ae_config
                            conds.append(cond.asnumpy())
                        else:
                            x_0 = ms.ops.stop_gradient(model_ae.encoder(ms.Tensor(x_bs, ms.float32)))
                        latent.append(x_0.asnumpy())

                    latent = np.concatenate(latent, axis=0)
                    conds = np.concatenate(conds, axis=0) if len(conds) > 0 else None
                    save_output(output_folder, video_path, latent, conds)

            end_time = time.time()
            logger.info(f"Time cost: {end_time-start_time:0.3f}s")
        logger.info(f"Done. Embeddings saved in {output_folder}")

    else:
        raise ValueError("Must provide csv file!")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        default="configs/opensora-v2-0/train/stage1_i2v.yaml",
        help="Path to load a config yaml file.",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        type=str,
        help="path to csv annotation file, If None, video_caption.csv is expected to live under `data_path`",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="output dir to save the embeddings, if None, will treat the parent dir of csv_path as output_path.",
    )
    parser.add_argument("--video_column", default="video", type=str, help="name of column for videos saved in csv file")
    parser.add_argument(
        "--caption_column", default="caption", type=str, help="name of column for captions saved in csv file"
    )
    parser.add_argument("--video_folder", default="", type=str, help="root dir for the video data")
    parser.add_argument("--filter_data", default=False, type=str2bool, help="Filter non-existing videos.")
    parser.add_argument("--image_size", nargs="+", default=[256, 256], type=int, help="image size")
    parser.add_argument(
        "--dl_return_all_frames",
        default=True,
        type=str2bool,
        help="dataloder return all frames. If True, read all frames in a video then do vae encoding in micro_batch_size (faster but cost more CPU memory). \
                If False, read a clip of frames in a video in micro_batch_size and do vae encoding iteratively. (slower but memory efficient",
    )
    parser.add_argument(
        "--vae_checkpoint",
        type=str,
        default="hpcai-tech/Open-Sora-v2/hunyuan_vae.safetensors",
        help="VAE checkpoint file path which is used to load vae weight.",
    )
    # parser.add_argument(
    #     "--sd_scale_factor", type=float, default=0.18215, help="VAE scale factor of Stable Diffusion model."
    # )
    parser.add_argument(
        "--vae_precision",
        type=str,
        default="fp32",
        choices=["bf16", "fp16", "fp32"],
        help="Precision mode for the VAE model: fp16, bf16, or fp32.",
    )

    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")
    parser.add_argument(
        "--enable_flash_attention",
        default=False,
        type=str2bool,
        help="whether to enable flash attention. Default is False",
    )
    # parser.add_argument(
    #     "--dtype",
    #     default="fp32",
    #     type=str,
    #     choices=["bf16", "fp16", "fp32"],
    #     help="what data type to use for latte. Default is `fp32`, which corresponds to ms.float16",
    # )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument("--frame_stride", default=1, type=int, help="frame sampling stride")
    parser.add_argument(
        "--transform_name",
        default="center",
        type=str,
        help="center or crop_resize, if center, resize by the short side to h \
                then center crop. If crop_resize, center crop maximally according to \
                the AR of target image size then resize, suitable for where target h != target w.",
    )
    parser.add_argument(
        "--vae_micro_batch_size",
        type=int,
        default=64,
        help="If not None, split batch_size*num_frames into smaller ones for VAE encoding to reduce memory limitation",
    )
    parser.add_argument(
        "--use_recompute",
        default=False,
        type=str2bool,
        help="whether use recompute.",
    )
    parser.add_argument(
        "--allow_overwrite",
        type=str2bool,
        default=False,
        help="If True, allow to overwrite the existing npz file. If False, will skip vae encoding if the latent npz file is already existed",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="batch size")
    parser.add_argument("--resize_by_max_value", default=False, type=str2bool, help="resize the image by max instead.")

    __dir__ = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.abspath(os.path.join(__dir__, ".."))
    args = parser.parse_args()
    # convert to absolute path, necessary for modelarts
    args.csv_path = to_abspath(abs_path, args.csv_path)
    args.output_path = to_abspath(abs_path, args.output_path)
    args.video_folder = to_abspath(abs_path, args.video_folder)
    args.vae_checkpoint = to_abspath(abs_path, args.vae_checkpoint)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
