"""
AnimateDiff dataset embedding cache
"""
import csv
import datetime
import logging
import os
import sys
from typing import Tuple

import numpy as np
from decord import VideoReader
from omegaconf import OmegaConf

import mindspore as ms
import mindspore.ops as ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore.mindrecord import FileWriter

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
animatediff_path = os.path.abspath(os.path.join(__dir__, "../"))
sys.path.insert(0, mindone_lib_path)
sys.path.insert(0, animatediff_path)

from ad.data.dataset import create_video_transforms, read_gif

# from ad.data.dataset import check_sanity
from ad.utils.load_models import update_unet2d_params_for_unet3d
from args_train import parse_args

from mindone.utils.config import get_obj_from_str
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed
from mindone.utils.version_control import is_old_ms_version

logger = logging.getLogger(__name__)


def build_model_from_config(config, unet_config_update=None):
    config = OmegaConf.load(config).model
    if unet_config_update is not None:
        # config["params"]["unet_config"]["params"]["enable_flash_attention"] = enable_flash_attention
        unet_args = config["params"]["unet_config"]["params"]
        for name, value in unet_config_update.items():
            if value is not None:
                logger.info("Arg `{}` updated: {} -> {}".format(name, unet_args[name], value))
                unet_args[name] = value
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())
    # config_params['cond_stage_trainable'] = cond_stage_trainable # TODO: easy config
    return get_obj_from_str(config["target"])(**config_params)


def load_pretrained_model(
    pretrained_ckpt, net, unet_initialize_random=False, load_unet3d_from_2d=False, unet3d_type="adv2"
):
    logger.info(f"Loading pretrained model from {pretrained_ckpt}")
    if os.path.exists(pretrained_ckpt):
        param_dict = load_checkpoint(pretrained_ckpt)

        if load_unet3d_from_2d:
            param_dict = update_unet2d_params_for_unet3d(param_dict, unet3d_type=unet3d_type)

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
        logger.info(
            "Net params not load: {}, Total net params not loaded: {}".format(param_not_load, len(param_not_load))
        )
        logger.info(
            "Ckpt params not load: {}, Total ckpt params not loaded: {}".format(ckpt_not_load, len(ckpt_not_load))
        )

        if not unet_initialize_random:
            assert (
                len(ckpt_not_load) == 0
            ), "All params in ckpt should be loaded to the network. See log for detailed missing params."
    else:
        logger.warning(f"Checkpoint file {pretrained_ckpt} dose not exist!!!")


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    max_device_memory: str = None,
    device_target: str = "Ascend",
) -> Tuple[int, int, int]:
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

    device_num = 1
    device_id = int(os.getenv("DEVICE_ID", 0))
    rank_id = 0
    ms.set_context(
        mode=mode,
        device_target=device_target,
        device_id=device_id,
        # ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # TODO: tune
    )

    return device_id, rank_id, device_num


def main(args):
    time_str = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    args.output_path = os.path.join(args.output_path, time_str)

    # 1. init
    device_id, rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        device_target=args.device_target,
        max_device_memory=args.max_device_memory,
    )
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. build model
    unet_config_update = dict(enable_flash_attention=args.enable_flash_attention, use_recompute=args.use_recompute)
    latent_diffusion_with_loss = build_model_from_config(args.model_config, unet_config_update)
    # 1) load sd pretrained weight
    load_pretrained_model(
        args.pretrained_model_path,
        latent_diffusion_with_loss,
        unet_initialize_random=args.unet_initialize_random,
        load_unet3d_from_2d=(not args.image_finetune),
        unet3d_type="adv1" if "mmv1" in args.model_config else "adv2",  # TODO: better not use filename to judge version
    )

    tokenizer = latent_diffusion_with_loss.cond_stage_model.tokenize
    video_column = args.video_column
    caption_column = args.caption_column
    csv_path = os.path.join(args.data_path, "video_caption.csv")
    video_folder = args.data_path
    train_data_type = args.train_data_type

    if args.save_data_type == "float32":
        save_data_type = np.float32
    elif args.save_data_type == "float16":
        save_data_type = np.float16
    else:
        raise ValueError("Save data type {} is not supported!".format(args.save_data_type))

    if train_data_type == "mindrecord":
        assert args.save_data_type == "float32"

    cache_folder = args.cache_folder
    new_csv_path = os.path.join(cache_folder, "video_caption.csv")

    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Data path: {args.data_path}",
            f"train_data_type: {args.train_data_type}",
            f"save_data_type: {args.save_data_type}",
            f"cache_folder: {args.cache_folder}",
            f"Precision: {latent_diffusion_with_loss.model.diffusion_model.dtype}",
            f"Image size: {args.image_size}",
            f"Frames: {args.num_frames}",
            f"Enable flash attention: {args.enable_flash_attention}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    with open(csv_path, "r") as csvfile:
        dataset = list(csv.DictReader(csvfile))

    new_dataset = dataset.copy()

    length = len(dataset)

    if train_data_type == "mindrecord":
        schema = {
            "video": {"type": "string"},
            "caption": {"type": "string"},
            "video_latent": {"type": args.save_data_type, "shape": [-1, 4, args.image_size // 8, args.image_size // 8]},
            "text_emb": {"type": args.save_data_type, "shape": [77, 768]},
        }
        count = 0
        dataset_name = args.data_path.split("/")[-1]
        filename = dataset_name + str(count) + ".mindrecord"
        file_path = os.path.join(cache_folder, filename)
        writer = FileWriter(file_path, shard_num=1, overwrite=True)
        writer.set_page_size(1 << 26)
        writer.add_schema(schema, "Preprocessed {} dataset.".format(dataset_name))

    logger.info("Start dataset embedding cache...")
    data = []
    for idx in range(length):
        video_dict = dataset[idx]
        video_fn, caption = video_dict[video_column], video_dict[caption_column]
        video_path = os.path.join(video_folder, video_fn)
        if video_path.endswith(".gif"):
            video_reader = read_gif(video_path, mode="RGB")
        else:
            video_reader = VideoReader(video_path)

        video_length = len(video_reader)
        batch_index = np.arange(video_length)

        if video_path.endswith(".gif"):
            pixel_values = video_reader[batch_index]
        else:
            pixel_values = video_reader.get_batch(batch_index).asnumpy()

        inputs = {"image": pixel_values[0]}
        num_frames = len(pixel_values)
        for i in range(num_frames - 1):
            inputs[f"image{i}"] = pixel_values[i + 1]

        pixel_transforms = create_video_transforms(
            args.image_size, args.image_size, num_frames, interpolation="bicubic", backend="al"
        )
        output = pixel_transforms(**inputs)
        pixel_values = np.stack(list(output.values()), axis=0)
        # (f h w c) -> (f c h w)
        pixel_values = np.transpose(pixel_values, (0, 3, 1, 2))
        pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)
        tokens = tokenizer(caption)

        if isinstance(tokens, list):
            tokens = np.array(tokens, dtype=np.int64)
        if len(tokens.shape) == 2:
            tokens = tokens[0]

        text_data = tokens
        pixel_values = ms.Tensor(pixel_values)
        text_data = ms.Tensor(text_data)

        frame_list = []
        for frame in pixel_values:
            frame_data = ops.unsqueeze(frame, dim=0)
            frame_latent = latent_diffusion_with_loss.first_stage_model.encode(frame_data)
            frame_list.append(frame_latent)
        video_latent = ops.concat(frame_list)

        text_data = ops.unsqueeze(text_data, dim=0)
        text_emb = latent_diffusion_with_loss.cond_stage_model(text_data)

        if train_data_type == "npz":
            video_path = os.path.join(cache_folder, video_fn.split(".")[0] + ".npz")
            np.savez_compressed(
                video_path,
                video_latent=video_latent.asnumpy().copy().astype(save_data_type),
                text_emb=text_emb.asnumpy().copy()[0].astype(save_data_type),
            )
        elif train_data_type == "mindrecord":
            if os.path.isfile(file_path):
                mindrecord_size = os.stat(file_path).st_size
                mindrecord_size = mindrecord_size / 1024 / 1024 / 1024
                if mindrecord_size > 19:
                    if data:
                        writer.write_raw_data(data)
                        data = []
                    writer.commit()
                    # close last filewriter when it exceeds 19GB
                    count += 1
                    filename = dataset_name + str(count) + ".mindrecord"
                    file_path = os.path.join(cache_folder, filename)
                    writer = FileWriter(file_path, shard_num=1, overwrite=True)
                    writer.set_page_size(1 << 26)
                    writer.add_schema(schema, "Preprocessed {} dataset.".format(dataset_name))

            sample = {
                "video": video_dict[video_column],
                "caption": video_dict[caption_column],
                "video_latent": video_latent.asnumpy().copy().astype(save_data_type),
                "text_emb": text_emb.asnumpy().copy()[0].astype(save_data_type),
            }
            data.append(sample)
            if idx % 10 == 0:
                writer.write_raw_data(data)
                data = []

        else:
            raise ValueError("Train data type {} is not supported!".format(train_data_type))

        new_dataset[idx]["embedding_path"] = video_fn.split(".")[0] + "." + train_data_type

    if data and train_data_type == "mindrecord":
        writer.write_raw_data(data)
        writer.commit()

    with open(new_csv_path, mode="w", newline="") as f:
        field_name = ["video", "caption", "embedding_path"]
        write = csv.DictWriter(f, fieldnames=field_name)
        write.writeheader()
        write.writerows(new_dataset)

    logger.info("Dataset embedding cache successfully saved in {}".format(cache_folder))


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
