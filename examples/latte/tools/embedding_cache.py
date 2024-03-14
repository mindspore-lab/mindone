"""
Dataset Embedding Cache script
"""
import datetime
import logging
import os
import sys
from typing import Tuple

import numpy as np
from mindrecord_writer import MindRecordEmbeddingCacheWriter
from npz_writer import NPZEmbeddingCacheWriter
from tqdm import tqdm

import mindspore as ms
from mindspore.communication.management import get_group_size, get_rank, init

__dir__ = os.path.dirname(os.path.abspath(__file__))
latte_path = os.path.abspath(os.path.join(__dir__, "../"))
sys.path.insert(0, latte_path)

from args_train import parse_args
from data.dataset import get_dataset
from diffusion import create_diffusion
from modules.autoencoder import SD_CONFIG, AutoencoderKL
from modules.text_encoders import initiate_clip_text_encoder
from omegaconf import OmegaConf
from pipelines import get_model_with_loss
from utils.model_utils import remove_pname_prefix

mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)

# load training modules
from mindone.models.latte import Latte_models
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.logger import set_logger
from mindone.utils.seed import set_random_seed

os.environ["HCCL_CONNECT_TIMEOUT"] = "6000"

logger = logging.getLogger(__name__)


def init_env(
    mode: int = ms.GRAPH_MODE,
    seed: int = 42,
    distributed: bool = False,
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

    if distributed:
        device_id = int(os.getenv("DEVICE_ID"))
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            # ascend_config={"precision_mode": "allow_fp32_to_fp16"}, # TODO: tune
        )
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        logger.debug(f"Device_id: {device_id}, rank_id: {rank_id}, device_num: {device_num}")
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
        device_id = int(os.getenv("DEVICE_ID", 0))
        rank_id = 0
        ms.set_context(
            mode=mode,
            device_target=device_target,
            device_id=device_id,
            # ascend_config={"precision_mode": "allow_fp32_to_fp16"},  # TODO: tune
        )

    return device_id, rank_id, device_num


def init_models(args):
    # 2.1 latte
    logger.info(f"{args.model_name}-{args.image_size}x{args.image_size} init")
    latent_size = args.image_size // 8
    latte_model = Latte_models[args.model_name](
        input_size=latent_size,
        num_classes=args.num_classes,
        block_kwargs={"enable_flash_attention": args.enable_flash_attention},
        condition=args.condition,
        num_frames=args.num_frames,
    )

    if args.use_fp16:
        latte_model = auto_mixed_precision(latte_model, amp_level="O2")

    if len(args.pretrained_model_path) > 0:
        param_dict = ms.load_checkpoint(args.pretrained_model_path)
        logger.info(f"Loading ckpt {args.pretrained_model_path} into Latte...")
        # in case a save ckpt with "network." prefix, removing it before loading
        param_dict = remove_pname_prefix(param_dict, prefix="network.")
        latte_model.load_params_from_ckpt(param_dict)
    else:
        logger.info("Use random initialization for Latte")

    # set train=False
    latte_model.set_train(False)
    for param in latte_model.get_parameters():
        param.requires_grad = False

    # 2.2 vae
    logger.info("vae init")
    vae = AutoencoderKL(
        SD_CONFIG,
        4,
        ckpt_path=args.vae_checkpoint,
        use_fp16=False,  # disable amp for vae
    )
    vae = vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    if args.condition == "text":
        text_encoder = initiate_clip_text_encoder(
            use_fp16=args.use_fp16,
            ckpt_path=args.clip_checkpoint,
            trainable=False,
        )
        tokenizer = text_encoder.tokenizer
        text_embed_shape = [77, 768]
    else:
        text_encoder, tokenizer, text_embed_shape = None, None, None
    diffusion = create_diffusion(timestep_respacing="")
    latent_diffusion_with_loss = get_model_with_loss(args.condition)(
        latte_model,
        diffusion,
        vae,
        args.sd_scale_factor,
        args.condition,
        text_encoder=text_encoder,
        cond_stage_trainable=False,
    )
    return latent_diffusion_with_loss, tokenizer, text_embed_shape


def init_data(args, tokenizer, device_num, rank_id):
    data_config = OmegaConf.load(args.data_config_file).data_config
    # set some data params from argument parser
    data_config.sample_size = args.image_size
    data_config.sample_n_frames = args.num_frames
    data_config.batch_size = args.train_batch_size
    data_config.shuffle = False

    dataset, _ = get_dataset(
        args.dataset_name, data_config, tokenizer=tokenizer, device_num=device_num, rank_id=rank_id, return_dataset=True
    )
    return dataset


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
    set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 2. model initiate and weight loading
    latent_diffusion_with_loss, tokenizer, text_embed_shape = init_models(args)
    # select dataset
    dataset = init_data(args, tokenizer, device_num, rank_id)

    length = len(dataset)
    # parse train and save data types
    cache_file_type = args.cache_file_type
    assert cache_file_type in [
        "mindrecord",
        "npz",
    ], "embedding_cache.py should work on cache file type in['mindrecord', 'npz'] "
    if args.save_data_type == "float32":
        save_data_type = np.float32
    elif args.save_data_type == "float16":
        save_data_type = np.float16
    else:
        raise ValueError("Save data type {} is not supported!".format(args.save_data_type))

    # initiate cache folder and new dataset
    cache_folder = args.cache_folder
    assert len(cache_folder) > 0, "cache folder must be provided!"
    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    # default save column names
    video_column = "video"
    caption_column = "caption"
    class_column = "class"
    start_video_index = 0 if args.resume_cache_index is None else args.resume_cache_index
    logger.info(f"Start embedding cache from {start_video_index}th video...")

    if cache_file_type == "mindrecord":
        assert args.save_data_type == "float32"
        schema = {
            video_column: {"type": "string"},
            "video_latent": {"type": args.save_data_type, "shape": [-1, 4, args.image_size // 8, args.image_size // 8]},
        }
        if args.condition == "class":
            schema[class_column] = {"type": "int"}
        elif args.condition == "text":
            schema[caption_column] = {"type": "string"}
            schema["text_emb"] = {"type": args.save_data_type, "shape": text_embed_shape}

        dataset_name = args.data_config_file.split("/")[-1].split(".")[0]
        embed_cache_writer = MindRecordEmbeddingCacheWriter(
            cache_folder,
            dataset_name,
            schema,
            start_lines_index=start_video_index,
            overwrite=False if args.resume_cache_index else True,
            max_page_size=args.max_page_size,
            dump_every_n_lines=min((length - start_video_index), 10),
        )
    else:
        embed_cache_writer = NPZEmbeddingCacheWriter(cache_folder, start_video_index)

    # print key info
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"cache_file_type: {args.cache_file_type}",
            f"save_data_type: {args.save_data_type}",
            f"cache_folder: {args.cache_folder}",
            f"Use FP16: {args.use_fp16}",
            f"Image size: {args.image_size}",
            f"Frames: {args.num_frames}",
            f"Enable flash attention: {args.enable_flash_attention}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)

    logger.info("Start dataset embedding cache...")
    data = []

    for video_index in tqdm(range(start_video_index, length), total=length):
        video_save_kwargs = {}
        video_latent = []
        for video_name, inputs in dataset.traverse_single_video_frames(video_index):
            clip_data = inputs["video"]
            clip_latent = latent_diffusion_with_loss.get_latents(ms.Tensor(clip_data, ms.float32)).asnumpy()
            video_latent.append(clip_latent)
        video_latent = np.concatenate(video_latent, axis=0)
        # save video_latent (tensor) and video_name (string)
        video_save_kwargs["video_latent"] = video_latent.copy().astype(save_data_type)
        video_save_kwargs[video_column] = video_name

        if args.condition == "text":
            # save text_emb (tensor) and caption (string)
            assert len(inputs) > 1, "incorrect data return shape"
            token_ids = inputs["text"]
            text_emb = latent_diffusion_with_loss.get_condition_embeddings(ms.Tensor(token_ids, ms.int32)).asnumpy()
            video_save_kwargs["text_emb"] = text_emb.copy().astype(save_data_type)

            caption = inputs["caption"]
            video_save_kwargs["caption"] = caption
        elif args.condition == "class":
            # save class_labels (tensor)
            assert len(inputs) > 1, "incorrect data return shape"
            class_labels = inputs["class"]
            video_save_kwargs[class_column] = class_labels.copy().astype(np.int32)

        if cache_file_type == "npz":
            # save video_name.npz
            embed_cache_writer.save(video_name, video_save_kwargs)
        elif cache_file_type == "mindrecord":
            data.append(video_save_kwargs)
            data = embed_cache_writer.save(data)
        else:
            raise ValueError("Train data type {} is not supported!".format(cache_file_type))

    if cache_file_type == "mindrecord":
        embed_cache_writer.save_data_and_close_writer(data)
    logger.info("Dataset embedding cache successfully saved in {}".format(cache_folder))
    embed_cache_writer.get_status()


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args()
    main(args)
