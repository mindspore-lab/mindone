import logging
import os
import sys
import time

from tqdm import tqdm

import mindspore as ms

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append("./")
from hyvideo.dataset import getdataset
from hyvideo.dataset.loader import create_dataloader
from hyvideo.train.commons import parse_args
from hyvideo.utils.dataset_utils import Collate, LengthGroupedSampler
from hyvideo.utils.message_utils import print_banner
from hyvideo.utils.ms_utils import init_env

from examples.hunyuanvideo.hyvideo.utils.parallel_states import get_sequence_parallel_state, hccl_info
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def set_all_reduce_fusion(
    params,
    split_num: int = 7,
    distributed: bool = False,
    parallel_mode: str = "data",
) -> None:
    """Set allreduce fusion strategy by split_num."""

    if distributed and parallel_mode == "data":
        all_params_num = len(params)
        step = all_params_num // split_num
        split_list = [i * step for i in range(1, split_num)]
        split_list.append(all_params_num - 1)
        logger.info(f"Distribute config set: dall_params_num: {all_params_num}, set all_reduce_fusion: {split_list}")
        ms.set_auto_parallel_context(all_reduce_fusion_config=split_list)


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    # 1. init
    if args.num_frames == 1 or args.use_image_num != 0:
        args.sp_size = 1
    save_src_strategy = args.use_parallel and args.parallel_mode != "data"
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        mempool_block_size=args.mempool_block_size,
        global_bf16=args.global_bf16,
        strategy_ckpt_save_file=os.path.join(args.output_dir, "src_strategy.ckpt") if save_src_strategy else "",
        optimizer_weight_shard_size=args.optimizer_weight_shard_size,
        sp_size=args.sp_size if args.num_frames != 1 and args.use_image_num == 0 else 1,
        jit_level=args.jit_level,
        enable_parallel_fusion=args.enable_parallel_fusion,
        jit_syntax_level=args.jit_syntax_level,
        comm_fusion=args.comm_fusion,
    )
    set_logger(name="", output_dir=args.output_dir, rank=rank_id, log_level=eval(args.log_level))

    # 2. Init and load models
    # define vae
    ae_stride_t, ae_stride_h, ae_stride_w = 4, 8, 8

    assert (
        ae_stride_h == ae_stride_w
    ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h

    patch_size_t, patch_size_h, patch_size_w = 1, 2, 2
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert (
        patch_size_h == patch_size_w
    ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    assert (
        args.max_height % ae_stride_h == 0
    ), f"Height must be divisible by ae_stride_h, but found Height ({args.max_height}), ae_stride_h ({ae_stride_h})."
    assert (
        args.num_frames - 1
    ) % ae_stride_t == 0, f"(Frames - 1) must be divisible by ae_stride_t, but found num_frames ({args.num_frames}), ae_stride_t ({ae_stride_t})."
    assert (
        args.max_width % ae_stride_h == 0
    ), f"Width size must be divisible by ae_stride_h, but found Width ({args.max_width}), ae_stride_h ({ae_stride_h})."

    args.stride_t = ae_stride_t * patch_size_t
    args.stride = ae_stride_h * patch_size_h

    args.latent_size_t = (args.num_frames - 1) // ae_stride_t + 1

    # 3. create dataset
    # TODO: replace it with new dataset
    assert args.dataset == "t2v", "Support t2v dataset only."
    print_banner("Training dataset Loading...")

    # Setup data:
    # TODO: to use in v1.3
    if args.trained_data_global_step is not None:
        initial_global_step_for_sampler = args.trained_data_global_step
    else:
        initial_global_step_for_sampler = 0
    total_batch_size = args.train_batch_size * device_num * args.gradient_accumulation_steps
    total_batch_size = total_batch_size // args.sp_size * args.train_sp_batch_size
    args.total_batch_size = total_batch_size
    if args.max_hxw is not None and args.min_hxw is None:
        args.min_hxw = args.max_hxw // 4

    train_dataset = getdataset(args, dataset_file=args.data)
    sampler = LengthGroupedSampler(
        args.train_batch_size,
        world_size=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        gradient_accumulation_size=args.gradient_accumulation_steps,
        initial_global_step=initial_global_step_for_sampler,
        lengths=train_dataset.lengths,
        group_data=args.group_data,
    )
    collate_fn = Collate(args.train_batch_size, args)
    dataloader = create_dataloader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=sampler is None,
        device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
        num_parallel_workers=args.dataloader_num_workers,
        max_rowsize=args.max_rowsize,
        prefetch_size=args.dataloader_prefetch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        column_names=[
            "pixel_values",
            "attention_mask",
            "text_embed",
            "encoder_attention_mask",
            "text_embed_2",
            "encoder_attention_mask_2",
        ],
    )
    dataloader_size = dataloader.get_dataset_size()
    assert (
        dataloader_size > 0
    ), "Incorrect training dataset size. Please check your dataset size and your global batch size"
    return train_dataset, dataloader


def test_dataset(ds):
    num_samples = len(ds)
    steps = min(20, num_samples)
    start = time.time()
    tot = 0
    for i in tqdm(range(steps)):
        batch = ds.__getitem__(i % num_samples)

        dur = time.time() - start
        tot += dur

        if i < 3:
            video = batch["pixel_values"]
            print("D--: ", video.shape, video.dtype, video.min(), video.max())
            print(f"{i+1}/{steps}, time cost: {dur * 1000} ms")

        start = time.time()

    mean = tot / steps
    print("Avg sample loading time: ", mean)


def test_dataloder(dl):
    num_batches = dl.get_dataset_size()

    steps = num_batches * 2
    iterator = dl.create_dict_iterator(2, output_numpy=True)
    tot = 0

    progress_bar = tqdm(range(steps))
    progress_bar.set_description("Steps")

    start = time.time()
    for epoch in range(steps // num_batches):
        for i, batch in enumerate(iterator):
            dur = time.time() - start
            tot += dur

            if epoch * num_batches + i < 3:
                for k in batch:
                    print(k, batch[k].shape, batch[k].dtype)  # , batch[k].min(), batch[k].max())
                print(f"time cost: {dur * 1000} ms")

            progress_bar.update(1)
            if i + 1 > steps:  # in case the data size is too large
                break
            start = time.time()

    mean = tot / steps
    print("Avg batch loading time: ", mean)


def parse_t2v_train_args(parser):
    # TODO: NEW in v1.3 , but may not use
    # dataset & dataloader
    parser.add_argument("--max_hxw", type=int, default=None)
    parser.add_argument("--min_hxw", type=int, default=None)
    parser.add_argument("--group_data", action="store_true")
    parser.add_argument("--hw_stride", type=int, default=32)
    parser.add_argument("--force_resolution", action="store_true")
    parser.add_argument("--trained_data_global_step", type=int, default=None)
    parser.add_argument(
        "--use_decord",
        type=str2bool,
        default=True,
        help="whether to use decord to load videos. If not, use opencv to load videos.",
    )
    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, default="t2v")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The training dataset text file specifying the path of video folder, text embedding cache folder, and the annotation json file",
    )
    parser.add_argument(
        "--filter_nonexistent",
        type=str2bool,
        default=True,
        help="Whether to filter out non-existent samples in image datasets and video datasets." "Defaults to True.",
    )

    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="./ckpts")
    parser.add_argument("--model_max_length_1", type=int, default=315)  # llava llama text encoder
    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument(
        "--model_max_length_2", type=int, default=77
    )  # for text encoder 2 tokenizer, but CLIP text encoder returns pooled hidden states
    parser.add_argument(
        "--text_embed_cache",
        type=str2bool,
        default=True,
        help="Whether to use T5 embedding cache. Must be provided in image/video_data.",
    )
    parser.add_argument("--multi_scale", action="store_true")

    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")

    parser.add_argument("--cfg", type=float, default=0.1)

    parser.add_argument("--dataloader_prefetch_size", type=int, default=None, help="minddata prefetch size setting")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel training")

    parser.add_argument(
        "--enable_parallel_fusion", default=True, type=str2bool, help="Whether to parallel fusion for AdamW"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument("--noise_offset", type=float, default=0.02, help="The scale of noise offset.")
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. More details here: \
            https://arxiv.org/abs/2303.09556.",
    )

    return parser


if __name__ == "__main__":
    logger.debug("process id:", os.getpid())
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    if args.resume_from_checkpoint == "True":
        args.resume_from_checkpoint = True
    dataset, dataloader = main(args)

    test_dataset(dataset)
    test_dataloder(dataloader)
