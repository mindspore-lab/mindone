import os
import sys
import time

from tqdm import tqdm

mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append("./")
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.dataset import getdataset
from opensora.dataset.loader import create_dataloader
from opensora.models.causalvideovae import ae_stride_config
from opensora.models.diffusion import Diffusion_models
from opensora.npu_config import npu_config
from opensora.train.commons import parse_args
from opensora.utils.dataset_utils import Collate, LengthGroupedSampler
from opensora.utils.message_utils import print_banner

from mindone.utils.config import str2bool


def load_dataset_and_dataloader(args, device_num=1, rank_id=0):
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]

    assert (
        ae_stride_h == ae_stride_w
    ), f"Support only ae_stride_h == ae_stride_w now, but found ae_stride_h ({ae_stride_h}), ae_stride_w ({ae_stride_w})"
    args.ae_stride_t, args.ae_stride_h, args.ae_stride_w = ae_stride_t, ae_stride_h, ae_stride_w
    args.ae_stride = args.ae_stride_h
    patch_size = args.model[-3:]
    patch_size_t, patch_size_h, patch_size_w = int(patch_size[0]), int(patch_size[1]), int(patch_size[2])
    args.patch_size = patch_size_h
    args.patch_size_t, args.patch_size_h, args.patch_size_w = patch_size_t, patch_size_h, patch_size_w
    assert (
        patch_size_h == patch_size_w
    ), f"Support only patch_size_h == patch_size_w now, but found patch_size_h ({patch_size_h}), patch_size_w ({patch_size_w})"
    assert (
        args.max_height % ae_stride_h == 0
    ), f"Height must be divisible by ae_stride_h, but found Height ({args.max_height}), ae_stride_h ({ae_stride_h})."
    assert (
        args.max_width % ae_stride_h == 0
    ), f"Width size must be divisible by ae_stride_h, but found Width ({args.max_width}), ae_stride_h ({ae_stride_h})."

    args.stride_t = ae_stride_t * patch_size_t
    args.stride = ae_stride_h * patch_size_h

    assert args.dataset == "t2v", "Support t2v dataset only."
    print_banner("Dataset Loading")
    # Setup data:
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
        column_names=["pixel_values", "attention_mask", "text_embed", "encoder_attention_mask"],
        drop_last=True,
    )
    dataset_size = dataloader.get_dataset_size()
    assert dataset_size > 0, "Incorrect dataset size. Please check your dataset size and your global batch size"
    return train_dataset, dataloader


def parse_t2v_train_args(parser):
    # TODO: NEW in v1.3 , but may not use
    # dataset & dataloader
    parser.add_argument("--max_hxw", type=int, default=None)
    parser.add_argument("--min_hxw", type=int, default=None)
    parser.add_argument("--ood_img_ratio", type=float, default=0.0)
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

    # text encoder & vae & diffusion model
    parser.add_argument("--vae_fp32", action="store_true")
    parser.add_argument("--extra_save_mem", action="store_true")
    parser.add_argument("--text_encoder_name_1", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--text_encoder_name_2", type=str, default=None)
    parser.add_argument("--sparse1d", action="store_true")
    parser.add_argument("--sparse_n", type=int, default=2)
    parser.add_argument("--skip_connection", action="store_true")
    parser.add_argument("--cogvideox_scheduler", action="store_true")
    parser.add_argument("--v1_5_scheduler", action="store_true")
    parser.add_argument("--rf_scheduler", action="store_true")
    parser.add_argument(
        "--weighting_scheme", type=str, default="logit_normal", choices=["sigma_sqrt", "logit_normal", "mode", "cosmap"]
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )

    # diffusion setting
    parser.add_argument("--offload_ema", action="store_true", help="Offload EMA model to CPU during training step.")
    parser.add_argument("--foreach_ema", action="store_true", help="Use faster foreach implementation of EMAModel.")
    parser.add_argument("--rescale_betas_zero_snr", action="store_true")

    # validation & logs
    parser.add_argument("--enable_profiling", action="store_true")
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.5)

    parser.add_argument("--output_dir", default="outputs/", help="The directory where training results are saved.")
    parser.add_argument("--dataset", type=str, default="t2v")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="The training dataset text file specifying the path of video folder, text embedding cache folder, and the annotation json file",
    )
    parser.add_argument(
        "--val_data",
        type=str,
        default=None,
        help="The validation dataset text file, same format as the training dataset text file.",
    )
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument(
        "--filter_nonexistent",
        type=str2bool,
        default=True,
        help="Whether to filter out non-existent samples in image datasets and video datasets." "Defaults to True.",
    )
    parser.add_argument(
        "--text_embed_cache",
        type=str2bool,
        default=True,
        help="Whether to use T5 embedding cache. Must be provided in image/video_data.",
    )
    parser.add_argument("--vae_latent_folder", default=None, type=str, help="root dir for the vae latent data")
    parser.add_argument("--model", type=str, choices=list(Diffusion_models.keys()), default="OpenSoraT2V_v1_3-2B/122")
    parser.add_argument("--interpolation_scale_h", type=float, default=1.0)
    parser.add_argument("--interpolation_scale_w", type=float, default=1.0)
    parser.add_argument("--interpolation_scale_t", type=float, default=1.0)
    parser.add_argument("--downsampler", type=str, default=None)
    parser.add_argument("--ae", type=str, default="WFVAEModel_D8_4x8x8")
    parser.add_argument("--ae_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.3.0")
    parser.add_argument("--sample_rate", type=int, default=1)
    parser.add_argument("--train_fps", type=int, default=24)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--num_frames", type=int, default=17)
    parser.add_argument("--max_height", type=int, default=320)
    parser.add_argument("--max_width", type=int, default=240)
    parser.add_argument("--group_frame", action="store_true")
    parser.add_argument("--group_resolution", action="store_true")
    parser.add_argument("--use_rope", action="store_true")
    parser.add_argument("--pretrained", type=str, default=None)

    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)
    parser.add_argument("--enable_tiling", action="store_true")

    parser.add_argument("--attention_mode", type=str, choices=["xformers", "math", "flash"], default="xformers")
    # parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--multi_scale", action="store_true")

    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--use_img_from_vid", action="store_true")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument("--cfg", type=float, default=0.1)
    parser.add_argument(
        "--num_no_recompute",
        type=int,
        default=0,
        help="If use_recompute is True, `num_no_recompute` blocks will be removed from the recomputation list."
        "This is a positive integer which can be tuned based on the memory usage.",
    )
    parser.add_argument("--dataloader_prefetch_size", type=int, default=None, help="minddata prefetch size setting")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")
    parser.add_argument("--train_sp_batch_size", type=int, default=1, help="Batch size for sequence parallel training")
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference, better to set to True when training vae",
    )
    parser.add_argument(
        "--vae_precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for vae. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--text_encoder_precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for T5 text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
    )
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
    parser.add_argument(
        "--prediction_type",
        type=str,
        default=None,
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. \
            If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument("--ema_start_step", type=int, default=0)
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )

    return parser


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


if __name__ == "__main__":
    args = parse_args(additional_parse_args=parse_t2v_train_args)
    if args.resume_from_checkpoint == "True":
        args.resume_from_checkpoint = True
    save_src_strategy = args.use_parallel and args.parallel_mode != "data"
    if args.num_frames == 1 or args.use_image_num != 0:
        args.sp_size = 1
    rank_id, device_num = npu_config.set_npu_env(args, strategy_ckpt_save_file=save_src_strategy)
    dataset, dataloader = load_dataset_and_dataloader(args, device_num=device_num, rank_id=rank_id)

    test_dataset(dataset)
    test_dataloder(dataloader)
