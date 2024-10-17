import argparse
import glob
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

import mindspore as ms
from mindspore import nn

# TODO: remove in future when mindone is ready for install
mindone_lib_path = os.path.abspath("../../")
sys.path.insert(0, mindone_lib_path)
sys.path.append(os.path.abspath("./"))
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.dataset.text_dataset import create_dataloader
from opensora.models.causalvideovae import CausalVAEModelWrapper, ae_stride_config
from opensora.models.causalvideovae.model.modules.updownsample import TrilinearInterpolate
from opensora.models.diffusion.opensora.modeling_opensora import LayerNorm, OpenSoraT2V
from opensora.models.diffusion.opensora.modules import Attention
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.utils.message_utils import print_banner
from opensora.utils.ms_utils import init_env
from opensora.utils.utils import _check_cfgs_in_parser, get_precision
from transformers import AutoTokenizer

from mindone.diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from mindone.diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from mindone.transformers import MT5EncoderModel

# from mindone.transformers.activations import NewGELUActivation
# from mindone.transformers.models.mt5.modeling_mt5 import MT5LayerNorm
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.logger import set_logger
from mindone.utils.params import count_params
from mindone.visualize.videos import save_videos

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument("--model_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.2.0")
    parser.add_argument(
        "--ms_checkpoint",
        type=str,
        default=None,
        help="If not provided, will search for ckpt file under `model_path`"
        "If provided, will use this pretrained ckpt path.",
    )
    parser.add_argument("--num_frames", type=int, default=1)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--ae", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--ae_path", type=str, default="CausalVAEModel_4x8x8")
    parser.add_argument("--sp_size", type=int, default=1, help="For sequence parallel")

    parser.add_argument("--text_encoder_name", type=str, default="DeepFloyd/t5-v1_1-xxl")
    parser.add_argument("--save_img_path", type=str, default="./sample_videos/t2v")

    parser.add_argument("--guidance_scale", type=float, default=7.5, help="the scale for classifier-free guidance")
    parser.add_argument("--max_sequence_length", type=int, default=300, help="the maximum text tokens length")

    parser.add_argument("--sample_method", type=str, default="PNDM")
    parser.add_argument("--num_sampling_steps", type=int, default=50, help="Diffusion Sampling Steps")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument(
        "--text_prompt",
        type=str,
        nargs="+",
        help="A list of text prompts to be generated with. Also allow input a txt file or csv file.",
    )
    parser.add_argument("--tile_overlap_factor", type=float, default=0.25)

    parser.add_argument("--enable_tiling", action="store_true", help="whether to use vae tiling to save memory")
    parser.add_argument("--model_3d", action="store_true")
    parser.add_argument("--udit", action="store_true")
    parser.add_argument("--save_memory", action="store_true")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for dataloader")
    # MS new args
    parser.add_argument("--device", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--max_device_memory", type=str, default=None, help="e.g. `30GB` for 910a, `59GB` for 910b")
    parser.add_argument("--mode", default=0, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument(
        "--jit_syntax_level", default="strict", choices=["strict", "lax"], help="Set jit syntax level: strict or lax"
    )
    parser.add_argument("--seed", type=int, default=4, help="Inference seed")

    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--global_bf16", action="store_true", help="whether to enable gloabal bf16 for diffusion model training."
    )
    parser.add_argument(
        "--vae_precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for vae. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--vae_keep_gn_fp32",
        default=False,
        type=str2bool,
        help="whether keep GroupNorm in fp32. Defaults to False in inference mode. If training vae, better set it to True",
    )
    parser.add_argument(
        "--text_encoder_precision",
        default="fp16",
        type=str,
        choices=["bf16", "fp16"],
        help="what data type to use for T5 text encoder. Default is `bf16`, which corresponds to ms.bfloat16",
    )
    parser.add_argument(
        "--amp_level", type=str, default="O2", help="Set the amp level for the transformer model. Defaults to O2."
    )
    parser.add_argument(
        "--precision_mode",
        default=None,
        type=str,
        help="If specified, set the precision mode for Ascend configurations.",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="the number of images to be generated for each prompt"
    )
    parser.add_argument(
        "--save_latents",
        action="store_true",
        help="Whether to save latents (before vae decoding) instead of video files.",
    )
    parser.add_argument(
        "--decode_latents",
        action="store_true",
        help="whether to load the existing latents saved in npy files and run vae decoding",
    )
    parser.add_argument(
        "--video_extension", default="mp4", choices=["gif", "mp4"], help="The file extension to save videos"
    )
    parser.add_argument("--model_type", type=str, default="dit", choices=["dit", "udit", "latte"])
    parser.add_argument("--cache_dir", type=str, default="./")
    parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
    default_args = parser.parse_args()
    abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ""))
    if default_args.config:
        logger.info(f"Overwrite default arguments with configuration file {default_args.config}")
        default_args.config = os.path.join(abs_path, default_args.config)
        with open(default_args.config, "r") as f:
            cfg = yaml.safe_load(f)
            _check_cfgs_in_parser(cfg, parser)
            parser.set_defaults(**cfg)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # 1. init env
    args = parse_args()
    save_dir = args.save_img_path
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)
    # 1. init
    rank_id, device_num = init_env(
        args.mode,
        seed=args.seed,
        distributed=args.use_parallel,
        device_target=args.device,
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        precision_mode=args.precision_mode,
        global_bf16=args.global_bf16,
        sp_size=args.sp_size,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
    )

    # 2. vae model initiate and weight loading
    print_banner("vae init")
    vae = CausalVAEModelWrapper(args.ae_path, cache_dir=args.cache_dir, use_safetensors=True)
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor
        vae.vae.tile_sample_min_size = 512
        vae.vae.tile_latent_min_size = 64
        vae.vae.tile_sample_min_size_t = 29
        vae.vae.tile_latent_min_size_t = 8
        if args.save_memory:
            vae.vae.tile_sample_min_size = 256
            vae.vae.tile_latent_min_size = 32
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8
    vae.vae_scale_factor = ae_stride_config[args.ae]
    # use amp level O2 for causal 3D VAE with bfloat16 or float16
    vae_dtype = get_precision(args.vae_precision)
    if vae_dtype == ms.float16:
        custom_fp32_cells = [nn.GroupNorm] if args.vae_keep_gn_fp32 else []
    else:
        custom_fp32_cells = [nn.AvgPool2d, TrilinearInterpolate]
    vae = auto_mixed_precision(vae, amp_level="O2", dtype=vae_dtype, custom_fp32_cells=custom_fp32_cells)
    logger.info(f"Use amp level O2 for causal 3D VAE with dtype={vae_dtype}, custom_fp32_cells: {custom_fp32_cells}")
    vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    # 3. handle input text prompts
    print_banner("text prompts loading")
    ext = (
        f"{args.video_extension}" if not (args.save_latents or args.decode_latents) else "npy"
    )  # save video as gif or save denoised latents as npy files.
    ext = "jpg" if args.num_frames == 1 else ext
    if not isinstance(args.text_prompt, list):
        args.text_prompt = [args.text_prompt]
    # if input is a text file, where each line is a caption, load it into a list
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith("txt"):
        captions = open(args.text_prompt[0], "r").readlines()
        args.text_prompt = [i.strip() for i in captions]
    if len(args.text_prompt) == 1 and args.text_prompt[0].endswith("csv"):
        captions = pd.read_csv(args.text_prompt[0])
        args.text_prompt = [i.strip() for i in captions["cap"]]
    n = len(args.text_prompt)
    assert n > 0, "No captions provided"
    logger.info(f"Number of prompts: {n}")
    logger.info(f"Number of generated samples for each prompt {args.num_videos_per_prompt}")

    # create dataloader for the captions
    csv_file = {"path": [], "cap": []}
    for i in range(n):
        for i_video in range(args.num_videos_per_prompt):
            csv_file["path"].append(f"{i_video}-{args.text_prompt[i].strip()[:100]}.{ext}")
            csv_file["cap"].append(args.text_prompt[i])
    temp_dataset_csv = os.path.join(save_dir, "dataset.csv")
    pd.DataFrame.from_dict(csv_file).to_csv(temp_dataset_csv, index=False, columns=csv_file.keys())

    ds_config = dict(
        data_file_path=temp_dataset_csv,
        tokenizer=None,  # tokenizer,
        file_column="path",
        caption_column="cap",
    )
    dataset = create_dataloader(
        ds_config,
        args.batch_size,
        ds_name="text",
        num_parallel_workers=12,
        max_rowsize=32,
        shuffle=False,  # be in order
        device_num=device_num if not get_sequence_parallel_state() else (device_num // hccl_info.world_size),
        rank_id=rank_id if not get_sequence_parallel_state() else hccl_info.group_id,
        drop_remainder=False,
    )
    dataset_size = dataset.get_dataset_size()
    logger.info(f"Num batches: {dataset_size}")
    ds_iter = dataset.create_dict_iterator(1, output_numpy=True)

    if args.decode_latents:
        for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
            file_paths = data["file_path"]
            loaded_latents = []
            for i_sample in range(args.batch_size):
                save_fp = os.path.join(save_dir, file_paths[i_sample])
                assert os.path.exists(
                    save_fp
                ), f"{save_fp} does not exist! Please check the npy files under {save_dir} or check if you run `--save_latents` ahead."
                loaded_latents.append(np.load(save_fp))
            loaded_latents = (
                np.stack(loaded_latents) if loaded_latents[0].ndim == 4 else np.concatenate(loaded_latents, axis=0)
            )
            decode_data = (
                vae.decode(ms.Tensor(loaded_latents)).permute(0, 1, 3, 4, 2).to(ms.float32)
            )  # (b t c h w) -> (b t h w c)
            decode_data = ms.ops.clip_by_value(
                (decode_data + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0
            ).asnumpy()
            for i_sample in range(args.batch_size):
                save_fp = os.path.join(save_dir, file_paths[i_sample]).replace(".npy", f".{args.video_extension}")
                save_video_data = decode_data[i_sample : i_sample + 1]
                save_videos(save_video_data, save_fp, loop=0, fps=args.fps)  # (b t h w c)
        sys.exit()

    # 4. latte model initiate and weight loading
    print_banner("transformer model init")
    FA_dtype = get_precision(args.precision) if get_precision(args.precision) != ms.float32 else ms.bfloat16
    assert args.model_type == "dit", "Currently only suppport model_type as 'dit'@"
    if args.ms_checkpoint is not None and os.path.exists(args.ms_checkpoint):
        logger.info(f"Initiate from MindSpore checkpoint file {args.ms_checkpoint}")
        state_dict = ms.load_checkpoint(args.ms_checkpoint)
        # rm 'network.' prefix
        state_dict = dict(
            [k.replace("network.", "") if k.startswith("network.") else k, v] for k, v in state_dict.items()
        )
    else:
        state_dict = None

    model_version = args.model_path.split("/")[-1]
    if int(model_version.split("x")[0]) != args.num_frames:
        logger.warning(
            f"Detect that the loaded model version is {model_version}, but found a mismatched number of frames {model_version.split('x')[0]}"
        )
    if int(model_version.split("x")[1][:-1]) != args.height:
        logger.warning(
            f"Detect that the loaded model version is {model_version}, but found a mismatched resolution {args.height}x{args.width}"
        )
    transformer_model, logging_info = OpenSoraT2V.from_pretrained(
        args.model_path, state_dict=state_dict, cache_dir=args.cache_dir, FA_dtype=FA_dtype, output_loading_info=True
    )
    logger.info(logging_info)
    # mixed precision
    dtype = get_precision(args.precision)
    if args.precision in ["fp16", "bf16"]:
        if not args.global_bf16:
            amp_level = args.amp_level
            transformer_model = auto_mixed_precision(
                transformer_model,
                amp_level=args.amp_level,
                dtype=dtype,
                custom_fp32_cells=[LayerNorm, Attention, nn.SiLU, nn.GELU, PixArtAlphaCombinedTimestepSizeEmbeddings]
                if dtype == ms.float16
                else [
                    nn.MaxPool2d,
                    nn.MaxPool3d,
                    LayerNorm,
                    nn.SiLU,
                    nn.GELU,
                    PixArtAlphaCombinedTimestepSizeEmbeddings,
                ],
            )
            logger.info(
                f"Set mixed precision to {args.amp_level} with dtype={args.precision}, custom fp32_cells {custom_fp32_cells}"
            )
        else:
            logger.info(f"Using global bf16. Force model dtype from {dtype} to ms.bfloat16")
            dtype = ms.bfloat16
    elif args.precision == "fp32":
        amp_level = "O0"
    else:
        raise ValueError(f"Unsupported precision {args.precision}")

    transformer_model = transformer_model.set_train(False)
    for param in transformer_model.get_parameters():  # freeze transformer_model
        param.requires_grad = False

    print_banner("text encoder init")
    text_encoder_dtype = get_precision(args.text_encoder_precision)
    text_encoder, loading_info = MT5EncoderModel.from_pretrained(
        args.text_encoder_name,
        cache_dir=args.cache_dir,
        output_loading_info=True,
        mindspore_dtype=text_encoder_dtype,
        use_safetensors=True,
    )
    loading_info.pop("unexpected_keys")  # decoder weights are ignored
    logger.info(loading_info)
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder_name, cache_dir=args.cache_dir)

    # 3. build inference pipeline
    if args.sample_method == "DDIM":
        scheduler = DDIMScheduler()
    elif args.sample_method == "DDPM":
        scheduler = DDPMScheduler()
    elif args.sample_method == "PNDM":
        scheduler = PNDMScheduler()
    elif args.sample_method == "EulerDiscrete":
        scheduler = EulerDiscreteScheduler()
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler = EulerAncestralDiscreteScheduler()
    else:
        raise ValueError(f"Not supported sampling method {args.sample_method}")

    pipeline = OpenSoraPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        transformer=transformer_model,
    )

    # 4. print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_latte, num_params_latte_trainable = count_params(transformer_model)
    num_params = num_params_vae + num_params_latte
    num_params_trainable = num_params_vae_trainable + num_params_latte_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Jit level: {args.jit_level}",
            f"Num of samples: {n}",
            f"Num params: {num_params:,} (latte: {num_params_latte:,}, vae: {num_params_vae:,})",
            f"Num trainable params: {num_params_trainable:,}",
            f"Transformer dtype: {dtype}",
            f"VAE dtype: {vae_dtype}",
            f"Text encoder dtype: {text_encoder_dtype}",
            f"Sampling steps {args.num_sampling_steps}",
            f"Sampling method: {args.sample_method}",
            f"CFG guidance scale: {args.guidance_scale}",
            f"FA dtype: {FA_dtype}",
            f"Inference shape (num_frames x height x width): {args.num_frames}x{args.height}x{args.width}",
        ]
    )
    key_info += "\n" + "=" * 50
    logger.info(key_info)
    start_time = time.time()
    if args.profile:
        profiler = ms.Profiler(output_path="./mem_info", profile_memory=True)
        ms.set_context(memory_optimize_level="O0")
        ms.set_context(pynative_synchronize=True)
    else:
        profiler = None
    # infer
    for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
        prompt = [x for x in data["caption"]]
        file_paths = data["file_path"]
        positive_prompt = (
            "(masterpiece), (best quality), (ultra-detailed), {}. emotional, "
            + "harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
        )
        negative_prompt = (
            "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
            + "extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
        )

        videos = (
            pipeline(
                positive_prompt.format(prompt),
                negative_prompt=negative_prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                output_type="latents" if args.save_latents else "pil",
                max_sequence_length=args.max_sequence_length,
            )
            .images.to(ms.float32)
            .asnumpy()
        )
        if step == 0 and profiler is not None:
            profiler.stop()

        if get_sequence_parallel_state() and hccl_info.rank % hccl_info.world_size != 0:
            pass
        else:
            # save result
            for i_sample in range(args.batch_size):
                file_path = os.path.join(save_dir, file_paths[i_sample])
                assert ext in file_path, f"Only support saving as {ext} files, but got {file_path}."
                if args.save_latents:
                    np.save(file_path, videos[i_sample : i_sample + 1])
                else:
                    if args.num_frames == 1:
                        ext = "jpg"
                        image = videos[i_sample, 0]  # (b t h w c)  -> (h, w, c)
                        image = (image * 255).round().clip(0, 255).astype(np.uint8)
                        Image.fromarray(image).save(file_path)
                    else:
                        save_video_data = videos[i_sample : i_sample + 1]  # (b t h w c)
                        save_videos(save_video_data, file_path, loop=0, fps=args.fps)

    end_time = time.time()
    time_cost = end_time - start_time
    logger.info(f"Inference time cost: {time_cost:0.3f}s")
    logger.info(f"Inference speed: {n / time_cost:0.3f} samples/s")
    logger.info(f"{'latents' if args.save_latents else 'videos' } saved to {save_dir}")

    # delete files that are no longer needed
    if os.path.exists(temp_dataset_csv):
        os.remove(temp_dataset_csv)

    if args.decode_latents:
        npy_files = glob.glob(os.path.join(save_dir, "*.npy"))
        for fp in npy_files:
            os.remove(fp)
