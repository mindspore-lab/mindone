import argparse
import glob
import logging
import os
import time

import numpy as np
import pandas as pd
import yaml
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info
from opensora.dataset.text_dataset import create_dataloader
from opensora.models.causalvideovae import ae_stride_config, ae_wrapper
from opensora.models.diffusion.common import PatchEmbed2D
from opensora.models.diffusion.opensora.modeling_opensora import OpenSoraT2V_v1_3
from opensora.models.diffusion.opensora.modules import Attention, LayerNorm
from opensora.sample.pipeline_opensora import OpenSoraPipeline
from opensora.utils.message_utils import print_banner
from opensora.utils.utils import _check_cfgs_in_parser, get_precision, remove_invalid_characters
from opensora.utils.video_utils import save_videos
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import nn

from mindone.diffusers import DPMSolverSinglestepScheduler  # CogVideoXDDIMScheduler,
from mindone.diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    FlowMatchEulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
)
from mindone.diffusers.models.embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings
from mindone.diffusers.training_utils import set_seed
from mindone.transformers import CLIPTextModelWithProjection, MT5EncoderModel, T5EncoderModel
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.params import count_params

logger = logging.getLogger(__name__)


def get_scheduler(args):
    kwargs = dict(
        prediction_type=args.prediction_type,
        rescale_betas_zero_snr=args.rescale_betas_zero_snr,
        timestep_spacing="trailing" if args.rescale_betas_zero_snr else "leading",
    )
    if args.v1_5_scheduler:
        kwargs["beta_start"] = 0.00085
        kwargs["beta_end"] = 0.0120
        kwargs["beta_schedule"] = "scaled_linear"
    if args.sample_method == "DDIM":
        scheduler_cls = DDIMScheduler
        kwargs["clip_sample"] = False
    elif args.sample_method == "EulerDiscrete":
        scheduler_cls = EulerDiscreteScheduler
    elif args.sample_method == "DDPM":
        scheduler_cls = DDPMScheduler
        kwargs["clip_sample"] = False
    elif args.sample_method == "DPMSolverMultistep":
        scheduler_cls = DPMSolverMultistepScheduler
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif args.sample_method == "PNDM":
        scheduler_cls = PNDMScheduler
        kwargs.pop("rescale_betas_zero_snr", None)
    elif args.sample_method == "HeunDiscrete":
        scheduler_cls = HeunDiscreteScheduler
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif args.sample_method == "DEISMultistep":
        scheduler_cls = DEISMultistepScheduler
        kwargs.pop("rescale_betas_zero_snr", None)
    elif args.sample_method == "KDPM2AncestralDiscrete":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    # elif args.sample_method == 'CogVideoX':
    #     scheduler_cls = CogVideoXDDIMScheduler
    elif args.sample_method == "FlowMatchEulerDiscrete":
        scheduler_cls = FlowMatchEulerDiscreteScheduler
        kwargs = {}
    else:
        raise NameError(f"Unsupport sample_method {args.sample_method}")
    scheduler = scheduler_cls(**kwargs)
    return scheduler


def prepare_pipeline(args):
    # VAE model initiate and weight loading
    print_banner("vae init")
    vae_dtype = get_precision(args.vae_precision)
    kwarg = {
        "use_safetensors": True,
        "dtype": vae_dtype,
    }
    vae = ae_wrapper[args.ae](args.ae_path, **kwarg)
    vae.vae_scale_factor = ae_stride_config[args.ae]
    if args.enable_tiling:
        vae.vae.enable_tiling()
        vae.vae.tile_overlap_factor = args.tile_overlap_factor

    vae.set_train(False)
    for param in vae.get_parameters():  # freeze vae
        param.requires_grad = False

    if args.decode_latents:
        print("To decode latents directly, skipped loading text endoers and transformer")
        return vae

    # Build text encoders
    print_banner("text encoder init")
    text_encoder_dtype = get_precision(args.text_encoder_precision)
    if "mt5" in args.text_encoder_name_1:
        text_encoder_1, loading_info = MT5EncoderModel.from_pretrained(
            args.text_encoder_name_1,
            cache_dir=args.cache_dir,
            output_loading_info=True,
            mindspore_dtype=text_encoder_dtype,
            use_safetensors=True,
        )
        # loading_info.pop("unexpected_keys")  # decoder weights are ignored
        # logger.info(f"Loaded MT5 Encoder: {loading_info}")
        text_encoder_1 = text_encoder_1.set_train(False)
    else:
        text_encoder_1 = T5EncoderModel.from_pretrained(
            args.text_encoder_name_1, cache_dir=args.cache_dir, mindspore_dtype=text_encoder_dtype
        ).set_train(False)
    tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=args.cache_dir)

    if args.text_encoder_name_2 is not None:
        text_encoder_2, loading_info = CLIPTextModelWithProjection.from_pretrained(
            args.text_encoder_name_2,
            cache_dir=args.cache_dir,
            mindspore_dtype=text_encoder_dtype,
            output_loading_info=True,
            use_safetensors=True,
        )
        # loading_info.pop("unexpected_keys")  # only load text model, ignore vision model
        # loading_info.pop("mising_keys") # Note: missed keys when loading open-clip models
        # logger.info(f"Loaded CLIP Encoder: {loading_info}")
        text_encoder_2 = text_encoder_2.set_train(False)
        tokenizer_2 = AutoTokenizer.from_pretrained(args.text_encoder_name_2, cache_dir=args.cache_dir)
    else:
        text_encoder_2, tokenizer_2 = None, None

    # Build transformer
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
        state_dict = dict([k.replace("_backbone.", "") if "_backbone." in k else k, v] for k, v in state_dict.items())
    else:
        state_dict = None
    model_version = args.model_path.split("/")[-1]
    if (args.version != "v1_3") and (model_version.split("x")[0][:3] != "any"):
        if int(model_version.split("x")[0]) != args.num_frames:
            logger.warning(
                f"Detect that the loaded model version is {model_version}, but found a mismatched number of frames {model_version.split('x')[0]}"
            )
        if int(model_version.split("x")[1][:-1]) != args.height:
            logger.warning(
                f"Detect that the loaded model version is {model_version}, but found a mismatched resolution {args.height}x{args.width}"
            )
    elif (args.version == "v1_3") and (
        model_version.split("x")[0] == "any93x640x640"
    ):  # TODO: currently only release one model
        if (args.height % 32 != 0) or (args.width % 32 != 0):
            logger.warning(
                f"Detect that the loaded model version is {model_version}, but found a mismatched resolution {args.height}x{args.width}. \
                    The resolution of the inference should be a multiple of 32."
            )
        if (args.num_frames - 1) % 4 != 0:
            logger.warning(
                f"Detect that the loaded model version is {model_version}, but found a mismatched number of frames {args.num_frames}. \
                    Frames needs to be 4n+1, e.g. 93, 77, 61, 45, 29, 1 (image)"
            )
    if args.version == "v1_3":
        # TODO
        # if args.model_type == 'inpaint' or args.model_type == 'i2v':
        #     transformer_model = OpenSoraInpaint_v1_3.from_pretrained(
        #         args.model_path, cache_dir=args.cache_dir,
        #         device_map=None, mindspore_dtype=weight_dtype
        #         ).set_train(False)
        # else:

        transformer_model, logging_info = OpenSoraT2V_v1_3.from_pretrained(
            args.model_path,
            state_dict=state_dict,
            cache_dir=args.cache_dir,
            FA_dtype=FA_dtype,
            output_loading_info=True,
        )
        logger.info(logging_info)
    elif args.version == "v1_5":
        if args.model_type == "inpaint" or args.model_type == "i2v":
            raise NotImplementedError("Inpainting model is not available in v1_5")
        else:
            from opensora.models.diffusion.opensora_v1_5.modeling_opensora import OpenSoraT2V_v1_5

            weight_dtype = ms.float32
            transformer_model = OpenSoraT2V_v1_5.from_pretrained(
                args.model_path,
                cache_dir=args.cache_dir,
                # device_map=None,
                mindspore_dtype=weight_dtype,
            )

    # Mixed precision
    dtype = get_precision(args.precision)
    if args.precision in ["fp16", "bf16"]:
        if dtype == ms.float16:
            custom_fp32_cells = [
                LayerNorm,
                Attention,
                PatchEmbed2D,
                nn.SiLU,
                nn.GELU,
                PixArtAlphaCombinedTimestepSizeEmbeddings,
            ]
        else:
            custom_fp32_cells = [
                nn.MaxPool2d,
                nn.MaxPool3d,  # do not support bf16
                PatchEmbed2D,  # low accuracy if using bf16
                LayerNorm,
                nn.SiLU,
                nn.GELU,
                PixArtAlphaCombinedTimestepSizeEmbeddings,
            ]
        transformer_model = auto_mixed_precision(
            transformer_model, amp_level=args.amp_level, dtype=dtype, custom_fp32_cells=custom_fp32_cells
        )
        logger.info(
            f"Set mixed precision to {args.amp_level} with dtype={args.precision}, custom fp32_cells {custom_fp32_cells}"
        )

    elif args.precision == "fp32":
        pass
    else:
        raise ValueError(f"Unsupported precision {args.precision}")
    transformer_model = transformer_model.set_train(False)
    for param in transformer_model.get_parameters():  # freeze transformer_model
        param.requires_grad = False

    # Build scheduler
    scheduler = get_scheduler(args)

    # Build inference pipeline
    # pipeline_class = OpenSoraInpaintPipeline if args.model_type == 'inpaint' or args.model_type == 'i2v' else OpenSoraPipeline
    pipeline_class = OpenSoraPipeline

    pipeline = pipeline_class(
        vae=vae,
        text_encoder=text_encoder_1,
        tokenizer=tokenizer_1,
        scheduler=scheduler,
        transformer=transformer_model,
        text_encoder_2=text_encoder_2,
        tokenizer_2=tokenizer_2,
    )

    if args.save_memory:  # TODO: Susan comment: I am not sure yet
        print("enable_model_cpu_offload AND enable_sequential_cpu_offload AND enable_tiling")
        pipeline.enable_model_cpu_offload()
        pipeline.enable_sequential_cpu_offload()
        if not args.enable_tiling:
            vae.vae.enable_tiling()
        vae.vae.t_chunk_enc = 8
        vae.vae.t_chunk_dec = vae.vae.t_chunk_enc // 2

    # Print key info
    num_params_vae, num_params_vae_trainable = count_params(vae)
    num_params_latte, num_params_latte_trainable = count_params(transformer_model)
    num_params = num_params_vae + num_params_latte
    num_params_trainable = num_params_vae_trainable + num_params_latte_trainable
    key_info = "Key Settings:\n" + "=" * 50 + "\n"
    key_info += "\n".join(
        [
            f"MindSpore mode[GRAPH(0)/PYNATIVE(1)]: {args.mode}",
            f"Jit level: {args.jit_level}",
            f"Distributed mode: {args.use_parallel}"
            + (
                f"\nParallel mode: {args.parallel_mode}"
                + (f"{args.zero_stage}" if args.parallel_mode == "zero" else "")
                if args.use_parallel
                else ""
            )
            + (f"\nsp_size: {args.sp_size}" if args.sp_size != 1 else ""),
            f"Num of samples: {len(args.text_prompt)}",
            f"Num params: {num_params:,} (dit: {num_params_latte:,}, vae: {num_params_vae:,})",
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

    return pipeline


# See npu_config.py set_npu_env()
# def init_npu_env(args):
#     local_rank = int(os.getenv('RANK', 0))
#     world_size = int(os.getenv('WORLD_SIZE', 1))
#     args.local_rank = local_rank
#     args.world_size = world_size
#     torch_npu.npu.set_device(local_rank)
#     dist.init_process_group(
#         backend='hccl', init_method='env://',
#         world_size=world_size, rank=local_rank
#         )
#     if args.sp:
#         initialize_sequence_parallel_state(world_size)
#     return args


def run_model_and_save_samples(
    args, pipeline, rank_id, device_num, save_dir, caption_refiner_model=None, enhance_video_model=None
):
    if args.seed is not None:
        set_seed(args.seed, rank=rank_id)

    # Handle input text prompts
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

    # Create dataloader for the captions
    csv_file = {"path": [], "cap": []}
    for i in range(n):
        for i_video in range(args.num_videos_per_prompt):
            csv_file["path"].append(
                remove_invalid_characters(f"{i_video}-{args.text_prompt[i].strip()[:100]}.{ext}")
            )  # a valid file name
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

    # Decode latents directly
    if args.decode_latents:
        assert isinstance(pipeline, ae_wrapper[args.ae])
        vae = pipeline
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

        # Delete files that are no longer needed
        if os.path.exists(temp_dataset_csv):
            os.remove(temp_dataset_csv)

        if args.decode_latents:
            npy_files = glob.glob(os.path.join(save_dir, "*.npy"))
            for fp in npy_files:
                os.remove(fp)

    # TODO
    # if args.model_type == 'inpaint' or args.model_type == 'i2v':
    #     if not isinstance(args.conditional_pixel_values_path, list):
    #         args.conditional_pixel_values_path = [args.conditional_pixel_values_path]
    #     if len(args.conditional_pixel_values_path) == 1 and args.conditional_pixel_values_path[0].endswith('txt'):
    #         temp = open(args.conditional_pixel_values_path[0], 'r').readlines()
    #         conditional_pixel_values_path = [i.strip().split(',') for i in temp]
    #     mask_type = args.mask_type if args.mask_type is not None else None

    positive_prompt = """
    high quality, high aesthetic, {}
    """
    negative_prompt = """
    nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality,
    low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
    """
    # positive_prompt = (
    #     "(masterpiece), (best quality), (ultra-detailed), {}. emotional, "
    #     + "harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, sharp focus, high budget, cinemascope, moody, epic, gorgeous"
    # )
    # negative_prompt = (
    #     "nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, "
    #     + "extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"
    # )

    def generate(step, data, ext, conditional_pixel_values_path=None, mask_type=None):
        prompt = [x for x in data["caption"]]
        if args.caption_refiner is not None:
            if args.model_type != "inpaint" and args.model_type != "i2v":
                refine_prompt = caption_refiner_model.get_refiner_output(prompt)
                print(f"\nOrigin prompt: {prompt}\n->\nRefine prompt: {refine_prompt}")
                prompt = refine_prompt
            else:
                # Due to the current use of LLM as the caption refiner, additional content that is not present in the
                # control image will be added. Therefore, caption refiner is not used in this mode.
                print("Caption refiner is not available for inpainting model, use the original prompt...")
                time.sleep(3)
        # TODO
        # input_prompt = positive_prompt.format(prompt)
        # if args.model_type == 'inpaint' or args.model_type == 'i2v':
        #     print(f'\nConditional pixel values path: {conditional_pixel_values_path}')
        #     videos = pipeline(
        #         conditional_pixel_values_path=conditional_pixel_values_path,
        #         mask_type=mask_type,
        #         crop_for_hw=args.crop_for_hw,
        #         max_hxw=args.max_hxw,
        #         prompt=input_prompt,
        #         negative_prompt=negative_prompt,
        #         num_frames=args.num_frames,
        #         height=args.height,
        #         width=args.width,
        #         num_inference_steps=args.num_sampling_steps,
        #         guidance_scale=args.guidance_scale,
        #         num_samples_per_prompt=args.num_samples_per_prompt,
        #         max_sequence_length=args.max_sequence_length,
        #     ).videos
        # else:
        file_paths = data["file_path"]
        input_prompt = positive_prompt.format(prompt[0])  # remove "[]"

        videos = (
            pipeline(
                input_prompt,
                negative_prompt=negative_prompt,
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_sampling_steps,
                guidance_scale=args.guidance_scale,
                num_samples_per_prompt=args.num_samples_per_prompt,
                output_type="latents" if args.save_latents else "pil",
                max_sequence_length=args.max_sequence_length,
            )
            .videos.to(ms.float32)
            .asnumpy()
        )
        # if enhance_video_model is not None:
        #     # b t h w c
        #     videos = enhance_video_model.enhance_a_video(videos, input_prompt, 2.0, args.fps, 250)
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

    if args.profile:
        profiler = ms.Profiler(output_path="./mem_info", profile_memory=True)
        ms.set_context(memory_optimize_level="O0")
        ms.set_context(pynative_synchronize=True)
    else:
        profiler = None

    # Infer
    # if args.model_type == 'inpaint' or args.model_type == 'i2v':
    #     for index, (prompt, cond_path) in enumerate(zip(args.text_prompt, conditional_pixel_values_path)):
    #         if not args.sp and args.local_rank != -1 and index % args.world_size != args.local_rank:
    #             continue
    #         generate(prompt, conditional_pixel_values_path=cond_path, mask_type=mask_type)
    #     print('completed, please check the saved images and videos')
    # else:
    for step, data in tqdm(enumerate(ds_iter), total=dataset_size):
        generate(step, data, ext)

    # Delete files that are no longer needed
    if os.path.exists(temp_dataset_csv):
        os.remove(temp_dataset_csv)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--version", type=str, default="v1_3", choices=["v1_3", "v1_5"])
    parser.add_argument("--caption_refiner", type=str, default=None, help="caption refiner model path")
    parser.add_argument("--enhance_video", type=str, default=None)
    parser.add_argument(
        "--text_encoder_name_1", type=str, default="DeepFloyd/t5-v1_1-xxl", help="google/mt5-xxl, DeepFloyd/t5-v1_1-xxl"
    )
    parser.add_argument(
        "--text_encoder_name_2",
        type=str,
        default=None,
        help=" openai/clip-vit-large-patch14, (laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)",
    )
    parser.add_argument("--num_samples_per_prompt", type=int, default=1)
    parser.add_argument("--refine_caption", action="store_true")
    # parser.add_argument('--compile', action='store_true')
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="epsilon",
        help="The prediction_type that shall be used for training. Choose between 'epsilon' or 'v_prediction' or leave `None`. \
            If left to `None` the default prediction type of the scheduler: `noise_scheduler.config.prediciton_type` is chosen.",
    )
    parser.add_argument("--rescale_betas_zero_snr", action="store_true")
    # parser.add_argument('--local_rank', type=int, default=-1)
    # parser.add_argument('--world_size', type=int, default=1)
    # parser.add_argument('--sp', action='store_true')
    parser.add_argument("--v1_5_scheduler", action="store_true")
    parser.add_argument("--conditional_pixel_values_path", type=str, default=None)
    parser.add_argument("--mask_type", type=str, default=None)
    parser.add_argument("--crop_for_hw", action="store_true")
    parser.add_argument("--max_hxw", type=int, default=236544)  # 236544=512x462????

    parser.add_argument(
        "--config",
        "-c",
        default="",
        type=str,
        help="path to load a config yaml file that describes the setting which will override the default arguments",
    )
    parser.add_argument("--model_path", type=str, default="LanguageBind/Open-Sora-Plan-v1.3.0")
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
    parser.add_argument("--max_sequence_length", type=int, default=512, help="the maximum text tokens length")

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
    parser.add_argument("--mode", default=1, type=int, help="Specify the mode: 0 for graph mode, 1 for pynative mode")
    parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--parallel_mode", default="data", type=str, choices=["data", "optim"], help="parallel mode: data, optim"
    )
    parser.add_argument("--jit_level", default="O0", help="Set jit level: # O0: KBK, O1:DVM, O2: GE")
    parser.add_argument(
        "--jit_syntax_level", default="strict", choices=["strict", "lax"], help="Set jit syntax level: strict or lax"
    )
    parser.add_argument("--seed", type=int, default=42, help="Inference seed")

    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what data type to use for latte. Default is `fp16`, which corresponds to ms.float16",
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
    parser.add_argument(
        "--model_type", type=str, default="dit", choices=["dit", "udit", "latte", "t2v", "inpaint", "i2v"]
    )
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

    assert not (args.use_parallel and args.num_frames == 1)

    return args
