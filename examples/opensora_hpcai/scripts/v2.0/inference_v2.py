import glob
import logging
import os
import sys
from time import perf_counter
from typing import Optional, Union

import numpy as np
from jsonargparse import ActionConfigFile, ArgumentParser
from jsonargparse.typing import Path_fr, path_type

from mindspore import runtime
from mindspore.communication import GlobalComm

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../../"))
sys.path.append(mindone_lib_path)
sys.path.append(os.path.join(__dir__, "../.."))

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.models.flux_vae.autoencoder import AutoEncoderFlux
from opensora.models.hunyuan_vae.autoencoder_kl_causal_3d import CausalVAE3D_HUNYUAN
from opensora.models.mmdit import Flux
from opensora.models.text_encoder import HFEmbedder
from opensora.pipelines.denoisers import DistilledDenoiser, I2VDenoiser
from opensora.pipelines.infer_pipeline_v2 import InferPipelineV2
from opensora.utils.inference import process_and_save
from opensora.utils.sampling import SamplingOption
from opensora.utils.saving import SavingOptions

from mindone.utils import init_env, set_logger

logger = logging.getLogger(__name__)

Path_dr = path_type("dr", docstring="path to a directory that exists and is readable")


def prepare_captions(
    prompts: Union[str, Path_fr] = "",
    neg_prompts: Union[str, Path_fr] = "",
    t5_dir: Optional[Path_dr] = None,
    clip_dir: Optional[Path_dr] = None,
    neg_t5_dir: Optional[Path_dr] = None,
    neg_clip_dir: Optional[Path_dr] = None,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    if prompts:
        if isinstance(prompts, str):
            logger.info(f"Prompt: {prompts}")
            return [prompts], [neg_prompts], [], [], [], []
        else:
            logger.info(f"Reading prompts from file {prompts}")
            with open(prompts, "r", encoding="utf-8") as f:
                prompts = f.read().splitlines()
            return prompts, [neg_prompts], [], [], [], []  # TODO: neg prompts as a file

    elif t5_dir is not None and clip_dir is not None:
        t5_emb = sorted(glob.glob(os.path.join(t5_dir, "*.npy")))
        clip_emb = sorted(glob.glob(os.path.join(clip_dir, "*.npy")))
        neg_t5_emb = sorted(glob.glob(os.path.join(neg_t5_dir, "*.npy")))
        neg_clip_emb = sorted(glob.glob(os.path.join(neg_clip_dir, "*.npy")))
        if len(t5_emb) != len(clip_emb) != len(neg_t5_emb) != len(neg_clip_emb):
            raise ValueError(
                f"t5_dir ({len(t5_emb)}), clip_dir ({len(clip_emb)}), neg_t5_dir ({len(neg_t5_emb)}),"
                f" and neg_clip_dir ({len(neg_clip_emb)})  must contain the same number of files"
            )
        return [], [], t5_emb, neg_t5_emb, clip_emb, neg_clip_emb

    else:
        raise ValueError("Either `prompts` or `t5_dir` and `clip_dir` must be specified.")


def main(args):
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    saving_options = SavingOptions(**args.saving_option)
    save_dir = saving_options.output_path
    os.makedirs(save_dir, exist_ok=True)
    set_logger(name="", output_dir=save_dir)

    _, rank_id, device_num = init_env(**args.env)

    # == prepare default params ==
    sampling_option = SamplingOption(**args.sampling_option)

    type_name = "image" if sampling_option.num_frames == 1 else "video"
    sub_dir = f"{type_name}_{sampling_option.resolution}"
    os.makedirs(os.path.join(save_dir, sub_dir), exist_ok=True)
    if sampling_option.use_t2i2v:
        os.makedirs(os.path.join(save_dir, sub_dir, "generated_condition"), exist_ok=True)

    # ======================================================
    # 2. prepare captions
    # ======================================================
    (
        full_prompts,
        full_neg_prompts,
        full_t5_embeds,
        full_neg_t5_embeds,
        full_clip_embeds,
        full_neg_clip_embeds,
    ) = prepare_captions(**args.prompts)
    # split captions across devices
    prompts, neg_prompts = full_prompts[rank_id::device_num], full_neg_prompts[rank_id::device_num]
    t5_embeds, neg_t5_embeds = full_t5_embeds[rank_id::device_num], full_neg_t5_embeds[rank_id::device_num]
    clip_embeds, neg_clip_embeds = full_clip_embeds[rank_id::device_num], full_neg_clip_embeds[rank_id::device_num]
    num_prompts, full_num_prompts = max(len(prompts), len(t5_embeds)), max(len(full_prompts), len(full_t5_embeds))
    ids = list(range(full_num_prompts))[rank_id::device_num]  # samples id generated on a current device

    if args.enable_sequence_parallel:
        logger.info("Sequence parallelism is enabled. Captions will be divided during the video generation stage.")
        logger.info(f"Total number of captions: {full_num_prompts}")
    logger.info(f"Number of captions{f' for rank {rank_id}' if device_num > 1 else ''}: {num_prompts}")

    # ======================================================
    # 3. build model
    # ======================================================
    t5_model, clip_model = None, None
    if prompts:
        logger.info("Building text embedder models...")
        t5_model = HFEmbedder(**args.t5)
        clip_model = HFEmbedder(**args.clip)
        # TODO: online text embedding

    paths = []
    if sampling_option.use_t2i2v and num_prompts:
        sampling_option_t2i = SamplingOption(**args.sampling_option_t2i)
        logger.info("Building image model...")
        model_img_flux = Flux(**args.img_model).set_train(False)
        logger.info("Building image AE model...")
        model_ae_img_flux = AutoEncoderFlux(**args.img_ae).set_train(False)

        denoiser_t2i = DistilledDenoiser()
        pipeline_img = InferPipelineV2(
            model_img_flux, model_ae_img_flux, t5_model, clip_model, num_inference_steps=sampling_option_t2i.num_steps
        )

        logger.info("Generating image condition with Flux...")
        for p in range(0, num_prompts, sampling_option_t2i.batch_size):
            prompt = prompts[p : p + sampling_option_t2i.batch_size]
            neg_prompt = neg_prompts[p : p + sampling_option_t2i.batch_size]
            t5_emb = t5_embeds[p : p + sampling_option_t2i.batch_size]
            neg_t5_emb = neg_t5_embeds[p : p + sampling_option_t2i.batch_size]
            clip_emb = clip_embeds[p : p + sampling_option_t2i.batch_size]
            neg_clip_emb = neg_clip_embeds[p : p + sampling_option_t2i.batch_size]
            save_ids = ids[p : p + sampling_option_t2i.batch_size]
            # read cached embeddings, if any
            if t5_emb and clip_emb:
                t5_emb = np.array([np.load(emb) for emb in t5_emb])
                neg_t5_emb = np.array([np.load(emb) for emb in neg_t5_emb])
                clip_emb = np.array([np.load(emb) for emb in clip_emb])
                neg_clip_emb = np.array([np.load(emb) for emb in neg_clip_emb])

            for i in range(sampling_option_t2i.num_samples):  # generate multiple samples with different seeds
                start = perf_counter()
                images, img_latents = pipeline_img(
                    text=prompt,
                    neg_text=neg_prompt,
                    t5_emb=t5_emb,
                    neg_t5_emb=neg_t5_emb,
                    clip_emb=clip_emb,
                    neg_clip_emb=neg_clip_emb,
                    denoiser=denoiser_t2i,
                    opt=sampling_option_t2i,
                    cond_type="t2v",  # FIXME: why fixed?
                    channel=model_img_flux.in_channels,
                )
                logger.info(f"Image generation time: {perf_counter() - start:.2f} s")

                if images is not None:
                    sample_paths = process_and_save(
                        images,
                        ids=save_ids,
                        sub_ids=[i] * len(images) if sampling_option_t2i.num_samples > 1 else [],
                        save_dir=saving_options.output_path,
                    )
                    paths.extend(sample_paths)
                    for path in sample_paths:
                        logger.info(f"Images saved to: {path}")

        del pipeline_img, model_img_flux, model_ae_img_flux  # release NPU memory

    if args.enable_sequence_parallel:
        set_sequence_parallel_group(GlobalComm.WORLD_COMM_GROUP)
        # restore all prompts on each device
        prompts, neg_prompts = full_prompts, full_neg_prompts
        t5_embeds, neg_t5_embeds = full_t5_embeds, full_neg_t5_embeds
        clip_embeds, neg_clip_embeds = full_clip_embeds, full_neg_clip_embeds
        num_prompts = max(len(prompts), len(t5_embeds))

    logger.info("Building video model...")
    model = Flux(**args.model).set_train(False)
    logger.info("Building video AE model...")
    ae = CausalVAE3D_HUNYUAN(**args.ae).set_train(False)  # FIXME: add DC-AE support
    denoiser_i2v = I2VDenoiser()
    pipeline_vid = InferPipelineV2(model, ae, t5_model, clip_model, num_inference_steps=sampling_option.num_steps)

    # ======================================================
    # 4. inference
    # ======================================================
    cond_type = "i2v_head" if sampling_option.use_t2i2v else "t2v"  # TODO: refactor it
    for p in range(0, num_prompts, sampling_option.batch_size):
        prompt = prompts[p : p + sampling_option.batch_size]
        neg_prompt = neg_prompts[p : p + sampling_option.batch_size]
        t5_emb = t5_embeds[p : p + sampling_option.batch_size]
        neg_t5_emb = neg_t5_embeds[p : p + sampling_option.batch_size]
        clip_emb = clip_embeds[p : p + sampling_option.batch_size]
        neg_clip_emb = neg_clip_embeds[p : p + sampling_option.batch_size]
        save_ids = ids[p : p + sampling_option.batch_size]
        # read cached embeddings, if any
        if t5_emb and clip_emb:
            t5_emb = np.array([np.load(emb) for emb in t5_emb])
            neg_t5_emb = np.array([np.load(emb) for emb in neg_t5_emb])
            clip_emb = np.array([np.load(emb) for emb in clip_emb])
            neg_clip_emb = np.array([np.load(emb) for emb in neg_clip_emb])

        for i in range(sampling_option.num_samples):  # generate multiple samples with different seeds
            # TODO: add FPS and motion score to prompts
            logger.info("Generating video...")
            start = perf_counter()
            videos, vid_latents = pipeline_vid(
                text=prompt,
                neg_text=neg_prompt,
                t5_emb=t5_emb,
                neg_t5_emb=neg_t5_emb,
                clip_emb=clip_emb,
                neg_clip_emb=neg_clip_emb,
                denoiser=denoiser_i2v,
                opt=sampling_option,
                cond_type=cond_type,
                channel=model.in_channels,
                references=paths[p : p + sampling_option.batch_size],
            )
            logger.info(f"Video generation time: {perf_counter() - start:.2f} s")

            if args.enable_sequence_parallel and rank_id > 0:
                # in sequence parallel mode, results from all ranks are identical, so save results only from rank 0
                continue

            if videos is not None:
                sample_paths = process_and_save(
                    videos,
                    ids=save_ids,
                    sub_ids=[i] * len(videos) if sampling_option.num_samples > 1 else [],
                    save_dir=saving_options.output_path,
                )
                for path in sample_paths:
                    logger.info(f"Videos saved to: {path}")

    logger.info("Inference finished.")
    logger.info(f"NPU max memory max memory allocated: {runtime.max_memory_allocated() / 1024**3:.1f} GB")
    logger.info(f"NPU max memory max memory reserved: {runtime.max_memory_reserved() / 1024**3:.1f} GB")


if __name__ == "__main__":
    parser = ArgumentParser(description="OpenSora v2.0 inference script.")
    parser.add_argument(
        "-c",
        "--config",
        action=ActionConfigFile,
        help="Path to load a config yaml file that describes the setting which will override the default arguments.",
    )
    parser.add_function_arguments(init_env, "env")
    parser.add_function_arguments(prepare_captions, "prompts")
    parser.add_function_arguments(Flux, "model")
    parser.add_function_arguments(CausalVAE3D_HUNYUAN, "ae")
    parser.add_function_arguments(Flux, "img_model")
    parser.add_function_arguments(AutoEncoderFlux, "img_ae")
    parser.add_argument(
        "--enable_sequence_parallel", type=bool, default=False, help="Enable sequence parallel for video generation."
    )
    parser.add_class_arguments(HFEmbedder, "t5", instantiate=False)
    parser.add_class_arguments(HFEmbedder, "clip", instantiate=False)
    parser.add_argument("sampling_option", type=SamplingOption)
    parser.add_argument("sampling_option_t2i", type=SamplingOption)
    parser.add_argument("saving_option", type=SavingOptions)
    cfg = parser.parse_args()
    main(cfg)
