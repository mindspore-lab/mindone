import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.visualize.videos import save_videos

# from hyvideo.utils.file_utils import save_videos_grid
sys.path.append(".")
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
from hyvideo.utils.file_utils import process_prompt_and_text_embed
from hyvideo.utils.ms_utils import init_env


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_dir = args.save_path if args.save_path_suffix == "" else f"{args.save_path}_{args.save_path_suffix}"
    if not os.path.exists(args.save_path):
        os.makedirs(save_dir, exist_ok=True)

    # ms env init
    rank_id, _ = init_env(
        args.ms_mode,
        seed=42 if args.seed is None else args.seed,
        distributed=args.use_parallel,
        device_target="Ascend",
        max_device_memory=args.max_device_memory,
        parallel_mode=args.parallel_mode,
        sp_size=args.sp_size,
        jit_level=args.jit_level,
        jit_syntax_level=args.jit_syntax_level,
    )

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args, rank_id=rank_id)

    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    if args.prompt is None and args.text_embed_path is None:
        raise ValueError("Either `prompt` or `text_embed_path` must be provided.")

    prompts, text_embed_paths = process_prompt_and_text_embed(args.prompt, args.text_embed_path)
    for prompt, text_embed_path in zip(prompts, text_embed_paths):
        if prompt is not None:
            logger.info(f"Sampling with prompt: {prompt}")
        if text_embed_path is not None:
            logger.info(f"Sampling with text embed path: {text_embed_path}")
        try:
            outputs = hunyuan_video_sampler.predict(
                prompt=prompt,
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
                output_type=args.output_type,
                text_embed_path=text_embed_path,
            )
            samples = outputs["samples"]
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            continue

        # Save samples
        if rank_id == 0:
            for i, sample in enumerate(samples):
                sample = samples[i].unsqueeze(0)
                time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
                save_path = f"{save_dir}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"

                if args.output_type != "latent":
                    # save_videos_grid(sample, save_path, fps=24)
                    # b c t h w -> b t h w c
                    sample = sample.permute(0, 2, 3, 4, 1).asnumpy()
                    save_videos(sample, save_path, fps=24)
                else:
                    save_path = save_path[:-4] + ".npy"
                    np.save(save_path, sample)

                logger.info(f"Sample save to: {save_path}")


if __name__ == "__main__":
    main()
