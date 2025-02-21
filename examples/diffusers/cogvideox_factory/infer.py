import argparse
import pathlib
import shutil

import numpy as np

import mindspore as ms

from mindone import diffusers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="THUDM/CogVideoX1.5-5b",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="prompt, if None, will set default prompt.",
    )
    parser.add_argument(
        "--transformer_ckpt_path",
        type=str,
        default=None,
        help="Path to the transformer checkpoint.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1360,
        help="The height of the output video.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=768,
        help="The width of the output video.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=77,
        help="The frame of the output video.",
    )
    parser.add_argument(
        "--npy_output_path",
        type=str,
        default=None,
        help="Path to save the inferred numpy array.",
    )
    parser.add_argument(
        "--video_output_path",
        type=str,
        default=None,
        help="Path to save the inferred video.",
    )
    return parser.parse_args()


def infer(args: argparse.Namespace) -> None:
    ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O1"})
    pipe = diffusers.CogVideoXPipeline.from_pretrained(
        args.pretrained_model_name_or_path, mindspore_dtype=ms.bfloat16, use_safetensors=True
    )

    if args.transformer_ckpt_path is not None:
        ckpt = ms.load_checkpoint(args.transformer_ckpt_path)
        processed_ckpt = {name[12:]: value for name, value in ckpt.items()}  # remove "transformer." prefix
        param_not_load, ckpt_not_load = ms.load_param_into_net(pipe.transformer, processed_ckpt)
        if param_not_load:
            raise RuntimeError(f"{param_not_load} was not loaded into net.")
        if ckpt_not_load:
            raise RuntimeError(f"{ckpt_not_load} was not loaded from checkpoint.")
        print("Successfully loaded transformer checkpoint.")

    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
    prompt = prompt if args.prompt is None else args.prompt
    video = pipe(prompt=prompt, height=args.height, width=args.width, num_frames=args.frame)[0][0]

    if args.npy_output_path is not None:
        path = pathlib.Path(args.npy_output_path)
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        index_max_length = len(str(len(video)))
        for index, image in enumerate(video):
            np.save(path / f"image_{str(index).zfill(index_max_length)}", np.array(image))
        print("Successfully saved the inferred numpy array.")

    if args.video_output_path is not None:
        diffusers.utils.export_to_video(video, args.video_output_path, fps=8)
        print("Successfully saved the inferred video.")


if __name__ == "__main__":
    args = parse_args()
    infer(args)
