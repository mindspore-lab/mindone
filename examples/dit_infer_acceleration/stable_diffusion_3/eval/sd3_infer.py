import os
import sys
import time

current_path = os.path.abspath(os.path.dirname(__file__))
parent_path = os.path.abspath(os.path.dirname(current_path))
sys.path.append(parent_path)

import argparse

import mindspore as ms

from mindone.diffusers import StableDiffusion3Pipeline


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Stable Diffusion 3 checkpoint",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        nargs="+",
        type=str,
        default="A cat holding a sign that says hello world",
        help="Prompt, do not support prompt list",
    )
    parser.add_argument("--negative_prompt", type=str, nargs="+", default="", help="Negative Prompt")
    parser.add_argument("--cache_and_prompt_gate", action="store_true", help="Use DiTCache and PromptGate algorithm")
    parser.add_argument("--todo", action="store_true", help="Use ToDo algorithm")
    parser.add_argument("--output_type", type=str, default="pil", help="Output Type")
    parser.add_argument("--max_seq_len", type=int, default=77, help="Max input Prompt length")
    parser.add_argument(
        "--image_size", type=int, nargs="+", default=[1024, 1024], help="Output image size (height, width)"
    )
    parser.add_argument("--save_path", type=str, default="./", help="Output save path")
    parser.add_argument("--use_graph_mode", action="store_true", help="Use graph mode to accelerate.")

    args = parser.parse_args()
    return args


def warmup(pipe, args):
    height, width = args.image_size
    pipe(
        prompt="0",
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        max_sequence_length=args.max_seq_len,
        use_cache_and_gate=args.cache_and_prompt_gate,
        use_todo=args.todo,
        output_type=args.output_type,
    )


def main():
    args = arg_parse()
    if args.use_graph_mode:
        ms.set_context(mode=0, jit_config={"jit_level": "O2"})

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.ckpt,
        custom_pipeline=os.path.join(parent_path, "pipeline_stable_diffusion_3.py"),
        mindspore_dtype=ms.float16,
    )

    if args.use_graph_mode:
        print("Start warmup...")
        warmup(pipe, args)

    print("Start generate >>>", flush=True)
    start = time.time()
    height, width = args.image_size
    imgs = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=width,
        height=height,
        max_sequence_length=args.max_seq_len,
        use_cache_and_gate=args.cache_and_prompt_gate,
        use_todo=args.todo,
        output_type=args.output_type,
    )[0]
    print(f"generate img spend time: {(time.time()-start):.2f}s")

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.output_type == "pil":
        for i, img in enumerate(imgs):
            img.save(os.path.join(args.save_path, f"output_{i}.png"))


if __name__ == "__main__":
    main()
