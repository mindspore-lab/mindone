import ast
import argparse
import os

from transformers import AutoTokenizer

import mindspore

from mindone.transformers import pipeline, AutoModelForCausalLM


def generate(args):

    pipe = pipeline("text-generation", model=args.model_name, mindspore_dtype=mindspore.bfloat16, model_kwargs={'attn_implementation': 'eager'})

    messages = [
        {"role": "user", "content": args.prompt},
    ]

    outputs = pipe(
        messages,
        max_new_tokens=256,
    )
    print(outputs[0]["generated_text"][-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qwen3 demo.")
    parser.add_argument("--prompt", type=str, default="Explain quantum mechanics clearly and concisely.")
    parser.add_argument("--model_name", type=str, default="openai/gpt-oss-20b", help="Path to the pre-trained model.")
    parser.add_argument("--enable_synchronize", type=ast.literal_eval, default=True)
    parser.add_argument("--enable_offload", type=ast.literal_eval, default=False)
    parser.add_argument("--dry_run", type=str, default="")
    args = parser.parse_args()

    
    if args.dry_run:
        os.environ["MS_SIMULATION_LEVEL"] = args.dry_run
        if int(args.dry_run) >= 1:
            os.environ["GLOG_v"] = "1"
        if int(args.dry_run) >= 2:
            os.environ["MS_ALLOC_CONF"] = "memory_tracker:True"
    if args.enable_synchronize:
        mindspore.launch_blocking()
    if args.enable_offload:
        mindspore.set_offload_context(offload_config={"offload_param":"cpu", "offload_path": "./offload", "offload_cpu_size":"512GB", "hbm_ratio":0.9})

    generate(args)
