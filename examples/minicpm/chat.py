import argparse
import ast
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from transformers import AutoTokenizer

import mindspore as ms

from mindone.transformers.models.minicpm4.modeling_minicpm import MiniCPMForCausalLM


def generate(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = MiniCPMForCausalLM.from_pretrained(
        args.model_name,
        mindspore_dtype=ms.bfloat16,
        _attn_implementation=args.attn_implementation,
    )

    if args.attn_implementation == "paged_attention":
        # infer boost
        from mindspore import JitConfig

        jitconfig = JitConfig(jit_level="O0", infer_boost="on")
        model.set_jit_config(jitconfig)

    responds, history = model.chat(tokenizer, args.prompt, do_sample=args.do_sample, use_cache=args.use_cache)
    print(responds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniCPM4 demo.")

    parser.add_argument("--prompt", type=str, default="Write an article about Artificial Intelligence.")
    parser.add_argument(
        "--model_name", type=str, default="openbmb/MiniCPM4-0.5B", help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["paged_attention", "flash_attention_2", "eager"],
    )
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)
    parser.add_argument("--do_sample", type=ast.literal_eval, default=False)

    # Parse the arguments
    args = parser.parse_args()

    ms.set_context(mode=ms.GRAPH_MODE)
    generate(args)
