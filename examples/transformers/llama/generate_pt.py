import argparse
import ast
import time

import torch
from transformers import AutoTokenizer
from transformers.models.llama import LlamaForCausalLM


def run_llama3_generate_pt(args):
    print("=====> test_llama3_generate:")
    print("=====> Building model...")

    s_time = time.time()

    device = "cuda:0"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(args.model_path, attn_implementation="eager")
    model.to(device)

    print("=====> Building model done.")

    while True:
        prompt = input("Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: ")

        if prompt == "q":
            print("Generate task done, see you next time!")
            break

        prompt = [
            prompt,
        ]
        input_ids = torch.tensor(tokenizer(prompt).input_ids).to(device)

        input_kwargs = {}
        if args.use_embed_input:
            input_kwargs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
        else:
            input_kwargs["input_ids"] = input_ids

        output_ids = model.generate(**input_kwargs, use_cache=args.use_cache, max_new_tokens=24, do_sample=False)
        output_ids = output_ids.detach().cpu().numpy()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
        print("=" * 46 + " Result " + "=" * 46)
        print(outputs)
        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--use_fa", type=ast.literal_eval, default=False)  # unavailable
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)
    parser.add_argument("--use_embed_input", type=ast.literal_eval, default=True)
    args, _ = parser.parse_known_args()

    run_llama3_generate_pt(args)
