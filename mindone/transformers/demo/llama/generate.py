import ast
import argparse
import time
import mindspore as ms

from transformers import AutoTokenizer

from mindone.transformers.models.llama import LlamaForCausalLM


def run_llama3_generate(args):

    print(f"=====> test_llama3_generate:")
    print(f"=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(args.model_path, use_flash_attention_2=args.use_fa)

    print(f"=====> Building model done.")

    while True:
        prompt = input("Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: ")

        if prompt == "q":
            print("generate task done, see you next time.")
            break

        prompt = [prompt,]
        input_ids = ms.Tensor(tokenizer(prompt).input_ids, ms.int32)

        input_kwargs = {}
        if args.use_embed_input:
            input_kwargs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
        else:
            input_kwargs["input_ids"] = input_ids

        output_ids = model.generate(
            **input_kwargs,
            use_cache=args.use_cache,
            max_new_tokens=20
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
        print("=" * 46 + " Result " + "=" * 46)
        print(outputs)
        print("=" * 100)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=0, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--pynative_synchronize", type=ast.literal_eval, default=True)
    parser.add_argument("--jit_level", type=str, default="O0")
    parser.add_argument("--model_path", type=str, default="../hf_configs/meta-llama/Meta-Llama-3-8B")
    parser.add_argument("--use_fa", type=ast.literal_eval, default=True)
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)
    parser.add_argument("--use_embed_input", type=ast.literal_eval, default=True)
    args, _ = parser.parse_known_args()

    ms.set_context(
        mode=args.ms_mode,
        jit_config={"jit_level": args.jit_level},
        pynative_synchronize=args.pynative_synchronize,
        deterministic="ON"
    )

    run_llama3_generate(args)
