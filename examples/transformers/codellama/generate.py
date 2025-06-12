import argparse
import ast
import os
import time

from transformers import CodeLlamaTokenizer

import mindspore as ms

from mindone.transformers.models.llama import LlamaForCausalLM


def run_codellama_generate(args):
    print("=====> test_codellama_generate:")
    print("=====> Building model...")

    s_time = time.time()

    tokenizer = CodeLlamaTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, use_flash_attention_2=args.use_fa, mindspore_dtype=ms.float16
    )

    print("=====> Building model done.")

    PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''

    prompt = [
        PROMPT,
    ]
    input_ids = ms.Tensor(tokenizer(prompt, return_tensors="np").input_ids, ms.int32)

    input_kwargs = {}
    if args.use_embed_input:
        input_kwargs["inputs_embeds"] = model.get_input_embeddings()(input_ids)
    else:
        input_kwargs["input_ids"] = input_ids

    generated_ids = model.generate(**input_kwargs, use_cache=args.use_cache, max_new_tokens=128, do_sample=False)
    generated_ids = generated_ids.asnumpy()

    filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True)[0].strip()

    print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
    print("=" * 46 + " Result " + "=" * 46)
    print(PROMPT.replace("<FILL_ME>", filling))
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=0, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--jit_level", type=str, default="O0")
    parser.add_argument("--model_path", type=str, default="meta-llama/CodeLlama-7b-hf")
    parser.add_argument("--use_fa", type=ast.literal_eval, default=True)
    parser.add_argument("--use_cache", type=ast.literal_eval, default=True)
    parser.add_argument("--use_embed_input", type=ast.literal_eval, default=False)
    args, _ = parser.parse_known_args()

    if args.ms_mode == ms.GRAPH_MODE:
        if os.environ.get("MS_DEV_RUNTIME_CONF") is None:
            os.environ["MS_DEV_RUNTIME_CONF"] = "synchronize:True"
            print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")
        else:
            if "synchronize:True" not in os.environ.get("MS_DEV_RUNTIME_CONF"):
                _old = os.environ.get("MS_DEV_RUNTIME_CONF")
                _old.replace("synchronize:False,", "")
                _old.replace(",synchronize:False", "")
                _old.replace("synchronize:False", "")
                _new = "synchronize:True," + _old if len(_old) > 0 else "synchronize:True"
                os.environ["MS_DEV_RUNTIME_CONF"] = _new
                print("WARNING: os environment MS_DEV_RUNTIME_CONF synchronize has not been set, force setting it now.")

        ms.set_context(
            mode=ms.GRAPH_MODE,
            device_target="Ascend",
            jit_config={"jit_level": args.jit_level},
            max_device_memory="59GB",
            deterministic="ON",
        )

    elif args.ms_mode == ms.PYNATIVE_MODE:
        ms.set_context(
            mode=ms.PYNATIVE_MODE,
            device_target="Ascend",
            pynative_synchronize=True,
            max_device_memory="59GB",
            deterministic="ON",
        )

    else:
        raise ValueError

    run_codellama_generate(args)
