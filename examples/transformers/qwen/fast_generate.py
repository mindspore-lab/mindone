import argparse
import ast
import os
import time

from transformers import AutoTokenizer

import mindspore

from mindone.transformers.models.qwen2_fast import FastInferQwen2ForCausalLM


def run_qwen2_generate(args):
    print("=====> test_qwen2_generate:")
    print("=====> Building model...")

    s_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = FastInferQwen2ForCausalLM.from_pretrained(
        args.model_path,
    )

    print("=====> Building model done.")

    is_first = True
    while True:
        if args.prompt is not None and is_first:
            prompt = args.prompt
        else:
            prompt = input("Enter your prompt [e.g. `What's your name?`] or enter [`q`] to exit: ")
        is_first = False

        if prompt == "q":
            print("Generate task done, see you next time!")
            break

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        # for batch inputs, pad to longest
        input_ids = mindspore.Tensor(tokenizer([text],
                                               return_tensors="np",
                                               add_special_tokens=True,
                                               padding=True).input_ids, 
                                    mindspore.int32)

        model_inputs = {}
        model_inputs["input_ids"] = input_ids

        output_ids = model.generate(
            **model_inputs,
            use_cache=True,
            # max_new_tokens=10,
            max_length=1000,
            do_sample=False,
        )
        if isinstance(output_ids, mindspore.Tensor):
            output_ids = output_ids.asnumpy()

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print(f"=====> input prompt: {prompt}, time cost: {time.time() - s_time:.2f}s")
        print("=" * 46 + " Result " + "=" * 46)
        print(outputs)
        print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument("--ms_mode", type=int, default=0, help="0 is Graph, 1 is Pynative")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--prompt", type=str, default=None)
    args, _ = parser.parse_known_args()

    # set context
    mindspore.set_context(
        mode=args.ms_mode,
        device_target="Ascend",
        ascend_config={"precision_mode": "must_keep_origin_dtype"},
        max_call_depth=10000,
        max_device_memory="59GB",
        jit_config={"jit_level": "O0", "infer_boost": "on"}
    )
    # mindspore.set_context(
    #     pynative_synchronize=True
    # )

    # set env
    os.environ["SHLVL"] = "2"
    os.environ["CRC32C_SW_MODE"] = "auto"
    os.environ["CUSTOM_MATMUL_SHUFFLE"] = "on"
    os.environ["LCCL_DETERMINISTIC"] = "0"
    os.environ["MS_ENABLE_GRACEFUL_EXIT"] = "0"
    os.environ["MS_ALLOC_CONF"] = "enable_vmm:False"
    os.environ["CPU_AFFINITY"] = "True"
    os.environ["MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST"] = "PagedAttention"
    os.environ["RUN_MODE"] = "predict"

    run_qwen2_generate(args)
