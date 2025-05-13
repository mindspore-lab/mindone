import argparse
import mindspore as ms
from mindspore import JitConfig
from transformers import AutoTokenizer
from mindone.transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM


def generate(args):

    # load model
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_name,
        mindspore_dtype=ms.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    jitconfig = JitConfig(jit_level="O0", infer_boost="on")
    model.set_jit_config(jitconfig)
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # info
    print("*" * 100)
    print(f"Using {config._attn_implementation}, use_cache {config.use_cache},"
        f"dtype {config.mindspore_dtype}, layer {config.num_hidden_layers}")
    print("Successfully loaded Qwen3ForCausalLM")


    # prepare inputs
    input_ids = ms.Tensor(tokenizer([args.prompt], return_tensors="np").input_ids, ms.int32)
    model_inputs = {}
    model_inputs["input_ids"] = input_ids

    # generate
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=50,
        do_sample=False,
        use_cache=False,
    )

    generated_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(input_ids, generated_ids)]
    outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(outputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="qwen3 demo.")

    parser.add_argument("--prompt", type=str, default="the secret to baking a really good cake is", required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B-Base", help="Path to the pre-trained model.")
    parser.add_argument("--attn_implementation", type=str, default="paged_attention", choices=["paged_attention", "flash_attentions_2", "eager"])

    # Parse the arguments
    args = parser.parse_args()

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    generate(args)