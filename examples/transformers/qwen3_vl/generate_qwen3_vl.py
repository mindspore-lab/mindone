import argparse

import numpy as np

import mindspore as ms

from mindone.transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def generate(args):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_name,
        mindspore_dtype=ms.bfloat16,
        attn_implementation=args.attn_implementation,
    )

    processor = AutoProcessor.from_pretrained(
        args.model_name,
        use_fast=False,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": args.image,
                },
                {
                    "type": "text",
                    "text": args.prompt,
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="np"
    )

    # convert input to Tensor
    for key, value in inputs.items():
        if isinstance(value, np.ndarray):
            inputs[key] = ms.tensor(value)
        elif isinstance(value, list):
            inputs[key] = ms.Tensor(value)

    generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qwen3VL demo.")

    parser.add_argument("--prompt", type=str, default="Describe this image.")
    parser.add_argument(
        "--image",
        type=str,
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
    )
    parser.add_argument(
        "--model_name", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Path to the pre-trained model."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "eager"],
    )

    # Parse the arguments
    args = parser.parse_args()

    generate(args)
