"""
This script performs inference using the Qwen2.5 Omni model with LoRA enhancements.
It supports single image/video VQA inference and batch inference using a dataset.
It by default works for linxy/LaTex_OCR test dataset after finetuning with linxy/LaTex_OCR training dataset.

Usage example:
```
DEVICE_ID=0 python examples/transformers/qwen2_5_omni/infer_qwen2_5_omni_lora.py \
    --model_path Qwen/Qwen2.5-Omni-3B \
    --lora_path ./outputs/lora \
    --image_path test_ocr_image.png \
    --prompt "Please convert the image content into LaTex"
```

OR evaluate the whole test dataset:

```
DEVICE_ID=0 python examples/transformers/qwen2_5_omni/infer_qwen2_5_omni_lora.py \
    --model_path Qwen/Qwen2.5-Omni-3B \
    --lora_path ./outputs/lora \
    --dataset_path linxy/LaTex_OCR
```
"""

from argparse import ArgumentParser

from datasets import load_dataset
from qwwen2_5_omni.utils import process_mm_info

import mindspore as ms

from mindone.diffusers._peft import PeftModel
from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-Omni-3B",
        help="Model ID from huggingface hub or local path",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        required=True,
        help="Path to LoRA checkpoint directory",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Vision prompt for VQA",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please convert the image content into LaTex",
        help="Text prompt for VQA",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="linxy/LaTex_OCR",
        help="Path to the test dataset for VQA",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        mindspore_dtype=ms.float16,
    )
    model.thinker.is_gradient_checkpointing = False
    lora_model = PeftModel.from_pretrained(
        model.thinker,
        args.lora_path,
    )  # get LoRA weights
    lora_model.merge_and_unload()  # merge LoRA weights into the base model
    model.thinker = lora_model.get_base_model()  # replace thinker with LoRA-enhanced model
    model.set_train(False)

    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    def inference(medium_path, prompt, medium_type="image", use_audio_in_video=False):
        sys_prompt = (
            "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
            "auditory and visual inputs, as well as generating text and speech."
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        medium = None
        if medium_type == "video":
            medium = {
                "type": medium_type,
                "video": medium_path,
                "max_pixels": 360 * 420,
            }
        elif medium_type == "image":
            medium = {
                "type": medium_type,
                "image": medium_path,
                "max_pixels": 360 * 420,
            }
        if medium is not None:
            messages[1]["content"].append(medium)

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="np",
            padding=True,
            use_audio_in_video=use_audio_in_video,
        )

        # convert input to Tensor
        for key, value in inputs.items():  # by default input numpy array or list
            inputs[key] = ms.Tensor(value)
            if inputs[key].dtype == ms.int64:
                inputs[key] = inputs[key].to(ms.int32)
            else:
                inputs[key] = inputs[key].to(model.dtype)

        text_ids = model.generate(**inputs, use_audio_in_video=use_audio_in_video, return_audio=False)
        text_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, text_ids)]
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text

    if args.image_path is None:
        with open("latex_ocr_lora_res.txt", "a") as f:
            f.write("*" * 50 + "\n")
            f.write(f"Evaluate finetuned model with LoRA from {args.lora_path}\n")

        dataset = load_dataset(args.dataset_path, name="human_handwrite", split="test")
        prompt = args.prompt
        correct = 0
        for idx, example in enumerate(dataset):
            medium = example["image"].convert("RGB")  # PIL
            answer = example["text"]
            response = inference(medium, prompt, medium_type="image", use_audio_in_video=False)
            print(f"Response #{idx}: {response}\n")

            with open("latex_ocr_lora_res.txt", "a") as f:
                f.write(f"Response #{idx}: {response}\n")
                if response != answer:
                    f.write(f"WRONG! GT #{idx}: {answer}\n")
                else:
                    correct += 1
        with open("latex_ocr_lora_res.txt", "a") as f:
            f.write(f"Accuracy: {correct}/{len(dataset)} = {correct/len(dataset):.2%}\n")
            print(f"Accuracy: {correct}/{len(dataset)} = {correct/len(dataset):.2%}\n")
    else:
        medium_path = args.image_path
        prompt = args.prompt
        response = inference(medium_path, prompt, medium_type="image", use_audio_in_video=False)
        print(f"Response: {response}\n")


if __name__ == "__main__":
    main()
