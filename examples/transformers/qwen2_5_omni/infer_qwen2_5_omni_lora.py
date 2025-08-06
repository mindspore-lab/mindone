from argparse import ArgumentParser

import mindspore as ms
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

def main():
    args = parse_args()
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2",
        mindspore_dtype=ms.float16,
    )
    model.thinker = PeftModel.from_pretrained(
        model.thinker,
        args.lora_path,
    ) # replace thinker with LoRA-enhanced model
    processor = Qwen2VLImageProcessor.from_pretrained(args.model_path)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if args.image_path is None:
        dataset = load_dataset(args.dataset_path, name="human_handwrite", split="test")
        dataset = dataset.select(range(100))
        for idx, example in enumerate(dataset):
            medium = example["image"] # PIL
            prompt = example["text"]
            response = inference(medium, prompt, medium_type="image", use_audio_in_video=False)
            print(f"Response #{idx}: {response}\n")
    else:
        medium_path = args.image_path
        prompt = args.prompt
        response = inference(medium_path, prompt, medium_type="image", use_audio_in_video=False)
        print(f"Response: {response}\n")


if __name__ == "__main__":
    main()
