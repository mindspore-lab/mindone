"""
Text-Only Generation for Qwen2.5-Omni
This script demonstrates how to use Qwen2.5-Omni to finish multimodal understainding tasks such as text Q&A, image, mutable video, audiable video understanding.
"""
import numpy as np

import mindspore as ms

from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info


# inference function
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
        if isinstance(value, np.ndarray):
            inputs[key] = ms.Tensor(value)
        elif isinstance(value, list):
            inputs[key] = ms.Tensor(value)
        if inputs[key].dtype == ms.int64:
            inputs[key] = inputs[key].to(ms.int32)
        else:
            inputs[key] = inputs[key].to(model.dtype)

    text_ids = model.generate(**inputs, use_audio_in_video=use_audio_in_video, return_audio=False)
    text_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, text_ids)]
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return text


# Load the model
MODEL_HUB = "Qwen/Qwen2.5-Omni-7B"
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    MODEL_HUB,
    mindspore_dtype=ms.float16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_HUB)
print("Finished loading model and processor.")

print("*" * 100)
print("***** Usage Case 1: text Q&A *****")
medium_path = None
prompt = "Who are you?"
response = inference(medium_path, prompt, medium_type=None, use_audio_in_video=False)
print("***** Response 1 *****")
print(response)

print("*" * 100)
print("***** Usage Case 2: image understanding *****")
medium_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
prompt = "What can you see in this image?"
response = inference(medium_path, prompt, medium_type="image", use_audio_in_video=False)
print("***** Response 2 *****")
print(response)

print("*" * 100)
print("***** Usage Case 3: video vision understanding *****")
medium_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
prompt = "What can you see in this video?"
response = inference(medium_path, prompt, medium_type="video", use_audio_in_video=False)
print("***** Response 3 *****")
print(response)

print("*" * 100)
print("***** Usage Case 4: video vision and audio understanding *****")
medium_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"
prompt = "What can you hear and see in this video?"
response = inference(medium_path, prompt, medium_type="video", use_audio_in_video=True)
print("***** Response 4 *****")
print(response)
