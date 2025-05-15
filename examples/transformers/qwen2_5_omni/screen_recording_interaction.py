"""
Screen Recording Interaction with Qwen2.5-Omni
This script demonstrates how to use Qwen2.5-Omni to get the information and content you want to know by asking questions in real time on the recording screen.
"""
import numpy as np

import mindspore as ms

from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from qwen_omni_utils import process_mm_info


# inference function
def inference(video_path, prompt, sys_prompt):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                },
            ],
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="np",
        padding=True,
        use_audio_in_video=False,
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

    output = model.generate(**inputs, use_audio_in_video=False, return_audio=False)

    text = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text


# Load the model
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    mindspore_dtype=ms.float16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
print("Finished loading model and processor.")


# Understanding
print("*" * 100)
print("***** Understanding *****")
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4"
prompt = "What the browser is used in this video?"
response = inference(video_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print(response[0])

# OCR
print("*" * 100)
print("***** OCR *****")
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4"
prompt = "Who is the authors of this paper?"
response = inference(video_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print(response[0])

# Summarize
print("*" * 100)
print("***** Summarize *****")
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4"
prompt = "Summarize this paper in short."
response = inference(video_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print(response[0])

# Assistant
print("*" * 100)
print("***** Assistant *****")
video_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/screen.mp4"
prompt = "Please trranslate the abstract of paper into Chinese."
response = inference(video_path, prompt=prompt, sys_prompt="You are a helpful assistant.")
print(response[0])
