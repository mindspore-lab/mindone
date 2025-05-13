"""
Multi Round Omni Chatting with Qwen2.5-Omni
This script demonstrates how to use Qwen2.5-Omni for multiple rounds of audio and video dialogues.
"""

import numpy as np
import soundfile as sf

import mindspore as ms

from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor
from .qwen_omni_utils import process_mm_info


def inference(conversations):
    text = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(conversations, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="np",
        padding=True,
        use_audio_in_video=True,
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

    text_ids, audio = model.generate(**inputs, use_audio_in_video=True, return_audio=True)
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text, audio


# Load the model
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    mindspore_dtype=ms.bfloat16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
print("Finished loading model and processor.")

sys_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."

# Omni Chatting Round 1
print("*" * 100)
print("***** Omni Chatting Round 1 *****")
conversations = [
    {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
]
video_path_round_1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw1.mp4"
conversations.append(
    {"role": "user", "content": [{"type": "video", "video": video_path_round_1, "max_pixels": 360 * 420}]}
)
response, audio = inference(conversations)
print(response[0])
sf.write(
    "output_multi-round_omni_chat_1.wav",
    audio.reshape(-1).asnumpy(),
    samplerate=24000,
)


# Omni Chatting Round 2
print("*" * 100)
print("***** Omni Chatting Round 2 *****")
conversations.append({"role": "assistant", "content": [{"type": "text", "text": response[0].split("\n")[-1]}]})
video_path_round_2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw2.mp4"
conversations.append(
    {"role": "user", "content": [{"type": "video", "video": video_path_round_2, "max_pixels": 360 * 420}]}
)
response, audio = inference(conversations)
print(response[0])
sf.write(
    "output_multi-round_omni_chat_2.wav",
    audio.reshape(-1).asnumpy(),
    samplerate=24000,
)

# Omni Chatting Round 3
print("*" * 100)
print("***** Omni Chatting Round 3 *****")
conversations.append({"role": "assistant", "content": [{"type": "text", "text": response[0].split("\n")[-1]}]})
video_path_round_3 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw3.mp4"
conversations.append(
    {"role": "user", "content": [{"type": "video", "video": video_path_round_3, "max_pixels": 360 * 420}]}
)
response, audio = inference(conversations)
print(response[0])
sf.write(
    "output_multi-round_omni_chat_3.wav",
    audio.reshape(-1).asnumpy(),
    samplerate=24000,
)
