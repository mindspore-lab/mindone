"""
Voice Chatting with Qwen2.5-Omni
This script demonstrates how to chat with Qwen2.5-Omni by voice input and output.
"""

import numpy as np
import soundfile as sf

import mindspore as ms

from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from .qwen_omni_utils import process_mm_info


# inference function
def inference(medium_path):
    sys_prompt = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving "
        "auditory and visual inputs, as well as generating text and speech."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": medium_path},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
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
    mindspore_dtype=ms.float16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
print("Finished loading model and processor.")

# Voice Chat Example 1
print("*" * 100)
print("***** Voice Chat Example 1 *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/guess_age_gender.wav"
response = inference(audio_path)
print(response[0][0])
# display(Audio(response[1], rate=24000))
sf.write(
    "output_voice_chat_1.wav",
    response[1].reshape(-1).asnumpy(),
    samplerate=24000,
)

# Voice Chat Example 2
print("*" * 100)
print("***** Voice Chat Example 2 *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/translate_to_chinese.wav"
response = inference(audio_path)
print(response[0][0])
sf.write(
    "output_voice_chat_2.wav",
    response[1].reshape(-1).asnumpy(),
    samplerate=24000,
)
