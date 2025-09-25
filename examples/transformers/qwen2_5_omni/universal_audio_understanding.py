"""
Universal Audio Understanding for Qwen2.5-Omni
This script demonstrates how to use Qwen2.5-Omni to finish audio tasks such as speech recongnition, speech-to-text translation and audio analysis.
"""

from qwen_omni_utils import process_mm_info

import mindspore as ms

from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor


# inference function
def inference(medium_path, prompt, sys_prompt):
    if sys_prompt is None:
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
                {"type": "audio", "audio": medium_path},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("text:", text)
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
        inputs[key] = ms.Tensor(value)
        if inputs[key].dtype == ms.int64:
            inputs[key] = inputs[key].to(ms.int32)
        else:
            inputs[key] = inputs[key].to(model.dtype)

    text_ids = model.generate(
        **inputs, use_audio_in_video=True, return_audio=False, thinker_max_new_tokens=256, thinker_do_sample=False
    )
    text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
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

# 1. Speeh Recognition
print("*" * 100)
print("***** 1. Speeh Recognition (English) *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"
prompt = "Transcribe the English audio into text without any punctuation marks."
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])

print("*" * 100)
print("***** 1. Speeh Recognition (Chinese) *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/BAC009S0764W0121.wav"
prompt = "请将这段中文语音转换为纯文本，去掉标点符号。"
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])

print("*" * 100)
print("***** 1. Speeh Recognition (Russian) *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/10000611681338527501.wav"
prompt = "Transcribe the Russian audio into text without including any punctuation marks."
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])

print("*" * 100)
print("***** 1. Speeh Recognition (French) *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/7105431834829365765.wav"
prompt = "Transcribe the French audio into text without including any punctuation marks."
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech recognition model.")
print(response[0])

# 2. Speech Translation
print("*" * 100)
print("***** 2. Speech Translation (English to Chinese) *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"
prompt = "Listen to the provided English speech and produce a translation in Chinese text."
response = inference(audio_path, prompt=prompt, sys_prompt="You are a speech translation model.")
print(response[0])

# 3. Vocal Sound Classification
print("*" * 100)
print("***** 3. Vocal Sound Classification *****")
audio_path = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/cough.wav"
prompt = "Classify the given human vocal sound in English."
response = inference(audio_path, prompt=prompt, sys_prompt="You are a vocal sound classification model.")
print(response[0])
