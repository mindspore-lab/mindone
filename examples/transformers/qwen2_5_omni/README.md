# Qwen2.5-Omni

## Overview

Qwen2.5-Omni [Qwen2.5-Omni](https://qwenlm.github.io/blog/qwen2.5-omni/) is an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner.

The abstract from the [Qwen2.5-Omni Technical Report](https://arxiv.org/abs/2503.20215) is the following:

> We present Qwen2.5-Omni, an end-to-end multimodal model designed to perceive diverse modalities, including text, images, audio, and video, while simultaneously generating text and natural speech responses in a streaming manner. To enable the streaming of multimodal information inputs, both audio and visual encoders utilize a block-wise processing approach. This strategy effectively decouples the handling of long sequences of multimodal data, assigning the perceptual responsibilities to the multimodal encoder and entrusting the modeling of extended sequences to a large language model. Such a division of labor enhances the fusion of different modalities via the shared attention mechanism. To synchronize the timestamps of video inputs with audio, we organized the audio and video sequentially in an interleaved manner and propose a novel position embedding approach, named TMRoPE (Time-aligned Multimodal RoPE). To concurrently generate text and speech while avoiding interference between the two modalities, we propose Thinker-Talker architecture. In this framework, Thinker functions as a large language model tasked with text generation, while Talker is a dual-track autoregressive model that directly utilizes the hidden representations from the Thinker to produce audio tokens as output. Both the Thinker and Talker models are designed to be trained and inferred in an end-to-end manner. For decoding audio tokens in a streaming manner, we introduce a sliding-window DiT that restricts the receptive field, aiming to reduce the initial package delay. Qwen2.5-Omni outperforms the similarly sized Qwen2-VL and Qwen2-Audio in both image and audio capabilities. Furthermore, Qwen2.5-Omni achieves state-of-the-art performance on multimodal benchmarks like Omni-Bench. Notably, Qwen2.5-Omni is the first open-source model to achieve a level of performance in end-to-end speech instruction following that is comparable to its capabilities with text inputs, as evidenced by benchmarks such as MMLU and GSM8K. As for speech generation, Qwen2.5-Omni’s streaming Talker outperform most existing streaming and non-streaming alternatives in robustness and naturalness.


## Get Started
## Requirements:
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.6.0 | 24.1.RC3 | 7.5.T11.0 | 8.1.RC1|

### Installation
```
# with installed mindone
cd examples/qwen2_5_omni
pip install -r requirements.txt
```
### Model Checkpoints

`Qwen2.5-Omni-7B`, `Qwen2.5-Omni-3B` chekpoints can be found on the [Huggingface Hub](https://huggingface.co/collections/Qwen/qwen25-omni-67de1e5f0f9464dc6314b36e).

The speakers checkpoint need to be converted before use:
```python
python mindone\transformers\models\qwen2_5_omni\convert_spk_dict_pt2np.py \
    --spk_path "Qwen/Qwen2.5-Omni-7B/spk_dict.pt" \
    --zip_spk_path"Qwen/Qwen2.5-Omni-7B/spk_dict.zip"
```
### Inference Usage Examples


Here are some usage chat examples and scripts with `mindone.transformers`:
|Example|	Description	|
|---|---|
|[Text-Only Generation](text_only_generation.py) | Q&A with Qwen2.5-Omni by text, image, video input and text-only output.
|[Universal Audio Understanding](universal_audio_understanding.py)|	Speech recongnition, speech-to-text translation and audio analysis.	|
|[Voice Chatting](voice_chatting.py)	| Chatting with Qwen2.5-Omni by voice input and output.	|
|[Video Information Extracting](video_information_extracting.py)	| Obtaining information from the video stream. |
[Multi Round Omni Chatting](multi_round_omni_chatting.py)	|Conducted multiple rounds of audio and video dialogues with Qwen2.5-Omni to provide the most comprehensive ability demonstration.|
|[Screen Recording Interaction](screen_recording_interaction.py)	| Get the information and content you want to know by asking questions in real time on the recording screen.	|
|[Omni Chatting for Music](omni_chatting_for_music.py)	| Chat with Qwen2.5-Omni about music content in a audio and video stream.|
| [Omni Chatting for Math](omni_chatting_for_math.py)	|Chat with Qwen2.5-Omni about math content in a audio and video stream.|
|

### Single Media inference

The model can accept text, images, audio and videos as input. Here's an example code for inference.

```python
import soundfile as sf
import numpy as np
import mindspore as ms
from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor
from mindone.transformers.models.qwen2_5_omni.qwen_omni_utils import process_mm_info

# Load model
model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text",
             "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "text", "text": "What cant you hear and see in this video?"},
        ],
    },
]

# Preparation for inference
USE_AUDIO_IN_VIDEO = True  # set use audio in video

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="np", padding=True,
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)

# convert input to Tensor
for key, value in inputs.items():
    if isinstance(value, np.ndarray):
        inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    else:
        inputs[key] = inputs[key].to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)
text_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, text_ids)
]
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).asnumpy(),
    samplerate=24000,
)
```

### Text-only generation

To generate only text output and save compute by not loading the audio generation model, we can set `return_audio=False` when running the model. See more example usages in `text_only_generation.py`.

```python
import mindspore as ms
from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    mindspore_dtype=ms.float16,
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text",
             "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/path/to/video.mp4"
                         "max_pixels": 360 * 420,
            },
            {"type": "text", "text": "What cant you hear and see in this video?"},
        ],
    },
]

# Preparation for inference
USE_AUDIO_IN_VIDEO = True  # set use audio in video

text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="np", padding=True,
                   use_audio_in_video=USE_AUDIO_IN_VIDEO)

# convert input to Tensor
for key, value in inputs.items():
    inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    else:
        inputs[key] = inputs[key].to(model.dtype)

# Inference: Generation of the output text and audio
# set not to return audio
text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
text_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, text_ids)
]
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
```

<!-- may OOM
### Batch Mixed Media Inference

The model can batch inputs composed of mixed samples of various types such as text, images, audio and videos as input when `return_audio=False` is set. Here is an example.

```python
import soundfile as sf
import mindspore as ms
from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

model = Qwen2_5OmniForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-Omni-7B")
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# Conversation with video only
conversation1 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "/path/to/video.mp4"},
        ]
    }
]

# Conversation with audio only
conversation2 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "audio", "audio": "/path/to/audio.wav"},
        ]
    }
]

# Conversation with pure text
conversation3 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "who are you?"}],
    }
]


# Conversation with mixed media
conversation4 = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "/path/to/image.jpg"},
            {"type": "video", "video": "/path/to/video.mp4"},
            {"type": "audio", "audio": "/path/to/audio.wav"},
            {"type": "text", "text": "What are the elements can you see and hear in these medias?"},
        ],
    }
]

conversations = [conversation1, conversation2, conversation3, conversation4]

inputs = processor.apply_chat_template(
    conversations,
    load_audio_from_video=True,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="np",
    video_fps=1,

    # kwargs to be passed to `Qwen2-5-OmniProcessor`
    padding=True,
    use_audio_in_video=True,
)
# convert input to Tensor
for key, value in inputs.items():
    if isinstance(value, np.ndarray):
        inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    else:
        inputs[key] = inputs[key].to(model.dtype)

text_ids = model.generate(**inputs, use_audio_in_video=True)
text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

print(text)
```
 -->

### Usage Tips

#### Image Resolution trade-off

The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs.

```python
min_pixels = 128*28*28
max_pixels = 768*28*28
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B", min_pixels=min_pixels, max_pixels=max_pixels)
```

#### Prompt for audio output
If users need audio output, the system prompt must be set as `"You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."`, otherwise the audio output may not work as expected.
```
{
    "role": "system",
    "content": [{"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}],
}
```

#### Use audio output or not

The model supports both text and audio outputs, if users do not need audio outputs, they can set `enable_audio_output` in the `from_pretrained` function. This option will save about `~2GB` of GPU memory but the `return_audio` option for `generate` function will only allow to be set at `False`.
```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    enable_audio_output=False,
)
```

In order to obtain a flexible experience, we recommend that users set `enable_audio_output` at `True` when initializing the model through `from_pretrained` function, and then decide whether to return audio when `generate` function is called. When `return_audio` is set to `False`, the model will only return text outputs to get text responses faster.

```python
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    enable_audio_output=True,
)
...
text_ids = model.generate(**inputs, return_audio=False)
```

#### Change voice type of output audio
Qwen2.5-Omni supports the ability to change the voice of the output audio. Users can use the `spk` parameter of `generate` function to specify the voice type. The `"Qwen/Qwen2.5-Omni-7B"` checkpoint support two voice types: `Chelsie` and `Ethan`, while `Chelsie` is a female voice and `Ethan` is a male voice. By defalut, if `spk` is not specified, the default voice type is `Chelsie`.

```python
text_ids, audio = model.generate(**inputs, spk="Chelsie")
```

```python
text_ids, audio = model.generate(**inputs, spk="Ethan")
```

#### Flash-Attention 2 to speed up generation

You should have Ascend hardware that is compatible with FlashAttention 2. Read more about it in the official documentation of the [flash attention repository](https://www.mindspore.cn/docs/en/master/api_python/ops/mindspore.ops.flash_attention_score.html).

To load and run a model using FlashAttention-2, add `attn_implementation="flash_attention_2"` when loading the model:

```python
import mindspore as ms
from mindone.transformers import Qwen2_5OmniForConditionalGeneration

model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-7B",
    mindspore_dtype=ms.bfloat16,
    attn_implementation="flash_attention_2",
)
```

### Finetuning

There are example scripts `finetune_lora_with_mindspore_trainer.py` and `finetune_lora_in_native_mindspore.py` for finetuning the model for OCR task with LoRA.
Here's an example code for finetuning:
```
DEVICE_ID=0 python finetune_lora_with_mindspore_trainer.py \
    --model_path Qwen/Qwen2.5-Omni-3B \
    --lora_rank 8 \
    --lora_alpha 16 \
    --dataset_path linxy/LaTex_OCR \
    --output_dir ./outputs/lora \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --save_total_limit 1
```
or
```
DEVICE_ID=0 python finetune_lora_in_native_mindspore.py \
    --model_path Qwen/Qwen2.5-Omni-3B \
    --dataset_path linxy/LaTex_OCR \
    --enable_flash_attention \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir ./outputs/lora \
    --num_train_epochs 1
```


# Peformance

## Inference
Experiments are tested on ascend 910* with mindspore 2.6.0 pynative mode.

|model| precision | task | resolution| fa | tokens/s | steps|
|---|---|---|---|---|---|---|
|Qwen2.5-Omni-7B| fp32 | pure text Q&A | N.A. | OFF | 1.88 | 22 |
|Qwen2.5-Omni-7B| fp32 | video VQA w/ audio| 20x280x504 | OFF | 2.18 | 48 |
|Qwen2.5-Omni-7B| bf16 | pure text Q&A | N.A. | OFF | 1.95 | 22 |
|Qwen2.5-Omni-7B| bf16 | video VQA w/ audio| 20x280x504 | OFF | 1.78 | 48 |
|Qwen2.5-Omni-7B| fp16 | pure text Q&A | N.A. | OFF | 1.87 | 22 |
|Qwen2.5-Omni-7B| fp16 | video VQA w/ audio| 20x280x504 | OFF | 1.95 | 48 |
|Qwen2.5-Omni-7B| bf16 | pure text Q&A | N.A. | ON | 4.77 | 22 |
|Qwen2.5-Omni-7B| bf16 | video VQA w/ audio| 20x280x504 | ON | 5.93 | 48 |
|Qwen2.5-Omni-7B| fp16 | pure text Q&A | N.A. | ON | 5.13 | 22 |
|Qwen2.5-Omni-7B| fp16 | video VQA w/ audio| 20x280x504 | ON | 4.43 | 48 |

*note：apply mixed precision, `AvgPool1d` uses fp32.

## Finetuning
Experiments are tested on ascend 910* with mindspore 2.6.0 pynative mode.

|model| precision |amp\*| task | resolution| fa |card| batch size| max token| recompute |  s/step |
|---|---|---|---|---|---|---|---|---|---|---|
|Qwen2.5-Omni-3B| fp32 |0| image VQA | 128x512 | ON | 1 | 1 | 4096 |OFF | 6.61 |
|Qwen2.5-Omni-3B| fp32 |0| image VQA | 128x512 | ON | 2 | 1 | 4096 |OFF | 5.32 |

*note：apply mixed precision, `AvgPool1d` uses fp32.
