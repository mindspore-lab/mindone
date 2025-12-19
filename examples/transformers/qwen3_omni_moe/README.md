# Qwen3-Omni

## Introduction
The Qwen3-Omni-MOE model is a unified multiple modalities model proposed in Qwen3-Omni Technical Report from Qwen team, Alibaba Group.

The abstract from the technical report is the following:

*We present Qwen3-Omni, a single multimodal model that, for the first time, maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audio-visual benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To reduce first-packet latency in streaming synthesis, Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license.

# Get Started

## Requirements:
| mindspore | 	ascend driver | firmware       | cann tookit/kernel |
|-----------|----------------|----------------|--------------------|
| 2.7.1     | 24.1.RC3.b080  | 7.5.T11.0.B088 | 8.3.RC1            |

### Installation:
```
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install -e .

pip install transformers==4.57.1

cd examples/transformers/qwen3_omni_moe
```

## **Notice**  
Note that adjusting `min_pixels` and `max_pixels` trades off between memory and accuracy. Please adjust min_pixel and max_pixel of processor if raising OOM error.

## Quick Start

Here is a usage example of Qwen3-Omni-30B-A3B-Instruct. you can use the following command:

```bash
# For Audio Understanding Task:
# If you want only return text, please set `return_audios=False`
msrun --worker_num=2 --local_worker_num=2 --master_port=8118 \
    --log_dir=msrun_log --join=True --cluster_time_out=300 \
    omni_understanding.py
```
Give it a try with various images, audios and prompts🤗🤗.

Omni Understanding Sample script:
`return_audio=False`could be set so that only text result would be returned.

```python
from functools import partial

import numpy as np
import soundfile as sf
from qwen_omni_utils import process_mm_info

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network
from mindone.transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

# set up card communication
dist.init_process_group(backend="hccl")
ms.set_auto_parallel_context(parallel_mode="data_parallel")

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
# MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Thinking"

model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    mindspore_dtype=ms.bfloat16,
    attn_implementation="flash_attention_2",
)

# use zero3 parallel
shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
model = shard_fn(model)

min_pixels = 56 * 56
max_pixels = 14 * 14 * 768
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH, min_pixels=min_pixels, max_pixels=max_pixels)

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"},
            {"type": "audio", "audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cough.wav"},
            {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
        ],
    },
]

# Set whether to use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(
    text=text,
    audio=audios,
    images=images,
    videos=videos,
    return_tensors="np",
    padding=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
)

for key, value in inputs.items():
    if isinstance(value, np.ndarray):
        inputs[key] = ms.tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
    elif inputs[key].dtype != ms.int32:
        inputs[key] = inputs[key].to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(
    **inputs,
    speaker="Ethan",
    thinker_return_dict_in_generate=True,
    use_audio_in_video=USE_AUDIO_IN_VIDEO,
    return_audio=False,
    talker_do_sample=False,
)

text = processor.batch_decode(
    text_ids.sequences[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(text)
if audio is not None:
    sf.write(
        "output.wav",
        audio.reshape(-1).asnumpy(),
        samplerate=24000,
    )

```

Text generation Outputs:
```
['The image displays four luxury cars-a Rolls-Royce, a Mercedes-Benz SUV, a Ferrari convertible and a Porsche 911-while the audio captures a person coughing.']
```

If `return_audio=True` is set, besides that above text generation results, a piece of audio that explains the image and audio would be generated.

## Inference Speed
|          model name	           | mindspore version | precision* | cards | Model part | attention type | 	tokens/s	 |
|:------------------------------:|:-----------------:|:----------:|:-----:|:----------:|:--------------:|:----------:|
|   Qwen3-Omni-30B-A3B-Instruct   |       2.7.1       |    bf16     |   2   |  Thinker   |   flash_attn   |    0.73     |
|   Qwen3-Omni-30B-A3B-Instruct   |       2.7.1       |    bf16     |   2   |   Talker   |   flash_attn   |    0.88    |
