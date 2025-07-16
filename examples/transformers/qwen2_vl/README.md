# Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
[Paper](https://arxiv.org/abs/2409.12191) | [HF Model Card](https://huggingface.co/collections/Qwen/qwen2-vl-66cee7455501d7126940800d)

# Introduction
> **LRDR;** Qwen2-VL is a multimodal vision-language model series based on Qwen2, which supports inputs of text, arbitrary-resolution image, long video (20min+) and multiple languages.

> **Abstract:** The Qwen2-VL Series, an advanced upgrade of the previous Qwen-VL models that redefines the conventional predetermined-resolution approach in visual processing. Qwen2-VL introduces the Naive Dynamic Resolution mechanism, which enables the model to dynamically process images of varying resolutions into different numbers of visual tokens. This approach allows the model to generate more efficient and accurate visual representations, closely aligning with human perceptual processes. The model also integrates Multimodal Rotary Position Embedding (M-RoPE), facilitating the effective fusion of positional information across text, images, and videos. We employ a unified paradigm for processing both images and videos, enhancing the model's visual perception capabilities. To explore the potential of large multimodal models, Qwen2-VL investigates the scaling laws for large vision-language models (LVLMs). By scaling both the model size-with versions at 2B, 8B, and 72B parameters-and the amount of training data, the Qwen2-VL Series achieves highly competitive performance. Notably, the Qwen2-VL-72B model achieves results comparable to leading models such as GPT-4o and Claude3.5-Sonnet across various multimodal benchmarks, outperforming other generalist models.

# Get Started

## Requirements:
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.5.0 | 24.1RC3 | 7.3.0.1.231 | 8.0.RC3.beta1|
|2.4.1 | 24.1RC3 | 7.3.0.1.231 | 8.0.RC3.beta1|

### Installation:
```
cd examples/transformers/qwen2-vl
pip install requirements.txt
```

Pretrained weights from huggingface hub: [Qwen2-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

## Quick Start


`test_vqa.py` and `video_understanding.py` provides examples of image and video VQA. Here is an usage example of image understanding:

```python
from transformers import AutoProcessor
from mindone.transformers import Qwen2VLForConditionalGeneration
from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info
from mindspore import Tensor
import numpy as np

model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen2/Qwen2-VL-7B-Instruct", mindspore_dtype=ms.float32)
processor = AutoProcessor.from_pretrained("Qwen2/Qwen2-VL-7B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "demo.jpeg",  # REPLACE with your own image
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="np",
)
# convert input to Tensor
for key, value in inputs.items():
    inputs[key] = ms.Tensor(value)
    if inputs[key].dtype == ms.int64:
        inputs[key] = inputs[key].to(ms.int32)
generated_ids = model.generate(**inputs, max_new_tokens=128)
output_text = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)[0]
print(output_text)
```

# Performance
## Inference

### Inference Speed
Experiments are tested on ascend 910* pynative mode.

Input an image or a list of video frames, and a text prompt, output textual response.


- mindspore 2.5.0

|model name	| precision* | cards	| batch size| resolution | flash attn |	s/step	| step | response/s | weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 1372x2044 (image) | OFF | 0.40 | 128 | 0.02 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 12x308x476(video) | OFF | 0.36 | 128 | 0.02 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 1372x2044 (image) | ON  | 0.34 | 127 | 0.02 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 12x308x476(video) | ON  | 0.30 | 128 | 0.03 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct | bf16 | 1 | 1 | 1372x2044 (image) | ON  | 0.32| 121 | 0.03 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct | bf16 | 1 | 1 | 12x308x476(video) | ON  | 0.30| 128| 0.03 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

- mindspore 2.4.1

|model name	| precision* | cards	| batch size| resolution | flash attn |	s/step	| step | response/s | weight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 1372x2044 (image) | OFF | 0.80 | 128 | 0.01 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 12x308x476(video) | OFF | 0.67 | 97  | 0.02 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 1372x2044 (image) | ON  | 0.35 | 127 | 0.02 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct |  fp16 | 1 | 1 | 12x308x476(video) | ON  | 0.25 | 128 | 0.03 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct | bf16 | 1 | 1 | 1372x2044 (image) | ON  | 0.35 | 121 | 0.02 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)
| Qwen2-VL-7B-Instruct | bf16 | 1 | 1 | 12x308x476(video) | ON  | 0.29 | 128 | 0.03 | [weight](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)

*note: use mixed precision, `Conv3d` uses bf16, `flash attention` fp16, `LayerNorm` fp32.

### Inference Results

Input an image or a list of video frames, and a text prompt (English, Chinese or other lanugage), output textual response.

#### Image VQA
Input:

*resolution: 1372x2044
<br>
<img src="https://github.com/user-attachments/assets/0520288b-2e21-4b10-b506-1c8b54a1737f" width="512px">

text prompt: `Describe this image.`
<br>
Response: 'This image depicts a serene beach scene at sunset. A woman and her dog are sitting on the sand, enjoying each other's company. The woman is wearing a plaid shirt and dark pants, and she is sitting cross-legged with her dog. The dog which appears to be a large breed, is wearing a harness and is giving a high-five to the woman. The beach is relatively empty, with gentle waves in the background. The lighting is warm and golden, indicating that the photo was taken during the golden hour, just before sunset. The overall atmosphere of the image is peaceful and joyful.'

text prompt: `请描述该图片。`<br>
Response: "图中是一个女人和一只狗在沙滩上玩耍。女人穿着格子衬衫，坐在沙滩上，她的狗是一只拉布拉多犬，穿着狗绳，伸出前爪和女人击掌。女人和狗都面带微笑，看起来非常开心。背景是广阔的海洋和天空，阳光洒在沙滩上，整个场景显得非常温馨和愉快。"

#### Video VQA
Input:
*resolution: 12x308x476
<br>
<a href="https://qianwen-res.oss-accelerate-overseas.aliyuncs.com/Qwen2-VL/Operate%20a%20Mobile%20Phone.mp4"><img src="https://github.com/user-attachments/assets/0b4afe5d-021d-45be-8877-75b9a39b1ccb" width="512px"></a>

text prompt: `Describe this video.`<br>
Response: 'The video shows a computer screen with a web browser open to a search engine. The search bar at the top of the screen displays the text "What's a good restaurant in San Diego?" The search results are displayed below the search bar, with the top result being a map of San Diego with several restaurant locations marked. The screen also shows a list of search suggestions, including "What's a good restaurant in San Diego?" and "What's a good restaurant in San Diego for families?" The video ends with the search results still displayed on the screen.'

text prompt: `请描述该视频。`<br>
Response: "视频中显示了一个电脑屏幕，上面有两个窗口。左边的窗口显示了一个网页，上面有一个搜索框和一些搜索建议。右边的窗口显示了一个命令行界面，显示了一些文本。"
