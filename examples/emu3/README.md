# Emu3: Next-Token Prediction is All You Need

<!-- [Emu3 Team, BAAI](https://www.baai.ac.cn/english.html) -->

 [Paper](https://arxiv.org/pdf/2409.18869) | [🤗HF Models](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f) | [Modelscope](https://modelscope.cn/collections/Emu3-9eacc8668b1043) |

<!-- <div align='center'>
<img src="./assets/arch.png" class="interpolation-image" alt="arch." height="80%" width="70%" />
</div> -->

## Introduction
**Emu3** is a new suite of state-of-the-art multimodal models trained solely with **<i>next-token prediction</i>**. By tokenizing images, text, and videos into a discrete space, a single transformer is trained from scratch on a mixture of multimodal sequences.

As a multimodal LLM, Emu3 uses vector quantization to tokenize images into discrete tokens. Discretized image tokens are later fused with text token ids for image and text generation. The model can additionally generate images by predicting image token ids.

<!-- ### Emu3 excels in both generation and perception -->
**Emu3** outperforms several well-established task-specific models in both ***generation*** and ***perception*** tasks, surpassing flagship open models such as SDXL, LLaVA-1.6 and OpenSora-1.2, while eliminating the need for diffusion or compositional architectures.

<!-- <div align='center'>
<img src="./assets/comparison.png" class="interpolation-image" alt="comparison." height="80%" width="80%" />
</div> -->

### Highlights

- **Emu3** is capable of generating high-quality images following the text input, by simply predicting the next vision token. The model naturally supports flexible resolutions and styles.
- **Emu3** shows strong vision-language understanding capabilities to see the physical world and provides coherent text responses. Notably, this capability is achieved without depending on a CLIP and a pretrained LLM.
- **Emu3** simply generates a video causally by predicting the next token in a video sequence, unlike the video diffusion model as in Sora. With a video in context, Emu3 can also naturally extend the video and predict what will happen next.

### Features

- Model weights of Vision Tokenizer, Emu3-Stage1, Emu3-Chat and Emu3-Gen.
- Inference code.
- Training scripts for sft.

## Get Started
### Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.4.1 | 24.1RC2 | 7.3.0.1.231 | 8.0.RC3.beta1|

### Dependencies

Enter this directory and install required packages:

```shell
cd examples/emu3
pip install -r requirements.txt
```

### Model Weights

<details>

| Model name | HF Weight | Modelscope | Wisemodel |
| --- | --- | --- | --- |
| **Emu3-Stage1**          | [🤗 HF link](https://huggingface.co/BAAI/Emu3-Stage1)          | [Modelscope link](https://modelscope.cn/models/BAAI/Emu3-Stage1)          |  |
| **Emu3-Chat**            | [🤗 HF link](https://huggingface.co/BAAI/Emu3-Chat)            | [Modelscope link](https://modelscope.cn/models/BAAI/Emu3-Chat)            | [Wisemodel link](https://wisemodel.cn/models/BAAI/Emu3-Chat)            |
| **Emu3-Gen**             | [🤗 HF link](https://huggingface.co/BAAI/Emu3-Gen)             | [Modelscope link](https://modelscope.cn/models/BAAI/Emu3-Gen)             | [Wisemodel link](https://wisemodel.cn/models/BAAI/Emu3-Gen)             |
| **Emu3-VisionTokenizer** | [🤗 HF link](https://huggingface.co/BAAI/Emu3-VisionTokenizer) | [Modelscope link](https://modelscope.cn/models/BAAI/Emu3-VisionTokenizer) | [Wisemodel link](https://wisemodel.cn/models/BAAI/Emu3-VisionTokenizer) |

</details>

#### Weight conversion:

For model **Emu3-VisionTokenizer**, there are some incompatible network layer variable names that cannot be automatically converted, we need to convert some weight names in advanved before loading the pre-trained weights:
```
python python convert_weights.py --safetensor_path ORIGINAL_MODEL.safetensors --target_safetensor_path model.safetensors
```

## Inference

#### Run Emu3-Gen/Stage1 for image generation
An inference script is provided in `scripts/infer_img_gen.sh`.<br>
An example to generate image is as follows:
<details>

```python
from emu3.mllm import Emu3ForCausalLM, Emu3Processor, Emu3Tokenizer
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image
from transformers.generation.configuration_utils import GenerationConfig
import mindspore as ms
from mindspore import Tensor, nn
from mindone.transformers.generation.logits_process import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from mindone.utils.amp import auto_mixed_precision

# prepare model and processor
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"
EMU_DTYPE = ms.bfloat16
VQ_DTYPE = ms.bfloat16

# prepare model and processor
model = Emu3ForCausalLM.from_pretrained(
    EMU_HUB,
    mindspore_dtype=EMU_DTYPE,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
).set_train(False)
tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, padding_side="left")
image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
image_tokenizer = Emu3VisionVQModel.from_pretrained(
    VQ_HUB,
    use_safetensors=True,
    mindspore_dtype=VQ_DTYPE
).set_train(False)
image_tokenizer = auto_mixed_precision(
    image_tokenizer, amp_level="O2", dtype=VQ_DTYPE, custom_fp32_cells=[nn.BatchNorm3d]
)
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = "a portrait of young girl."
prompt += POSITIVE_PROMPT

kwargs = dict(
    mode='G',
    ratio="1:1",
    image_area=model.config.image_area,
    return_tensors="np",
    padding="longest",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=40960,
    do_sample=True,
    top_k=2048,
)

h = pos_inputs.image_size[:, 0]
w = pos_inputs.image_size[:, 1]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList([
    UnbatchedClassifierFreeGuidanceLogitsProcessor(
        classifier_free_guidance,
        model,
        unconditional_ids=neg_inputs.input_ids.to("cuda:0"),
    ),
    PrefixConstrainedLogitsProcessor(
        constrained_fn ,
        num_beams=1,
    ),
])

# generate
outputs = model.generate(
    Tensor(pos_inputs.input_ids, dtype=ms.int32),
    GENERATION_CONFIG,
    logits_processor=logits_processor,
    attention_mask=Tensor(pos_inputs.attention_mask),
)
out = outputs[0]
if not model.config.img_token_id in out:  # img_token_id was deleted in generate() output
    out = ops.cat(Tensor([model.config.img_token_id], out))
mm_list = processor.decode(out)
for idx, im in enumerate(mm_list):
    if not isinstance(im, Image.Image):
        continue
    im.save(f"result_{idx}.png")
```
</details>

#### Run Emu3-Chat/Stage1 for vision-language understanding
An inference script is provided in `scripts/infer_vqa.sh`.<br>
An example is as follows:
<details>

```python
from emu3.mllm import Emu3ForCausalLM, Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image
from transformers.generation.configuration_utils import GenerationConfig
import mindspore as ms
from mindspore import Tensor, nn
from mindone.utils.amp import auto_mixed_precision

# model path
EMU_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"
EMU_DTYPE = ms.bfloat16
VQ_DTYPE = ms.bfloat16

# prepare model and processor
model = Emu3ForCausalLM.from_pretrained(
    EMU_HUB,
    mindspore_dtype=EMU_DTYPE,
    use_safetensors=True,
    attn_implementation="flash_attention_2",  # optional: "eager"
).set_train(False)
tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, padding_side="left")
image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
image_tokenizer = Emu3VisionVQModel.from_pretrained(
    VQ_HUB,
    use_safetensors=True,
    mindspore_dtype=VQ_DTYPE
).set_train(False)
image_tokenizer = auto_mixed_precision(
    image_tokenizer, amp_level="O2", dtype=VQ_DTYPE, custom_fp32_cells=[nn.BatchNorm3d]
)
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
text = ["Please describe the image", "请描述该图片"]
image = Image.open("assets/demo.png") # TODO: you need to modify the path here
image = [image, image]
inputs = processor(
    text=text,
    image=image,
    mode="U",
    padding_image=True,
    padding="longest",
    return_tensors="np",
)

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id
)
# generate
outputs = model.generate(
    Tensor(inputs.input_ids, dtype=ms.int32),
    GENERATION_CONFIG,
    max_new_tokens=1024,
    attention_mask=Tensor(inputs.attention_mask),
)
answers = processor.batch_decode(outputs, skip_special_tokens=True)
for ans in answers:
    print(ans)
```
</details>

#### Run Emu3-VisionTokenzier for vision encoding and decoding
An inference script is provided in `scripts/infer_img_rec.sh`.<br>
An example to reconstruct image/video is as follows:

<details>

```python
import os
from PIL import Image
import mindspore as ms
from mindspore import Tensor, nn
from mindone.utils.amp import auto_mixed_precision
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel

# TODO: you need to modify the path here
MODEL_HUB = "BAAI/Emu3-VisionTokenizer"
MS_DTYPE = ms.bfloat16
model = Emu3VisionVQModel.from_pretrained(
        MODEL_HUB,
        use_safetensors=True,
        mindspore_dtype=MS_DTYPE
    ).set_train(False)
model = auto_mixed_precision(
    model, amp_level="O2", dtype=MS_DTYPE, custom_fp32_cells=[nn.BatchNorm3d]
)
processor = Emu3VisionVQImageProcessor.from_pretrained(MODEL_HUB)

# TODO: you need to modify the path here
VIDEO_FRAMES_PATH = "YOUR_VIDEO_FRAMES_PATH"

video = os.listdir(VIDEO_FRAMES_PATH)
video.sort()
video = [Image.open(os.path.join(VIDEO_FRAMES_PATH, v)) for v in video]

images = processor(video, return_tensors="np")["pixel_values"]
images = Tensor(images).unsqueeze(0)

# image autoencode
image = images[:, 0]
print(image.shape)
codes = model.encode(image)
recon = model.decode(codes)

recon = recon.view(-1, *recon.shape[2:])
recon_image = processor.postprocess(recon)["pixel_values"][0]
recon_image.save("recon_image.png")

# video autoencode
# NOTE: number of frames must be multiple of `model.config.temporal_downsample_factor`
images = images.view(
    -1, # if OOM, reduce batch size
    model.config.temporal_downsample_factor,
    *images.shape[2:],
)
codes = model.encode(images)
recon = model.decode(codes)

recon = recon.view(-1, *recon.shape[2:])
recon_images = processor.postprocess(recon)["pixel_values"]
for idx, im in enumerate(recon_images):
    im.save(f"recon_video_{idx}.png")
```

</details>

## Training
### Supervised Fine-tuning (SFT)

## Performance
### Inference
#### Image Reconstruction


Input an image or a clip of video frames, outout the reconstructed image(s).
<br>
Experiments are tested on ascend 910* with mindspore 2.4.1 pynative mode.
<br>
*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` fp16.

| model name	| precision* | cards | batch size| resolution |	s/step | img/s |
| --- | --- | --- | --- | --- | --- | --- |
| Emu3-VisionTokenizer | bfloat16 | 1 | 1         | 768x1360 | 2.93 | 0.34 |
| Emu3-VisionTokenizer | bfloat16 | 1 | 4 (video) | 768x1360 | 0.97 | 4.13 |

<br>
Input an image or a clip of video frames, outout the reconstructed image(s).
<br>
Experiments are tested on ascend 910* with mindspore 2.4.1 graph mode.
<br>

*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` fp16.

| model name | precision* | cards | batch size| resolution | graph compile |	s/step | img/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Emu3-VisionTokenizer | bfloat16 | 1 | 1         | 768x1360 | 15.23 | 3.20 | 0.31 |
| Emu3-VisionTokenizer | bfloat16 | 1 | 4 (video) | 768x1360 | 15.23 | 5.14 | 0.78 |



#### Text-to-Image Generation
Input a text prompt, output an image. <br>
Experiments are tested on ascend 910* with mindspore 2.4.1 pynative mode.

|model name	| precision* | cards | batch size| resolution | flash attn |	s/step	| step | img/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Emu3-Gen | bfloat16 | 1 | 1 | 720x720 | OFF | 1.27 | 8192 | 9.57e-5 |
| Emu3-Gen | bfloat16 | 1 | 1 | 720x720 | ON  | 0.54 | 8192 | 2.27e-4 |

*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` fp16, `FlashAttention` fp16.

#### VQA
Input an image and a text prompt, output textual response. <br>
Experiments are tested on ascend 910* with mindspore 2.4.1 pynative mode.

|model name	| precision* | cards | batch size| resolution | flash attn |	s/step	| step | response/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Emu3-Chat | bfloat16 | 1 | 1 | 720x720 | OFF | 0.24 | 577 | 0.007 |
| Emu3-Chat | bfloat16 | 1 | 1 | 720x720 | ON  | 0.28 | 654 | 0.005 |

*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` fp16, `FlashAttention` fp16.
