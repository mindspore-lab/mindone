# Emu3: Next-Token Prediction is All You Need


 [Paper](https://arxiv.org/pdf/2409.18869) | [🤗HF Models](https://huggingface.co/collections/BAAI/emu3-66f4e64f70850ff358a2e60f) | [Modelscope](https://modelscope.cn/collections/Emu3-9eacc8668b1043) |


## Introduction
**Emu3** is a new suite of state-of-the-art multimodal models trained solely with **<i>next-token prediction</i>**. By tokenizing images, text, and videos into a discrete space, a single transformer is trained from scratch on a mixture of multimodal sequences.

As a multimodal LLM, Emu3 uses vector quantization to tokenize images into discrete tokens. Discretized image tokens are later fused with text token ids for image and text generation. The model can additionally generate images by predicting image token ids.

<!-- ### Emu3 excels in both generation and perception -->
**Emu3** outperforms several well-established task-specific models in both ***generation*** and ***perception*** tasks, surpassing flagship open models such as SDXL, LLaVA-1.6 and OpenSora-1.2, while eliminating the need for diffusion or compositional architectures.


### Highlights

- **Emu3** is capable of generating high-quality images following the text input, by simply predicting the next vision token. The model naturally supports flexible resolutions and styles.
- **Emu3** shows strong vision-language understanding capabilities to see the physical world and provides coherent text responses. Notably, this capability is achieved without depending on a CLIP and a pretrained LLM.
- **Emu3** simply generates a video causally by predicting the next token in a video sequence, unlike the video diffusion model as in Sora. With a video in context, Emu3 can also naturally extend the video and predict what will happen next.

### Features

- Model weights of Vision Tokenizer, Emu3-Stage1, Emu3-Chat and Emu3-Gen.
- Inference code.
- Training scripts for sft.

## Demos
Text to Image Generation:

|input| generated image|
|---|---|
|prompt: `"a portrait of young girl"`<br> image cnofig: {ratio:1:1, image_area:720x720} |<img src="https://github.com/user-attachments/assets/3e79b937-2d6d-4098-bb37-977d6c592854" height="384px"> |
|prompt: `"a shiba inu"`<br> image cnofig: {ratio:16:9, image_area:720x720}|<img src="https://github.com/user-attachments/assets/d7b3c287-42d2-40ea-93ec-e46b02289850" height="384px">|

Image VQA:

|input | response|
|---|---|
|<img src="https://github.com/user-attachments/assets/bb84d826-39fd-4e97-9471-e0592d1158b7" width="512px"><br> prompt: `"Please describe the image"`| The image is a closeup of a dog with a happy expression, looking directly at the camera. The dog has a brown and white coat, with a distinctive white stripe running down the center of its face. The background is blurred, with hints of greenery suggesting an outdoor setting. The dog appears to be sitting on grass, with a few yellow flowers visible in the lower part of the image. The overall tone of the image is cheerful and friendly.|

## Get Started
### Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.5.0 | 24.1RC2 | 7.3.0.1.231 | 8.0.RC3.beta1|

### Dependencies

Enter this directory and install required packages:

```shell
cd mindone
pip install -e .[training]
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

Emu3-Stage1 can be fine-tuned on text-to-image(T2I) or vision-question answering (VQA) tasks.<br>
The model continues training with the next token prediction task using standard cross-entropy loss.<br>
For T2I task, supervision is applied exclusively on vision tokens; while for VQA task, supervision is applied exclusively on response text tokens.

Some SFT scripts are provided in `scripts/XX_sft_seq_parallel.sh`. To fine-tune Emu3-Stage1, run the following script:
```
bash scripts/t2i_sft_seq_parallel.sh # for T2I task
bash scripts/vqa_sft_seq_parallel.sh # for VQA task
```

#### Dataset Preparation
Training data should store input and output, vision and text tokens.

`train/prepare_vision_data.py` and `train/prepare_vqa_data.py` are used to format T2I and VQA data respectively into:
```md
DATA_DIR
├── list
│   │   └── train.json
│   ├── feature
│   │   ├── 0000.ckpt
│   │   ├── 0001.ckpt
│   │   ├── ...
│   │   ├── N-2.ckpt
│   │   └── N-1.ckpt
```
`train.json` lists all `*.ckpt` locations, each `*.ckpt` stores:
```
{
    "name": name,
    "images": token_ids,
    "texts": input_prompt,
    "response": answer_prompt # optional
}
```

## Performance
### Inference
#### Image Reconstruction


Input an image or a clip of video frames, outout the reconstructed image(s).
<br>
Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 pynative mode.

| model name	| precision* | cards | batch size| resolution |	s/step | img/s |
| --- | --- | --- | --- | --- | --- | --- |
| Emu3-VisionTokenizer | bfloat16 | 1 | 1         | 768x1360 | 2.65 | 0.38 |
| Emu3-VisionTokenizer | bfloat16 | 1 | 4 (video) | 768x1360 | 0.96 | 1.04 |

*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` and `Flash Attention` use fp16.

<br>
Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 graph mode.

| model name | precision* | cards | batch size| resolution | graph compile |	s/step | img/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Emu3-VisionTokenizer | bfloat16 | 1 | 1         | 768x1360 | 15s | 3.23 | 0.31 |
| Emu3-VisionTokenizer | bfloat16 | 1 | 4 (video) | 768x1360 | 15s | 5.46 | 0.18 |

*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` and `Flash Attention` use fp16.

#### Text-to-Image Generation
Input a text prompt, output an image.
<br>
Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 pynative mode.

|model name	| precision* | cards | batch size| resolution | flash attn |	s/step	| step | img/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Emu3-Gen | bfloat16 | 1 | 1 | 720x720 | OFF | 0.50 | 8193 | 2.27-e4 |
| Emu3-Gen | bfloat16 | 1 | 1 | 720x720 | ON  | 0.49 | 8193 | 2.50-e4 |

*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` and `Flash Attention` use fp16.

#### VQA
Input an image and a text prompt, output textual response.
<br>
Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0  pynative mode.

|model name	| precision* | cards | batch size| resolution | flash attn |	s/step	| step | response/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Emu3-Chat | bfloat16 | 1 | 1 | 384x384 | OFF | 0.29 | 131 | 0.03 |
| Emu3-Chat | bfloat16 | 1 | 1 | 384x384 | ON  | 0.24 | 92 | 0.05 |

*note: mixed precision, `BatchNorm3d` uses fp32, `Conv3d` and `Flash Attention` use fp16.

### Training

Experiments are tested on Ascend Atlas 800T A2 machines with mindspore 2.5.0 pynative mode.

| stage | pre-trained model	| precision* | cards | batch size| resolution | max token | init lr | recompute | zero stage | grad accu |flash attn | sequence parallel |	s/step	| step | sample/s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage2-T2I | Emu3-Stage1 | float16 | 8 | 1 | 512x512 | 4200 | 1e-6 | ON | 3 | 1 | ON | 8 shards | 2.61 | 4996 | 0.38 |
| stage2-VQA | Emu3-Stage1 | float16 | 4 | 1 | 384x384 | 2560 | 1e-5 | ON | 3 | 1 | ON | 8 shards | 3.08 | 4993 | 0.32 |

*note: mixed precision, `BatchNorm3d` and `Emu3RMSNorm` use fp32.
