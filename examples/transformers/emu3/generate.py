import time

import requests
import torch
from PIL import Image
from transformers import Emu3Processor

import mindspore as ms

from mindone.transformers import Emu3ForConditionalGeneration

start_time = time.time()
processor = Emu3Processor.from_pretrained("BAAI/Emu3-Chat-hf")
model = Emu3ForConditionalGeneration.from_pretrained("BAAI/Emu3-Chat-hf", mindspore_dtype=ms.bfloat16)
print(
    "Finished loading Emu3Processor and Emu3ForConditionalGeneration, time elapsed: %.4fs" % (time.time() - start_time)
)

##############################
# Text Generation / Image Understanding Inference
print("*" * 50)
print("Image Understanding Inference")

# prepare image and text prompt
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
prompt = "What do you see in this image?<image>"

inputs = processor(images=image, text=prompt, return_tensors="np")
for k, v in inputs.items():
    inputs[k] = ms.Tensor(v, dtype=ms.bfloat16)
print(inputs)

# autoregressively complete prompt
output = model.generate(**inputs, max_new_tokens=50)
print(processor.decode(output[0], skip_special_tokens=True))

##############################
# Image Generation Inference
print("*" * 50)
print("Image Generation Inference")

inputs = processor(
    text=["a portrait of young girl. masterpiece, film grained, best quality.", "a dog running under the rain"],
    padding=True,
    return_tensors="np",
    return_for_image_generation=True,
)
for k, v in inputs.items():
    inputs[k] = ms.Tensor(v, dtype=ms.bfloat16)
print(inputs)

neg_prompt = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, " \
"cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."
neg_inputs = processor(text=[neg_prompt] * 2, return_tensors="np")
for k, v in neg_inputs.items():
    neg_inputs[k] = ms.Tensor(v, dtype=ms.bfloat16)
print("neg_inputs", neg_inputs)

image_sizes = inputs.pop("image_sizes")
HEIGHT, WIDTH = image_sizes[0]
VISUAL_TOKENS = model.vocabulary_mapping.image_tokens


def prefix_allowed_tokens_fn(batch_id, input_ids):
    height, width = HEIGHT, WIDTH
    visual_tokens = VISUAL_TOKENS
    image_wrapper_token_id = ms.tensor([processor.tokenizer.image_wrapper_token_id])
    eoi_token_id = ms.tensor([processor.tokenizer.eoi_token_id])
    eos_token_id = ms.tensor([processor.tokenizer.eos_token_id])
    pad_token_id = ms.tensor([processor.tokenizer.pad_token_id])
    eof_token_id = ms.tensor([processor.tokenizer.eof_token_id])
    eol_token_id = ms.tensor(processor.tokenizer.encode("<|extra_200|>", return_tensors="np")[0])

    position = torch.nonzero(input_ids == image_wrapper_token_id, as_tuple=True)[0][0]
    offset = input_ids.shape[0] - position
    if offset % (width + 1) == 0:
        return (eol_token_id,)
    elif offset == (width + 1) * height + 1:
        return (eof_token_id,)
    elif offset == (width + 1) * height + 2:
        return (eoi_token_id,)
    elif offset == (width + 1) * height + 3:
        return (eos_token_id,)
    elif offset > (width + 1) * height + 3:
        return (pad_token_id,)
    else:
        return visual_tokens


out = model.generate(
    **inputs,
    max_new_tokens=50_000,  # make sure to have enough tokens for one image
    prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    return_dict_in_generate=True,
    negative_prompt_ids=neg_inputs.input_ids,  # indicate for Classifier-Free Guidance
    negative_prompt_attention_mask=neg_inputs.attention_mask,
)

image = model.decode_image_tokens(out.sequences[:, inputs.input_ids.shape[1] :], height=HEIGHT, width=WIDTH)
images = processor.postprocess(
    list(image.float()), return_tensors="PIL.Image.Image"
)  # internally we convert to np but it's not supported in bf16 precision
for i, image in enumerate(images["pixel_values"]):
    image.save(f"result{i}.png")
