# -*- coding: utf-8 -*-
# from transformers import AutoModel, AutoTokenizer, AutoImageProcessor, AutoModelForCausalLM
from emu3.mllm import Emu3ForCausalLM, Emu3Processor, Emu3Tokenizer
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image

# TODO: from mindone.transformers import Emu3ForCausalLM
from transformers.generation.configuration_utils import GenerationConfig

import mindspore as ms
from mindspore import Tensor

from mindone.transformers.generation.logits_process import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)

# model path
EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

# prepare model and processor
PATH_TO_CONVERTED_EMU3_WEIGHTS = "BAAI/Emu3-Gen"
model = Emu3ForCausalLM.from_pretrained(
    PATH_TO_CONVERTED_EMU3_WEIGHTS,
    mindspore_dtype=ms.bfloat16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
    trust_remote_code=True,
)
# model = AutoModelForCausalLM.from_pretrained(
#     EMU_HUB,
#     device_map="cuda:0",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     trust_remote_code=True,
# )
model.set_train(False)

tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, padding_side="left")  # TODO
image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)  # TODO
image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB).set_train(False)  # TODO
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
POSITIVE_PROMPT = "masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, \
     fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = ["a portrait of young girl.", "a shiba inu"]
prompt = [p + POSITIVE_PROMPT for p in prompt]

kwargs = dict(
    mode="G",
    ratio=["1:1", "16:9"],
    image_area=model.config.image_area,
    return_tensors="np",
    padding="longest",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=[NEGATIVE_PROMPT] * len(prompt), **kwargs)

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    use_cache=True,
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.pad_token_id,
    max_new_tokens=40960,
    do_sample=True,
    top_k=2048,
)

h = pos_inputs.image_size[:, 0]
w = pos_inputs.image_size[:, 1]
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList(
    [
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            classifier_free_guidance,
            model,
            unconditional_ids=Tensor(neg_inputs.input_ids),
        ),
        PrefixConstrainedLogitsProcessor(
            constrained_fn,
            num_beams=1,
        ),
    ]
)

# generate
outputs = model.generate(
    Tensor(pos_inputs.input_ids),
    GENERATION_CONFIG,
    logits_processor=logits_processor,
    attention_mask=Tensor(pos_inputs.attention_mask),
)

for idx_i, out in enumerate(outputs):
    mm_list = processor.decode(out)
    for idx_j, im in enumerate(mm_list):
        if not isinstance(im, Image.Image):
            continue
        im.save(f"result_{idx_i}_{idx_j}.png")
