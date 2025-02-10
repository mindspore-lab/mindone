# -*- coding: utf-8 -*-
from emu3.mllm import Emu3ForCausalLM, Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor

# from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image

# TODO: from mindone.transformers import Emu3ForCausalLM
from transformers.generation.configuration_utils import GenerationConfig

import mindspore as ms
from mindspore import Tensor, nn

from mindone.utils.amp import auto_mixed_precision

# model path
EMU_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"
MS_DTYPE = ms.bfloat16

# prepare model and processor
model = Emu3ForCausalLM.from_pretrained(
    EMU_HUB,
    mindspore_dtype=ms.bfloat16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
)
model.set_train(False)

tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, padding_side="left")
image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB, use_safetensors=True, mindspore_dtype=MS_DTYPE).set_train(
    False
)
image_tokenizer = auto_mixed_precision(
    image_tokenizer, amp_level="O2", dtype=MS_DTYPE, custom_fp32_cells=[nn.BatchNorm3d]
)
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
text = ["Please describe the image", "Please describe the image"]
image = Image.open("assets/demo.png")
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
    pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
)

# generate
outputs = model.generate(
    Tensor(inputs.input_ids, dtype=ms.int32),
    GENERATION_CONFIG,
    max_new_tokens=1024,
    attention_mask=Tensor(inputs.attention_mask),
)

outputs = outputs[:, inputs.input_ids.shape[-1] :]
answers = processor.batch_decode(outputs, skip_special_tokens=True)
for ans in answers:
    print(ans)
