import time

from emu3.mllm import Emu3ForCausalLM, Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor

# TODO: from mindone.transformers import Emu3ForCausalLM
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image

from transformers.generation.configuration_utils import GenerationConfig

import mindspore as ms
from mindspore import Tensor, nn

from mindone.utils.amp import auto_mixed_precision

ms.set_context(mode=ms.PYNATIVE_MODE)  # only support PYNATIVE using DynamicCache

# 1. Load Models and Processor
start_time = time.time()

# model path
EMU_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"
MS_DTYPE = ms.bfloat16

# prepare model and processor
model = Emu3ForCausalLM.from_pretrained(
    EMU_HUB,
    mindspore_dtype=ms.bfloat16,
    use_safetensors=True,
    attn_implementation="eager", # optional: "flash_attention_2"
).set_train(False)

tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, padding_side="left")
image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB, use_safetensors=True, mindspore_dtype=MS_DTYPE).set_train(
    False
)
image_tokenizer = auto_mixed_precision(
    image_tokenizer, amp_level="O2", dtype=MS_DTYPE, custom_fp32_cells=[nn.BatchNorm3d]
)
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
print("Loaded all models, time elapsed: %.4fs"%(time.time() - start_time))


# 2. Prepare Input
start_time = time.time()

text = ["Please describe the image", "请描述该图片"]
image = Image.open("assets/demo.png") # NOTE: replace with your own image path
image = [image, image] # batch = 2 for example

inputs = processor(
    text=text,
    image=image,
    mode="U",
    padding_image=True,
    padding="longest",
    return_tensors="np",
)
print("Prepared inputs, time elapsed: %.4fs"%(time.time() - start_time))

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
)

# 3. Generate Next Tokens, Decode Tokens

# generate
start_time = time.time()
outputs = model.generate(
    Tensor(inputs.input_ids, dtype=ms.int32),
    GENERATION_CONFIG,
    max_new_tokens=1024,
    attention_mask=Tensor(inputs.attention_mask),
)
print("Finish generation, time elapsed: %.4fs"%(time.time() - start_time))

# detokenization
start_time = time.time()
# outputs = outputs[:, inputs.input_ids.shape[-1] :]
answers = processor.batch_decode(outputs, skip_special_tokens=True)
for ans in answers:
    print(ans)

print("\nFinished, time elapsed: %.4fs"%(time.time() - start_time))
