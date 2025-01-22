# -*- coding: utf-8 -*-
from PIL import Image
# from transformers import AutoTokenizer, AutoModel, AutoImageProcessor, AutoModelForCausalLM
from emu3.tokenizer import Emu3VisionVQModel, Emu3VisionVQImageProcessor
from emu3.mllm import Emu3ForCausalLM, Emu3Tokenizer
#TODO: from mindone.transformers import Emu3ForCausalLM 
from transformers.generation.configuration_utils import GenerationConfig 
from emu3.mllm.processing_emu3 import Emu3Processor
from mindspore import Tensor

# model path
EMU_HUB = "BAAI/Emu3-Chat"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"

# prepare model and processor
PATH_TO_CONVERTED_EMU3_WEIGHTS="BAAI/Emu3-Chat"
model = Emu3ForCausalLM.from_pretrained(   
    PATH_TO_CONVERTED_EMU3_WEIGHTS,
    mindspore_dtype=ms.bfloat16,
    use_safetensors=True,
    attn_implementation="flash_attention_2",
    trust_remote_code=True
)
# model = AutoModelForCausalLM.from_pretrained(
#     EMU_HUB,
#     device_map="cuda:0",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     trust_remote_code=True,
# )
model.set_train(False)

tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, trust_remote_code=True, padding_side="left")
image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB, device_map="cuda:0", trust_remote_code=True).eval() # TODO
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

# prepare input
text = ["Please describe the image", "Please describe the image"]
image = Image.open("assets/demo.png")
image = [image, image]

inputs = processor(
    text=text,
    image=image,
    mode='U',
    padding_image=True,
    padding="longest",
    return_tensors="np",
)

# prepare hyper parameters
GENERATION_CONFIG = GenerationConfig(pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)

# generate
outputs = model.generate(
    Tensor(inputs.input_ids),
    GENERATION_CONFIG,
    max_new_tokens=1024,
    attention_mask=inputs.attention_mask.to("cuda:0"),
)

outputs = outputs[:, inputs.input_ids.shape[-1]:]
answers = processor.batch_decode(outputs, skip_special_tokens=True)
for ans in answers:
    print(ans)
