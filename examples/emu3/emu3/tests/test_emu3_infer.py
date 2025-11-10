# Debug and testing use only
# Test inference pipeline

import time

from emu3.mllm import Emu3ForCausalLM
from emu3.mllm.configuration_emu3 import Emu3Config
from PIL import Image

import mindspore as ms
from mindspore import mint

ms.set_context(mode=ms.PYNATIVE_MODE, pynative_synchronize=True)
# ms.set_context(mode = ms.GRAPH_MODE) # NOT SUPPORTED YET

# config.json
config_json = {
    "architectures": ["Emu3ForCausalLM"],
    "attention_dropout": 0.1,
    "auto_map": {
        "AutoConfig": "configuration_emu3.Emu3Config",
        "AutoModelForCausalLM": "modeling_emu3.Emu3ForCausalLM",
    },
    "boi_token_id": 151852,
    "bos_token_id": 151849,
    "eof_token_id": 151847,
    "eoi_token_id": 151853,
    "eol_token_id": 151846,
    "eos_token_id": 151850,
    "hidden_act": "silu",
    "hidden_size": 4096,
    "image_area": 262144,
    "img_token_id": 151851,
    "initializer_range": 0.02,
    "intermediate_size": 14336,
    "max_position_embeddings": 5120,
    "model_type": "Emu3",
    "num_attention_heads": 32,
    "num_hidden_layers": 32,
    "num_key_value_heads": 8,
    "pad_token_id": 151643,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 1000000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.44.0",
    "use_cache": True,
    "vocab_size": 184622,
    "attn_implementation": "flash_attention_2",
}

# TEST: loading model
start_time = time.time()
config = Emu3Config(**config_json)
try:
    model = Emu3ForCausalLM(config).set_train(False)
    print("*" * 100)
    print("Test passed: Sucessfully loaded Emu3ForCausalLM")
    print("Time elapsed: %.4fs" % (time.time() - start_time))
    print("*" * 100)
except RuntimeError:
    raise RuntimeError("Load Emu3ForCausalLM Error.")

# TEST: load processor
start_time = time.time()
from emu3.mllm import Emu3ForCausalLM, Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel

from mindone.utils.amp import auto_mixed_precision

EMU_HUB = "BAAI/Emu3-Gen"
VQ_HUB = "BAAI/Emu3-VisionTokenizer"
VQ_DTYPE = ms.bfloat16
try:
    tokenizer = Emu3Tokenizer.from_pretrained(EMU_HUB, padding_side="left")
    image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(
        VQ_HUB, use_safetensors=True, mindspore_dtype=VQ_DTYPE
    ).set_train(False)
    image_tokenizer = auto_mixed_precision(
        image_tokenizer, amp_level="O2", dtype=VQ_DTYPE, custom_fp32_cells=[mint.nn.BatchNorm3d]
    )
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    print("*" * 100)
    print("Test passed: Sucessfully loaded Emu3Processor")
    print("Time elapsed: %.4fs" % (time.time() - start_time))
    print("*" * 100)
except RuntimeError:
    raise RuntimeError("Load Emu3Processor Error.")

# TEST: process input
start_time = time.time()
text = ["Describe this image."]
w, h = 1024, 512
image_path = "demo.jpeg"  # REPLACE with your image
image_inputs = [Image.open(image_path).convert("RGB").resize((w, h))]
# image = np.uint8(np.random.rand(h, w, 3) * 255)
# image_inputs = [Image.fromarray(image).convert("RGB")]
inputs = processor(
    text=text,
    image=image_inputs,
    mode="U",
    padding_image=True,
    padding="longest",
    return_tensors="np",
)
print("*" * 100)
print("Test passed: Sucessfully processed input data using Emu3Processor")
print("Time elapsed: %.4fs" % (time.time() - start_time))
print("*" * 100)

# TEST: dummy inference
from transformers.generation.configuration_utils import GenerationConfig

GENERATION_CONFIG = GenerationConfig(
    pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
)
start_time = time.time()
try:
    generated_ids = model.generate(
        ms.Tensor(inputs.input_ids, dtype=ms.int32),
        GENERATION_CONFIG,
        max_new_tokens=128,
        attention_mask=ms.Tensor(inputs.attention_mask),
    )
    print("*" * 100)
    print("Test passed: Sucessfully generated tokens using Emu3ForCausalLM")
    print(f"generated_ids length / #steps: {len(generated_ids[0])}")
    elapsed = time.time() - start_time
    print("Time elapsed: %.4fs" % (elapsed))
    print("Average speed %.4fs/step" % (elapsed / len(generated_ids[0])))
    print("*" * 100)
except RuntimeError:
    raise RuntimeError("Run Emu3ForCausalLM.generate() Error.")

start_time = time.time()
try:
    answers = processor.batch_decode(generated_ids, skip_special_tokens=True)
    for ans in answers:
        print(ans)
    print("*" * 100)
    print("Test passed: Sucessfully detokenize generated tokens")
    print("Time elapsed: %.4fs" % (time.time() - start_time))
    print("*" * 100)
except RuntimeError:
    raise RuntimeError("Run Emu3Processor.decode() Error.")
