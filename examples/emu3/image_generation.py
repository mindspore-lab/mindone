# Adapted from https://github.com/baaivision/Emu3 to work with MindSpore.
import time

from emu3.mllm import Emu3ForCausalLM, Emu3Processor, Emu3Tokenizer
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image
from transformers.generation.configuration_utils import GenerationConfig

import mindspore as ms
from mindspore import Tensor, mint, ops

from mindone.transformers.generation.logits_process import (
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
    UnbatchedClassifierFreeGuidanceLogitsProcessor,
)
from mindone.utils.amp import auto_mixed_precision

# Both modes are supported
ms.set_context(mode=ms.PYNATIVE_MODE)  # PYNATIVE
# ms.set_context(mode=ms.GRAPH_MODE)      # GRAPH. Not supported yet

# 1. Load Models and Processor
start_time = time.time()

# model path
# NOTE: you need to modify the path of EMU_HUB and VQ_HUB here
EMU_HUB = "BAAI/Emu3-Gen"
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
image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB, use_safetensors=True, mindspore_dtype=VQ_DTYPE).set_train(
    False
)
image_tokenizer = auto_mixed_precision(
    image_tokenizer, amp_level="O2", dtype=VQ_DTYPE, custom_fp32_cells=[mint.nn.BatchNorm3d]
)
processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

print("Loaded all models, time elapsed: %.4fs" % (time.time() - start_time))

# 2. Prepare Input
start_time = time.time()

POSITIVE_PROMPT = "masterpiece, film grained, best quality."
NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, \
     fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

classifier_free_guidance = 3.0
prompt = [
    "a portrait of young girl.",
    "a shiba inu",
]  # NOTE: if OOM, reduce to batch=1, e.g. ["a portrait of young girl."]
prompt = [p + POSITIVE_PROMPT for p in prompt]

kwargs = dict(
    mode="G",
    ratio=["1:1", "16:9"],  # NOTE: if OOM, reduce to batch=1, e.g. ["1:1"]
    image_area=model.config.image_area,  # 720*720, NOTE: if OOM, reduce it to 512*512
    return_tensors="np",
    padding="longest",
)
pos_inputs = processor(text=prompt, **kwargs)
neg_inputs = processor(text=[NEGATIVE_PROMPT] * len(prompt), **kwargs)

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

h = Tensor(pos_inputs.image_size[:, 0])
w = Tensor(pos_inputs.image_size[:, 1])
constrained_fn = processor.build_prefix_constrained_fn(h, w)
logits_processor = LogitsProcessorList(
    [
        UnbatchedClassifierFreeGuidanceLogitsProcessor(
            classifier_free_guidance,
            model,
            unconditional_ids=Tensor(neg_inputs.input_ids, dtype=ms.int32),
        ),
        PrefixConstrainedLogitsProcessor(
            constrained_fn,
            num_beams=1,
        ),
    ]
)

print("Prepared inputs, time elapsed: %.4fs" % (time.time() - start_time))


# 3. Generate Next Tokens, Decode Tokens

start_time = time.time()
outputs = model.generate(
    Tensor(pos_inputs.input_ids, dtype=ms.int32),
    GENERATION_CONFIG,
    logits_processor=logits_processor,
    attention_mask=Tensor(pos_inputs.attention_mask),
)
print(f"generated_ids length / #steps: {len(outputs[0])}")
elapsed = time.time() - start_time
print("Average speed %.4fs/step" % (elapsed / len(outputs[0])))
print("Finish generation, time elapsed: %.4fs" % (time.time() - start_time))

# since input_ids are deleted in generate() output
# need to add input_ids back ahead, which contains visual boi/eoi tokens and meta data for image detokenization
outputs = ops.cat((Tensor(pos_inputs.input_ids, dtype=outputs.dtype), outputs), axis=1)
start_time = time.time()
for idx_i, out in enumerate(outputs):
    mm_list = processor.decode(out)
    for idx_j, im in enumerate(mm_list):
        if not isinstance(im, Image.Image):
            continue
        im.save(f"result_{idx_i}_{idx_j}.png")
        print(f"Saved result_{idx_i}_{idx_j}.png")
print("Finish detokenization, time elapsed: %.4fs" % (time.time() - start_time))
