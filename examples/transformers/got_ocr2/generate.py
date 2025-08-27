import time
import mindspore as ms
from transformers import AutoProcessor, GotOcr2ImageProcessor
from mindone.transformers import AutoModelForImageTextToText

MODEL_HUB = "stepfun-ai/GOT-OCR-2.0-hf"
IMAGE = "demo.png"

start = time.time()
processor = AutoProcessor.from_pretrained(MODEL_HUB)
processor.image_processor = GotOcr2ImageProcessor.from_pretrained(MODEL_HUB)
print(f"Loaded processor in {time.time()-start:.4f}s")

start = time.time()
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_HUB,
    mindspore_dtype=ms.bfloat16,
    attn_implementation="eager",
)
print(f"Loaded model in {time.time()-start:.4f}s")

np_inputs = processor(IMAGE, return_tensors="np")
inputs = {}
for k, v in np_inputs.items():
    t = ms.Tensor(v)
    t = t.astype(ms.int32) if t.dtype == ms.int64 else t.astype(model.dtype)
    inputs[k] = t

start = time.time()
generated_ids = model.generate(
    **inputs,
    do_sample=False,
    max_new_tokens=200,
    eos_token_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
    pad_token_id = processor.tokenizer.pad_token_id,
)
print(f"Inference in {time.time()-start:.4f}s")

prompt_len = np_inputs["input_ids"].shape[1]
out_ids = generated_ids[0].asnumpy()[prompt_len:]
text = processor.decode(out_ids, skip_special_tokens=True)
print(text)