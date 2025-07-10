import mindspore as ms
from transformers import AutoProcessor
from mindone.transformers import IdeficsForVisionText2Text


# Idefics #
print("Loading Idefics")
MODEL_HUB = "HuggingFaceM4/idefics-9b"
model = IdeficsForVisionText2Text.from_pretrained(
    MODEL_HUB,
    mindspore_dtype=ms.bfloat16
).set_train(False)
processor = AutoProcessor.from_pretrained(MODEL_HUB)

# We feed to the model an arbitrary sequence of text strings and images. Images can be either URLs or PIL Images.
prompts = [
    [
        "https://upload.wikimedia.org/wikipedia/commons/8/86/Id%C3%A9fix.JPG",
        "In this picture from Asterix and Obelix, we can see"
    ],
]

# --batched mode
inputs = processor(prompts, return_tensors="np")
# --single sample mode
# inputs = processor(prompts[0], return_tensors="np")
for k, v in inputs.items():
    inputs[k] = ms.tensor(v)
    if inputs[k].dtype == ms.int64:
        inputs[k] = inputs[k].to(ms.int32)
    else:
        inputs[k] = inputs[k].to(model.dtype)

# Generation args
bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

generated_ids = model.generate(**inputs, bad_words_ids=bad_words_ids, max_length=100)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
for i, t in enumerate(generated_text):
    print(f"{i}:\n{t}\n")