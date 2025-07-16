# Adapted from https://huggingface.co/docs/transformers/model_doc/led#transformers.LEDForConditionalGeneration.forward.example-2
from transformers import AutoTokenizer

from mindspore import Tensor

from mindone.transformers import LEDForConditionalGeneration

TXT = "My friends are <mask> but they eat too many carbs."
# load from huggingface
tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", revision="refs/pr/4")

# load from modelscope
# run `modelscope download --model allenai/led-base-16384` ahead
# python convert_to_safetensors.py \
# --model_path ~/.cache/modelscope/hub/models/allenai/led-base-16384/pytorch_model.bin \
# --output_path ~/.cache/modelscope/hub/models/allenai/led-base-16384/model.safetensors

# uncomment the following lines to load from modelscope
# checkpoint_path = "/home/user/.cache/modelscope/hub/models/allenai/led-base-16384"
# tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
# model = LEDForConditionalGeneration.from_pretrained(checkpoint_path, mindspore_dtype=ms.float32)


input_ids = Tensor(tokenizer([TXT], return_tensors="np")["input_ids"])
prediction = model.generate(input_ids)[0]
print(tokenizer.decode(prediction, skip_special_tokens=True))
