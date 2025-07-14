# Adapted from https://huggingface.co/docs/transformers/model_doc/led#transformers.LEDForConditionalGeneration.forward.example-2
from transformers import AutoTokenizer

from mindspore import Tensor

from mindone.transformers import LEDForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
TXT = "My friends are <mask> but they eat too many carbs."

model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", revision="refs/pr/4")
input_ids = Tensor(tokenizer([TXT], return_tensors="np")["input_ids"])

prediction = model.generate(input_ids)[0]
print(tokenizer.decode(prediction, skip_special_tokens=True))
