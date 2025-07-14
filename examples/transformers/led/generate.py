# Adapted from https://huggingface.co/docs/transformers/model_doc/led#transformers.LEDForConditionalGeneration.forward.example-2
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import LEDForConditionalGeneration

TXT = "My friends are <mask> but they eat too many carbs."
# load from huggingface
# tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
# model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", revision="refs/pr/4")


# load from modelscope (run `modelscope download --model allenai/led-base-16384` ahead)
checkpoint_path = "~/.cache/modelscope/hub/models/allenai/led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = LEDForConditionalGeneration.from_pretrained(checkpoint_path, mindspore_dtype=ms.float16)
input_ids = Tensor(tokenizer([TXT], return_tensors="np")["input_ids"])

prediction = model.generate(input_ids)[0]
print(tokenizer.decode(prediction, skip_special_tokens=True))
