from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import LEDModel

TXT = "My friends are <mask> but they eat too many carbs."
# load from huggingface
# tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
# model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", revision="refs/pr/4")


# load from modelscope (run `modelscope download --model allenai/led-base-16384` ahead)
checkpoint_path = "/home/user/.cache/modelscope/hub/models/allenai/led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = LEDModel.from_pretrained(checkpoint_path, mindspore_dtype=ms.float32)

input_ids = Tensor(tokenizer([TXT], return_tensors="np")["input_ids"])
prediction = model(input_ids).last_hidden_state
