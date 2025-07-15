import numpy as np
from transformers import AutoTokenizer

import mindspore
import mindspore as ms
from mindspore import Tensor

from mindone.transformers import LEDModel

TXT = "My friends are <mask> but they eat too many carbs."


# load from modelscope (run `modelscope download --model allenai/led-base-16384` ahead)
checkpoint_path = "/home/user/.cache/modelscope/hub/models/allenai/led-base-16384"
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = LEDModel.from_pretrained(checkpoint_path, mindspore_dtype=ms.float32)

input_ids = Tensor(tokenizer([TXT], return_tensors="np")["input_ids"])

# Create input tensors
batch_size, seq_len = 2, 1024
hidden_states = np.random.randn(batch_size, seq_len, 768)

# Convert to respective framework tensors
ms_hidden_states = Tensor(hidden_states, dtype=mindspore.float32)

# Create attention masks
attention_mask = np.ones((batch_size, seq_len))
ms_attention_mask = Tensor(attention_mask, dtype=mindspore.float32)

ms_is_index_masked = Tensor([False] * (seq_len // 2) + [True] * (seq_len // 2), dtype=mindspore.bool_).broadcast_to(
    (batch_size, seq_len)
)

ms_is_index_global_attn = Tensor([False] * seq_len, dtype=mindspore.bool_).broadcast_to((batch_size, seq_len))


prediction = model.encoder.layers[0](
    ms_hidden_states,
    attention_mask=ms_attention_mask,
    layer_head_mask=None,
    is_index_masked=ms_is_index_masked,
    is_index_global_attn=ms_is_index_global_attn,
    output_attentions=False,
)


# torch
# load from huggingface
# tokenizer = AutoTokenizer.from_pretrained("allenai/led-base-16384")
# model = LEDForConditionalGeneration.from_pretrained("allenai/led-base-16384", revision="refs/pr/4")
# Create input tensors
# batch_size, seq_len = 2, 1024
# hidden_states = np.load("hidden_states.npy")

# torch_hidden_states = torch.tensor(hidden_states, dtype=torch.float32)
# torch_attention_mask = torch.tensor(attention_mask, dtype=torch.float32)
# torch_is_index_masked = torch.tensor(
#     [False] * (seq_len // 2) + [True] * (seq_len // 2), dtype=torch.bool
# ).expand(batch_size, seq_len)
# torch_is_index_global_attn = torch.tensor([False] * seq_len, dtype=torch.bool).expand(batch_size, seq_len)

# prediction = model.encoder.layers[0](
#     ms_hidden_states,
#     attention_mask=ms_attention_mask,
#     layer_head_mask=None,
#     is_index_masked=ms_is_index_masked,
#     is_index_global_attn=ms_is_index_global_attn,
#     output_attentions=False,
# )
