import mindspore as ms

from PIL import Image
from transformers import AutoTokenizer
from mindone.transformers import MiniCPMV_v2_6

model = MiniCPMV_v2_6.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True, attn_implementation='eager', mindspore_dtype=ms.float32)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6', trust_remote_code=True)

image = Image.open('airplane.jepg').convert('RGB')

# First Round Chat
question = "Tell me the model of this aircraft"
msgs = [{"role": 'user', 'content': [image, question]}]
answer = model.chat(image=image, msgs=msgs, tokenizer=tokenizer)
print(answer)

# Second round chat
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": [answer]})
msgs.append({"role": "user", "content": ["Introduce something about Airbus A380."]})

answer = model.chat(image=None, msgs=msgs, tokenizer=tokenizer)
print(answer)