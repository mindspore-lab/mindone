# MindONE Transformers Models

Mindone transformers models has supported CLIP Model Series and T5 Model Series.

## T5 Model Series

T5 Model is first pre-trained on a data-rich task and then fine-tune on a downstrem task.
As its excellent performance on NLP task, it is widely used as the text encoder in downstream tasks such as Imagen, stable diffusion 3 and sora.
Instruction tuning is used in Flan-t5 so that the model could be scaling up to bigger sizes and more tasks. Also, Flan-t5 is more suitable for
reasoing task. Therefore, Flan-t5 model is ideal to be used as text encoder.

In this part, T5EncoderModel, T5Model and T5ForConditionalGeneration are supported.
Additionally, google-t5/t5-small, DeepFloyd/t5-v1.1-xxl and google/flan-t5-large have been supported in this framework by so far.

These models can be utilized conveniently by the following commands:

```shell
from transformers import AutoTokenizer
from mindone.transformers.models import T5Model

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5Model.from_pretrained("google-t5/t5-small")

input_ids = tokenizer(
     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(decoder_input_ids)

# forward pass
outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
last_hidden_states = outputs[0]
encoder_outputs = outputs[1]
```

```shell
from transformers import AutoTokenizer
from mindone.transformers.models import T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl", revision="refs/pr/3")
model = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl", revision="refs/pr/3")
input_ids = tokenizer(
     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
).input_ids  # Batch size 1
outputs = model(input_ids=input_ids)
encoder_outputs = outputs
```

```shell
from transformers import AutoTokenizer
from mindone.transformers.models import T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
outputs = model(input_ids=input_ids, labels=labels)
logits = outputs[0]
encoder_outputs = outputs[1]
```

The variable "encoder_outputs" could be used as text encoder in T5 Model Series.
