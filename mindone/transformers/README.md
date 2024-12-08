# Make 🤗 Transformers run on MindSpore

<br>

> State-of-the-art transformers models to perform tasks on different modalities such as text, vision,
> and audio in MindSpore. We've tried to provide a similar interface and usage with the
> [huggingface/transformers](https://github.com/huggingface/transformers). Only necessary changes are made to
> the [huggingface/transformers](https://github.com/huggingface/transformers) to make it seamless for users from torch.

🤗 **Development Principles**

- Only necessary changes are made to the [huggingface/transformers](https://github.com/huggingface/transformers)
- Configuration, Tokenizer, etc. will utilize the original Transformers.

🤗 **Currently**,
we provides pretrained models, generation api, trainer, etc.
to be enables researchers or developers in the field of AIGC and MLLMs to utilize Transformers on MindSpore.

🤗 **Comming Soon**,
latest state-of-the-art models, auto class, pipeline, agent, distributed and so on.


## Quick Tour

The following lines of code are an example that shows you how to download and use the pretrained models.
Remember that the models are from `mindone.transformers`, and anything else is from 🤗 Transformers.

```diff
+from mindspore import Tensor

# use tokenizer from 🤗 transformers
from transformers import AutoTokenizer

# replace model from 🤗 transformers to mindone.transformers
-from transformers import LlamaForCausalLM
+from mindone.transformers import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B)
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

inputs = tokenizer(
    ["A list of colors: red, blue", "Portugal is"],
    padding=True,
-    return_tensors="pt",
+    return_tensors="np"
)
-outputs = model(**inputs)
+outputs = model(Tensor(inputs.input_ids))
```

Then run text generation.

```diff
generated_ids = model.generate(
-    **inputs,
+    input_ids=Tensor(inputs.input_ids),
    max_new_tokens=30,
    use_cache=True,
    do_sample=False
)

tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
```


## Tutorials

| Section                                                                                              | Description                                                                                            |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| [Generation with LLMs](../../docs/transformers/tutorials/generation.md)                              | Generate text with an LLM                                                                              |
| [Training and fine-tuning](../../docs/transformers/tutorials/finetune.md)                            | Using the models provided by 🤗 Transformers in a native MindSpore training loop and the `Trainer` API |
| [Distributed training and mixed precision](../../docs/transformers/tutorials/finetune_distribute.md) | Example scripts for fine-tuning models using distribute and mix precision                              |


## Model Zoo

We introduced some of the provided models and basic usage, as detailed below:

<details onclose>

### CLIP

The CLIP model was proposed in [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) by
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever.
CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on a variety of (image, text) pairs.
It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3.

We have tested the following pretrained weights from huggingface hub. Any other pretrained weights of CLIP model probably also works.

#### OpenAI & LAION

- [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- [laion/CLIP-ViT-H-14-laion2B-s32B-b79K](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K)
- [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)

```python
from mindspore import Tensor
from transformers import CLIPTokenizer
from mindone.transformers import CLIPTextModel

MODEL_NAME = "choose-one-from-the-above-list"
model = CLIPTextModel.from_pretrained(MODEL_NAME)
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

text_inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="np")
text_outputs = model(Tensor(text_inputs.input_ids))
```

#### Stable Diffusion 2.1

[stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) is a model that can be used to generate and modify images based on text prompts.
It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses a fixed, pretrained text encoder ([OpenCLIP-ViT/H](https://github.com/mlfoundations/open_clip)).

```python
from mindspore import Tensor
from transformers import CLIPTokenizer
from mindone.transformers import CLIPTextModel

MODEL_NAME="stabilityai/stable-diffusion-2-1"
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
text_inputs = tokenizer(
    ["a photo of a cat", "a photo of a dog"],
    max_length=tokenizer.model_max_length,
    padding="max_length",
    truncation=True,
    return_tensors="np",
)
encoder_hidden_states = text_encoder(Tensor(text_inputs.input_ids))[0]
```

#### Stable Diffusion XL

[stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) is a model that can be used to generate and modify images based on text prompts.
It is a [Latent Diffusion Model](https://arxiv.org/abs/2112.10752) that uses two fixed, pretrained text encoders ([OpenCLIP-ViT/G](https://github.com/mlfoundations/open_clip) and [CLIP-ViT/L](https://github.com/openai/CLIP)).

```python
from mindspore import Tensor
from transformers import AutoTokenizer
from mindone.transformers import CLIPTextModel, CLIPTextModelWithProjection

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
tokenizer_one = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(MODEL_NAME, subfolder="tokenizer_2", use_fast=False)
text_encoder_one = CLIPTextModel.from_pretrained(MODEL_NAME, subfolder="text_encoder")
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME, subfolder="text_encoder_2")

for tokenizer, text_encoder in zip([tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]):
    text_inputs = tokenizer(
        ["a photo of a cat", "a photo of a dog"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="np",
    )
    text_input_ids = text_inputs.input_ids
    prompt_embeds = text_encoder(Tensor(text_input_ids), output_hidden_states=True)
```

### T5

The T5 model was presented in [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/pdf/1910.10683.pdf) by
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu.

We have tested the following pretrained weights from huggingface hub. Any other pretrained weights of T5 model probably also works.

#### google-t5/t5-small

[google-t5/t5-small](https://huggingface.co/google-t5/t5-small) is the checkpoint with 60 million parameters.
It can be used as an encoder-decoder architecture `T5Model`, or just the encoder part `T5Model.encoder`.

```python
from mindspore import Tensor
from transformers import AutoTokenizer
from mindone.transformers import T5Model

tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = T5Model.from_pretrained("google-t5/t5-small")

input_ids = tokenizer(
     "Studies have been shown that owning a dog is good for you", return_tensors="np"
).input_ids  # Batch size 1
decoder_input_ids = tokenizer("Studies show that", return_tensors="np").input_ids  # Batch size 1

# preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
# This is not needed for T5ForConditionalGeneration as it does this internally using labels arg.
decoder_input_ids = model._shift_right(Tensor(decoder_input_ids))

# forward pass
outputs = model(input_ids=Tensor(input_ids), decoder_input_ids=decoder_input_ids)
last_hidden_states = outputs[0]
encoder_outputs = outputs[1]
```

#### DeepFloyd/t5-v1_1-xxl

[DeepFloyd/t5-v1_1-xxl](https://huggingface.co/DeepFloyd/t5-v1_1-xxl) is an instance of `T5EncoderModel`, which only has the encoder part.

```python
from mindspore import Tensor
from transformers import AutoTokenizer
from mindone.transformers import T5EncoderModel

tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl", revision="refs/pr/3")
model = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl", revision="refs/pr/3")
input_ids = tokenizer(
     "Studies have been shown that owning a dog is good for you", return_tensors="np"
).input_ids  # Batch size 1
outputs = model(input_ids=Tensor(input_ids))
encoder_outputs = outputs
```

#### google/flan-t5-large

If you already know T5, [google/flan-t5-large](https://huggingface.co/google/flan-t5-large) is just better at everything. For the same number of parameters, these models have been fine-tuned on more than 1000 additional tasks covering also more languages.

```python
from mindspore import Tensor
from transformers import AutoTokenizer
from mindone.transformers import T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")

input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="np").input_ids
labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="np").input_ids
outputs = model(input_ids=Tensor(input_ids), labels=Tensor(labels))
logits = outputs[0]
encoder_outputs = outputs[1]
```

</details>


## Numerical Parity

We compare the numerical parity with the [huggingface/transformer](https://github.com/huggingface/transformers), as detailed below:

<details onclose>

MindSpore 2.2/2.3 @ Ascend **_vs._** Pytorch 2.2 @ CPU(aarch64)

Error Formula: `max(abs(ms-pt)) / mean(abs(pt))`

### MindSpore 2.2.10

- PyNative Mode, FP16

| model                                          | diff   |
|------------------------------------------------|--------|
| openai/clip-vit-large-patch14                  | 0.0385 |
| stabilityai/stable-diffusion-2-1               | 0.0272 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 0.0212 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 0.0233 |
| google-t5/t5-small                             | 0.0488 |
| DeepFloyd/t5-v1_1-xxl                          | 0.0480 |
| google/flan-t5-large                           | 0.0032 |

- PyNative Mode, FP32

| model                                          | diff   |
|------------------------------------------------|--------|
| openai/clip-vit-large-patch14                  | 0.0062 |
| stabilityai/stable-diffusion-2-1               | 0.0053 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 0.0030 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 0.0023 |
| google-t5/t5-small                             | 0.0466 |
| DeepFloyd/t5-v1_1-xxl                          | 0.0349 |
| google/flan-t5-large                           | 0.0009 |

- Graph Mode, FP16

| model                                          | diff   |
|------------------------------------------------|--------|
| openai/clip-vit-large-patch14                  | 0.0385 |
| stabilityai/stable-diffusion-2-1               | 0.0340 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 0.0151 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 0.0208 |
| google-t5/t5-small                             | N.A.   |
| DeepFloyd/t5-v1_1-xxl                          | N.A.   |
| google/flan-t5-large                           | N.A.   |

- Graph Mode, FP32

| model                                          | diff   |
|------------------------------------------------|--------|
| openai/clip-vit-large-patch14                  | 0.0186 |
| stabilityai/stable-diffusion-2-1               | 0.0084 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 0.0059 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 0.0039 |
| google-t5/t5-small                             | N.A.   |
| DeepFloyd/t5-v1_1-xxl                          | N.A.   |
| google/flan-t5-large                           | N.A.   |

### MindSpore 2.3

- PyNative Mode, FP16

| model                                          | diff   |
|------------------------------------------------|--------|
| openai/clip-vit-large-patch14                  | 0.0385 |
| stabilityai/stable-diffusion-2-1               | 0.0272 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 0.0212 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 0.0233 |
| google-t5/t5-small                             | 0.0636 |
| DeepFloyd/t5-v1_1-xxl                          | 0.0426 |
| google/flan-t5-large                           | 0.0025 |

- PyNative Mode, FP32

| model                                          | diff     |
|------------------------------------------------|----------|
| openai/clip-vit-large-patch14                  | 8.30E-05 |
| stabilityai/stable-diffusion-2-1               | 3.28E-05 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 1.55E-05 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 1.38E-05 |
| google-t5/t5-small                             | 6.72E-05 |
| DeepFloyd/t5-v1_1-xxl                          | 1.40E-04 |
| google/flan-t5-large                           | 4.55E-06 |

- Graph Mode, FP16

| model                                          | diff   |
|------------------------------------------------|--------|
| openai/clip-vit-large-patch14                  | 0.0385 |
| stabilityai/stable-diffusion-2-1               | 0.0268 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 0.0151 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 0.0258 |
| google-t5/t5-small                             | 0.0488 |
| DeepFloyd/t5-v1_1-xxl                          | 0.0918 |
| google/flan-t5-large                           | 0.0023 |

- Graph Mode, FP32

| model                                          | diff     |
|------------------------------------------------|----------|
| openai/clip-vit-large-patch14                  | 7.81E-05 |
| stabilityai/stable-diffusion-2-1               | 3.16E-05 |
| stabilityai/stable-diffusion-xl-base-1.0(H)    | 1.72E-05 |
| stabilityai/stable-diffusion-xl-base-1.0(bigG) | 1.75E-05 |
| google-t5/t5-small                             | 4.88E-05 |
| DeepFloyd/t5-v1_1-xxl                          | 9.84E-05 |
| google/flan-t5-large                           | 4.55E-06 |

</details>
