# myt5

## Overview

The myt5 model was proposed in [MYTE: Morphology-Driven Byte Encoding for Better and Fairer Multilingual Language Modeling](https://arxiv.org/pdf/2403.10691.pdf) by Tomasz Limisiewicz, Terra Blevins, Hila Gonen, Orevaoghene Ahia, and Luke Zettlemoyer.
MyT5 (**My**te **T5**) is a multilingual language model based on T5 architecture.
The model uses a **m**orphologically-driven **byte** (**MYTE**) representation described in our paper.
**MYTE** uses codepoints corresponding to morphemes in contrast to characters used in UTF-8 encoding.
As a pre-requisite, we used unsupervised morphological segmentation ([Morfessor](https://aclanthology.org/E14-2006.pdf)) to obtain morpheme inventories for 99 languages.
However, the morphological segmentation step is not needed when using the pre-defined morpheme inventory from the hub (see: [Tomli/myt5-base](https://huggingface.co/Tomlim/myt5-base)).

The abstract from the paper is the following:

*A major consideration in multilingual language modeling is how to best represent languages with diverse vocabularies and scripts. Although contemporary text encoding methods cover most of the world’s writing systems, they exhibit bias towards the high-resource languages of the Global West. As a result, texts of underrepresented languages tend to be segmented into long sequences of linguistically meaningless units. To address the disparities, we introduce a new paradigm that encodes the same information with segments of consistent size across diverse languages. Our encoding convention (MYTE) is based on morphemes, as their inventories are more balanced across languages than characters, which are used in previous methods. We show that MYTE produces shorter encodings for all 99 analyzed languages, with the most notable improvements for non-European languages and non-Latin scripts. This, in turn, improves multilingual LM performance and diminishes the perplexity gap throughout diverse languages.*

This model was contributed by [Tomasz Limisiewicz](https://huggingface.co/Tomlim).
The original code can be found [here](https://github.com/tomlimi/MYTE).

## Examples

Here are some example usages:

```python
import mindspore as ms
from mindspore import mint
from transformers import MyT5Tokenizer
from mindone.transformers import T5ForConditionalGeneration

MODEL_SIZE = "large" # small, base, or large

model = T5ForConditionalGeneration.from_pretrained(f"Tomlim/MyT5_{MODEL_SIZE}", use_safetensors=True)
tokenizer = MyT5Tokenizer.from_pretrained(f"Tomlim/MyT5_{MODEL_SIZE}")

pre_texts = ['"We now have',
            '„Mamy teraz myszy w wieku',
            '"""எங்களிடம் இப்போது']
post_texts = ['4-month-old mice that are non-diabetic that used to be diabetic," he added.',
              '4 miesięcy, które miały cukrzycę, ale zostały z niej wyleczone” – dodał.',
              '4-மாத-வயதுடைய எலி ஒன்று உள்ளது, முன்னர் அதற்கு நீரிழிவு இருந்தது தற்போது இல்லை"" என்று அவர் மேலும் கூறினார்."']

inputs = tokenizer(pre_texts, padding="longest", return_tensors="np")
targets = tokenizer(post_texts, padding="longest", return_tensors="np")
for k, v in inputs.items():
    inputs[k] = ms.tensor(v)
for k, v in targets.items():
    targets[k] = ms.tensor(v)

outputs = model(**inputs, labels=targets.input_ids.to(ms.int32))
probs = mint.nn.functional.softmax(outputs.logits, dim=-1)
```
