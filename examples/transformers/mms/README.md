# MMS

## Introduction

The MMS model was proposed in [Scaling Speech Technology to 1,000+ Languages](https://huggingface.co/papers/2305.13516) by Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, Michael Auli

The abstract from the paper is the following:

_Expanding the language coverage of speech technology has the potential to improve access to information for many more people. However, current speech technology is restricted to about one hundred languages which is a small fraction of the over 7,000 languages spoken around the world. The Massively Multilingual Speech (MMS) project increases the number of supported languages by 10-40x, depending on the task. The main ingredients are a new dataset based on readings of publicly available religious texts and effectively leveraging self-supervised learning. We built pre-trained wav2vec 2.0 models covering 1,406 languages, a single multilingual automatic speech recognition model for 1,107 languages, speech synthesis models for the same number of languages, as well as a language identification model for 4,017 languages. Experiments show that our multilingual speech recognition model more than halves the word error rate of Whisper on 54 languages of the FLEURS benchmark while being trained on a small fraction of the labeled data._

## Examples

Here are some example usages:

### Automatic Speech Recognition (ASR)

```python
from datasets import load_dataset, Audio
from transformers import AutoProcessor
from mindone.transformers import Wav2Vec2ForCTC
import mindspore as ms

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

model_id = "facebook/mms-1b-all"

processor = AutoProcessor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(model_id)

inputs = processor(en_sample, sampling_rate=16_000, return_tensors="np")
inputs = {k: ms.tensor(v) for k, v in inputs.items()}

outputs = model(**inputs).logits

ids = ms.mint.argmax(outputs, dim=-1)[0]
transcription = processor.decode(ids)
# 'joe keton disapproved of films and buster also had reservations about the media'
```

### Speech Synthesis (TTS)

```python
import mindspore as ms
from transformers import VitsTokenizer
from mindone.transformers import VitsModel
import scipy

tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng")

inputs = tokenizer(text="Hello - my dog is cute", return_tensors="np")
inputs = {k: ms.tensor(v) for k, v in inputs.items()}

ms.set_seed(555)  # make deterministic
outputs = model(**inputs)

waveform = outputs[0][0]
scipy.io.wavfile.write("synthesized_speech.wav", rate=model.config.sampling_rate, data=waveform.numpy())
```

### Language Identification (LID)

```python
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor
from mindone.transformers import Wav2Vec2ForSequenceClassification
import mindspore as ms

# English
stream_data = load_dataset("mozilla-foundation/common_voice_13_0", "en", split="test", streaming=True)
stream_data = stream_data.cast_column("audio", Audio(sampling_rate=16000))
en_sample = next(iter(stream_data))["audio"]["array"]

model_id = "facebook/mms-lid-126"

processor = AutoFeatureExtractor.from_pretrained(model_id)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_id)

inputs = processor(en_sample, sampling_rate=16_000, return_tensors="np")
inputs = {k: ms.tensor(v) for k, v in inputs.items()}

outputs = model(**inputs).logits

lang_id = ms.mint.argmax(outputs, dim=-1)[0].item()
detected_lang = model.config.id2label[lang_id]
# 'eng'
```
