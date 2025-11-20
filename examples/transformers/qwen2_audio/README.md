# Qwen2-Audio: Multimodal Audio-Language Model

[Qwen2-Audio](https://huggingface.co/Qwen/Qwen2-Audio-7B) is an advanced multimodal model that combines audio understanding with natural language processing capabilities. It can process audio inputs and generate textual responses, enabling applications like speech recognition, audio captioning, and audio-based question answering.

## Introduction

Qwen2-Audio is designed to bridge the gap between audio processing and language understanding. The model can handle various audio tasks including speech transcription, audio event detection, music understanding, and audio-based conversational interactions.

## Get Started

## 📦 Requirements
mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |




## Quick Start

### Audio Captioning and Understanding

```python
import librosa
import numpy as np
from transformers import AutoProcessor
import mindspore as ms
from mindone.transformers import Qwen2AudioForConditionalGeneration

# Model configuration
model_name = "Qwen/Qwen2-Audio-7B"
dtype_name = ms.bfloat16

# Load model
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    model_name,
    tie_word_embeddings=False,
    attn_implementation="eager",
    mindspore_dtype=dtype_name,
)

# Load processor
processor = AutoProcessor.from_pretrained(model_name)

# Prepare audio
audio_path = "/path/to/audio.wav"  # Replace with your audio file path
audio, sample_rate = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)

# Create prompt for different tasks
prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"

# Process inputs
inputs = processor(text=prompt, audios=audio, return_tensors="np")

# Convert to MindSpore tensors
for key, value in inputs.items():
    tensor = ms.Tensor(value)
    if tensor.dtype == ms.int64:
        tensor = tensor.to(ms.int32)
    elif tensor.dtype != ms.bool_:
        tensor = tensor.to(model.dtype)
    inputs[key] = tensor

# Generate response
output = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

# Decode response
response_ids = output[0][inputs["input_ids"].shape[1]:]
response = processor.decode(response_ids, skip_special_tokens=True)
print(f"Audio description: {response}")
```

### Audio Question Answering

```python
# For question answering about audio content
qa_prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>What instruments can you hear in this audio?"
qa_inputs = processor(text=qa_prompt, audios=audio, return_tensors="np")

# Convert and generate
for key, value in qa_inputs.items():
    tensor = ms.Tensor(value)
    qa_inputs[key] = tensor.astype(ms.int32) if tensor.dtype == ms.int64 else tensor.astype(model.dtype)

qa_output = model.generate(**qa_inputs, max_new_tokens=50)
qa_response_ids = qa_output[0][qa_inputs["input_ids"].shape[1]:]
qa_response = processor.decode(qa_response_ids, skip_special_tokens=True)
print(f"Question answering: {qa_response}")
```

### Run the Example

```bash
# Make sure you have an audio file and update the path in the script
python qwen2_audio_generate.py
```

## Features

- **Audio Understanding**: Processes various audio formats and content types
- **Speech Recognition**: Transcribes speech to text accurately
- **Audio Captioning**: Generates natural language descriptions of audio content
- **Question Answering**: Answers questions about audio content
- **Multilingual Support**: Handles multiple languages in audio processing
- **Event Detection**: Identifies and describes audio events and sound sources

## Model Architecture

Qwen2-Audio combines:
- **Audio Encoder**: Advanced audio feature extraction using wav2vec-style architecture
- **Language Model**: Qwen2 text generation capabilities
- **Cross-modal Fusion**: Integration of audio and text representations
- **Sequence Processing**: Handles variable-length audio inputs effectively

## Performance

Qwen2-Audio achieves strong performance on audio understanding tasks:

- **Audio Captioning**: Generates detailed and accurate audio descriptions
- **Speech Transcription**: High accuracy speech-to-text conversion
- **Audio QA**: Effective question answering about audio content
- **Event Classification**: Accurate identification of audio events
- **Multilingual Audio**: Strong performance across different languages

## Use Cases

- **Audio Description**: Generating descriptions for visually impaired users
- **Content Moderation**: Analyzing audio content for appropriate content
- **Podcast Analysis**: Understanding and summarizing podcast content
- **Music Information Retrieval**: Analyzing musical content and metadata
- **Environmental Monitoring**: Detecting and classifying environmental sounds
- **Meeting Transcription**: Transcribing and summarizing meetings
- **Accessibility Services**: Making audio content accessible to all users
- **Research Applications**: Audio analysis and understanding research</contents>
</xai:function_call
<xai:function_call name="run_terminal_cmd">
<parameter name="command">cd /Users/weizheng/work/tmp/mindone/examples/transformers && ls -la */README.md | head -20
