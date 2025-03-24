<!-- Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLAllegro

The 3D variational autoencoder (VAE) model with KL loss used in [Allegro](https://github.com/rhymes-ai/Allegro) was introduced in [Allegro: Open the Black Box of Commercial-Level Video Generation Model](https://huggingface.co/papers/2410.15458) by RhymesAI.

The model can be loaded with the following code snippet.

```python
from mindone.diffusers import AutoencoderKLAllegro

vae = AutoencoderKLCogVideoX.from_pretrained("rhymes-ai/Allegro", subfolder="vae", mindspore_dtype=ms.float32)
```

::: mindone.diffusers.AutoencoderKLAllegro

::: mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKLOutput

::: mindone.diffusers.models.autoencoders.vae.DecoderOutput
