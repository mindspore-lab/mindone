<!-- Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License. -->

# AutoencoderKLHunyuanImage

The 2D variational autoencoder (VAE) model with KL loss used in [HunyuanImage2.1].

The model can be loaded with the following code snippet.

```python
from mindone.diffusers import AutoencoderKLHunyuanImage

vae = AutoencoderKLHunyuanImage.from_pretrained("hunyuanvideo-community/HunyuanImage-2.1-Diffusers", subfolder="vae", mindspore_dtype=ms.bfloat16)
```

::: mindone.diffusers.AutoencoderKLHunyuanImage

::: mindone.diffusers.models.autoencoders.vae.DecoderOutput
