<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Overview

🤗 Diffusers provides a collection of training scripts for you to train your own diffusion models. You can find all of our training scripts in [diffusers/examples](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers).

Each training script is:

- **Self-contained**: the training script does not depend on any local files, and all packages required to run the script are installed from the `requirements.txt` file.
- **Easy-to-tweak**: the training scripts are an example of how to train a diffusion model for a specific task and won't work out-of-the-box for every training scenario. You'll likely need to adapt the training script for your specific use-case. To help you with that, we've fully exposed the data preprocessing code and the training loop so you can modify it for your own use.
- **Beginner-friendly**: the training scripts are designed to be beginner-friendly and easy to understand, rather than including the latest state-of-the-art methods to get the best and most competitive results. Any training methods we consider too complex are purposefully left out.
- **Single-purpose**: each training script is expressly designed for only one task to keep it readable and understandable.

Our current collection of training scripts include:

| Training | SDXL-support | LoRA-support |
|---|---|---|
| [unconditional image generation](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers/unconditional_image_generation) |  |  |
| [text-to-image](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers/text_to_image) | 👍 | 👍 |
| [textual inversion](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers/textual_inversion) |  |  |
| [DreamBooth](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers/dreambooth) | 👍 | 👍 |
| [ControlNet](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers/controlnet) | 👍 |  |

These examples are **actively** maintained, so please feel free to open an issue if they aren't working as expected. If you feel like another training example should be included, you're more than welcome to start a [Feature Request](https://github.com/mindspore-lab/mindone/issues/new?assignees=zhanghuiyao&labels=rfc&projects=&template=feature_request.md&title=) to discuss your feature idea with us and whether it meets our criteria of being self-contained, easy-to-tweak, beginner-friendly, and single-purpose.

## Install

Make sure you can successfully run the latest versions of the example scripts by installing the library from source in a new virtual environment:

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
pip install .
```

Then navigate to the folder of the training script (for example, [DreamBooth](https://github.com/mindspore-lab/mindone/tree/master/examples/diffusers/dreambooth)) and install the `requirements.txt` file. Some training scripts have a specific requirement file for SDXL, LoRA or Flax. If you're using one of these scripts, make sure you install its corresponding requirements file.

```bash
cd examples/diffusers/dreambooth
pip install -r requirements_sd3.txt
```
