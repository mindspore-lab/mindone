<p align="center">
  <img src="https://raw.githubusercontent.com/VectorSpaceLab/OmniGen2/refs/heads/main/assets/brand.png" width="65%">
</p>

# OmniGen2 (MindSpore)

Efficient MindSpore implementation of [OmniGen2](https://github.com/VectorSpaceLab/OmniGen2): a unified multimodal image
generation and editing framework supporting text-to-image, instruction-guided image editing, and in-context generation.

## Overview

**OmniGen2** is a powerful and efficient generative model. Unlike OmniGen v1, OmniGen2 features two distinct decoding
pathways for text and image modalities, utilizing unshared parameters and a decoupled image tokenizer. OmniGen2 has
competitive performance across four primary capabilities:

- **Visual Understanding**: Inherits the robust ability to interpret and analyze image content from its Qwen-VL-2.5
  foundation.
- **Text-to-Image Generation**: Creates high-fidelity and aesthetically pleasing images from textual prompts.
- **Instruction-guided Image Editing**: Executes complex, instruction-based image modifications with high precision,
  achieving state-of-the-art performance among open-source models.
- **In-context Generation**: A versatile capability to process and flexibly combine diverse inputs—including humans,
  reference objects, and scenes—to produce novel and coherent visual outputs.

## News

- MindSpore inference pipeline and Gradio demo are available under `examples/omnigen2/`.
- Example presets are provided via `configs/app.yaml` and support URL-based images.

## 📦 Requirements

mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |



1) Install MindSpore and Ascend software per the official docs:

    - CANN 8.1.RC1: https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.1.RC1
    - MindSpore: https://www.mindspore.cn/install/

2) Install Python dependencies:

    ```shell
    cd examples/omnigen2
    pip install -r requirements.txt
    ```

## Demo

<details><summary>Text-to-Image</summary>

| 1024x1024                                                                                                                                                                                                                 | 1024x1024                                                                                                                                    |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="320" height="320" src="https://github.com/user-attachments/assets/adaa9673-dc98-4009-9152-26be5715b997" />                                                                                                    | <img width="320" height="320" src="https://github.com/user-attachments/assets/50c95eae-8ea6-4bd1-b35f-e40d9959af90" />                       |
| <details><summary>Caption</summary>The sun rises slightly, the dew on the rose petals in the garden is clear, a crystal ladybug is crawling to the dew, the background is the early morning garden, macro lens.</details> | <details><summary>Caption</summary>A snow maiden with pale translucent skin, frosty white lashes, and a soft expression of longing</details> |
| <img width="320" height="320" src="https://github.com/user-attachments/assets/ad50a62c-27da-401e-a73e-8f362ab98265" />                                                                                                    | <img width="320" height="320" src="https://github.com/user-attachments/assets/40099213-cd45-47fa-9afa-62174dd30027" />                       |
| <details><summary>Caption</summary>This dreamlike digital art captures a vibrant, kaleidoscopic bird in a lush rainforest</details>                                                                                       | <details><summary>Caption</summary>A cat holds a white board writing text "OmniGen2" and a red heart</details>                               |

</details>

<details><summary>Image Editing</summary>

|                                 Prompt                                 |                                                   Input                                                   |                                                  Output                                                   |
|:----------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
|                             Raise his hand                             | <img width="320" src="https://github.com/user-attachments/assets/4abb37c3-66a4-4ba6-ab37-654eaa1eafb0" /> | <img width="320" src="https://github.com/user-attachments/assets/f6c95ecb-48ed-4111-99a6-7fe5be76f665" /> |
|               Convert this image into <br/>Ghibli style                | <img width="320" src="https://github.com/user-attachments/assets/444169dc-0019-4841-b4da-16dc6dd570c3" /> | <img width="320" src="https://github.com/user-attachments/assets/ef17d4ed-ef17-46e4-ac3c-c890ef8f7bd6" /> |
|                       Change the dress to blue.                        | <img width="320" src="https://github.com/user-attachments/assets/29cf608c-745f-4f4f-b4f4-97875089fcc7" /> | <img width="320" src="https://github.com/user-attachments/assets/3e9110df-53fc-4357-a328-8f2d547915dd" /> |
| Generate an anime-style <br/>figurine based on <br/>the original image | <img width="320" src="https://github.com/user-attachments/assets/afb17530-4099-41a9-b9e2-00e92d66c319" /> | <img width="320" src="https://github.com/user-attachments/assets/28b989d9-5fa3-4ec2-abcd-20013d06c8cc" /> |
|                             Make him smile                             | <img width="320" src="https://github.com/user-attachments/assets/8a2e6b1a-9891-4cd2-a2b2-e53db2af987a" /> | <img width="320" src="https://github.com/user-attachments/assets/be7ddf6d-596d-4fd6-af7a-1b9c340425cb" /> |

</details>

<details><summary>In-context Generation</summary>

|                                                                 Prompt                                                                  |                                                  Input 1                                                  |                                                  Input 2                                                  |                                                  Output                                                   |
|:---------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|
|                  In a cozy café, <br/>the anime figure is sitting <br/>in front of a laptop, <br/>smiling confidently.                  | <img width="320" src="https://github.com/user-attachments/assets/6e9929aa-6355-4ced-b66e-635e93388ac5" /> |                                                     -                                                     | <img width="320" src="https://github.com/user-attachments/assets/14eb5bb3-b182-48a0-926d-f8acb8ffe353" /> |
|                                        Let the girl and the boy <br/>get married in the church.                                         | <img width="320" src="https://github.com/user-attachments/assets/87596136-f392-49dc-8db2-faae4c1e5b71" /> | <img width="320" src="https://github.com/user-attachments/assets/6734c2f9-f0ce-492b-a42d-c4ad057095f3" /> | <img width="320" src="https://github.com/user-attachments/assets/080b0979-578f-4d6f-8127-eeaa914989c7" /> |
| The two people shown in the <br/>images are sitting in a theater, <br/>watching the screen. <br/>One person points at the other person. | <img width="320" src="https://github.com/user-attachments/assets/9a19ffb6-dc68-4926-b633-6e85dd7dfcb7" /> | <img width="320" src="https://github.com/user-attachments/assets/0923be1b-d087-4141-8a02-87ff3217e96b" /> | <img width="320" src="https://github.com/user-attachments/assets/072c1d3d-4845-4c9d-9f23-cd1e112452cc" /> |
|                        Replace the woman in <br/>the second image <br/>with the woman from <br/>the first image                         | <img width="320" src="https://github.com/user-attachments/assets/b5b768c2-d195-4971-abbe-0eeac3cff019" /> | <img width="320" src="https://github.com/user-attachments/assets/4c827856-bf5b-45c9-90cd-8a858bfdb6e1" /> | <img width="320" src="https://github.com/user-attachments/assets/a03f8a2f-c45a-4a0e-895c-6dbdb6d5e1a2" /> |

</details>

<details><summary>Visual Understanding</summary>

| Input                                                                                                     | Prompt                              | Output                                                                                                                                                                                                                                                                                                        |
|-----------------------------------------------------------------------------------------------------------|-------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <img width="320" src="https://github.com/user-attachments/assets/c69c9b7e-1588-4ebc-94fb-696976dc6768" /> | Please briefly describe this image. | The image is a close-up of an anime character with light-colored hair and blue eyes. The character is wearing a green suit with a red tie and a white shirt. The background appears to be an indoor setting, possibly a room with wooden paneling. The character's expression is neutral or slightly serious. |
| <img width="320" src="https://github.com/user-attachments/assets/12862a6c-77c7-4ea0-86bf-5e54a1a99752" /> | Please describe this image briefly. | The image shows a plush toy bear sitting on a grassy surface. The bear has a brown body with white paws and a white muzzle. It is wearing a blue bow on its head and a white bib with the text "Get Well" written on it. The background consists of green grass with some small plants and clover.            |

</details>

## Model Weights

OmniGen2 weights and assets are hosted on Hugging Face.

```shell
hf download OmniGen2/OmniGen2 --exclude "assets/*"
```

> [!TIP]
> For users in Mainland China set the `HF_ENDPOINT=https://hf-mirror.com` environment variable.

## Inference

### Command Line

Usage examples are available under `scripts/run`. These scripts provide ready-to-use commands for common inference
scenarios and demonstrate various OmniGen2 capabilities.

For full list of flags, see `python scripts/inference.py --help`.

#### Speedup Inference with Caching

- For TeaCache (~30% speedup at default threshold), add the following flags:

```shell
--enable_teacache --teacache_rel_l1_thresh 0.05
```

- For TaylorSeer (up to ~2× speedup, mutually exclusive with TeaCache):

```shell
--enable_taylorseer
```

### Gradio App

A local demo UI is available at `app.py`.

```shell
pip install gradio
python app.py
```

## Usage Tips

To achieve optimal results with OmniGen2, you can adjust the following key hyperparameters based on your specific use
case.

- **Guidance scales**
    - `text_guidance_scale`: stronger adherence to text (default 5.0)
    - `image_guidance_scale`: stronger adherence to input images (edit/in-context). Try 1.2–2.0 for editing; 2.5–3.0 for
      in-context.
- **Scheduler**: `euler` (default) or `dpmsolver++` for potentially fewer steps at similar quality.
- **CFG range**: Lower `--cfg_range_end` can reduce latency with minor quality impact.
- **Prompts**: Be specific. English prompts work best currently. Longer, descriptive prompts often help.
- **Inputs**: Prefer clear images ≥ 512×512.

## Training

### 1. Preparation

Before launching the training, you need to prepare the following configuration files.

#### Step 1: Set Up the Training Configuration

This is a YAML file that specifies crucial parameters for your training job, including the model architecture,
optimizer, dataset paths, and validation settings.

We provide two templates to get you started:

* **Full-Parameter Fine-Tuning:** `configs/finetune/ft.yml`
* **LoRA Fine-Tuning:** `configs/finetune/ft_lora.yml`

Copy one of these templates and modify it according to your needs. Below are some of the most important parameters you
may want to adjust:

- `name`: The experiment name. This is used to create a directory for logs and saved model weights (e.g.,
  `experiments/your_exp_name`).
- `data.config_path`: Path to the data configuration file that defines your training data sources and mixing ratios.
- `data.max_output_pixels`: The maximum number of pixels for an output image. Larger images will be downsampled while
  maintaining their aspect ratio.
- `data.max_input_pixels`: A list specifying the maximum pixel count for input images, corresponding to one, two, three,
  or more inputs.
- `data.max_side_length`: The maximum side length for any image (input or output). Images exceeding this will be
  downsampled while maintaining their aspect ratio.
- `dataloader.batch_size`: The batch size per NPU.
- `train.steps`: The total number of training steps to run.
- `train.lr_scheduler.lr`: The learning rate for the optimizer. **Note:** This often requires tuning based on your
  dataset size and whether you are using LoRA. We recommend using lower learning rate for full-parameter fine-tuning.

#### Step 2: Configure Your Dataset

The data configuration consists of a set of `yaml` and `jsonl` files.

* The `.yml` file defines the mixing ratios for different data sources.
* The `.jsonl` files contain the actual data entries, with each line representing a single data sample.

For a practical example, please refer to `configs/finetune/data/mix.yml`.
Each line in a `.jsonl` file describes a sample, generally following this format:

```json
{
  "task_type": "edit",
  "instruction": "add a hat to the person",
  "input_images": [
    "/path/to/your/data/edit/input1.png",
    "/path/to/your/data/edit/input2.png"
  ],
  "output_image": "/path/to/your/data/edit/output.png"
}
```

*Note: The `input_images` field can be omitted for text-to-image (T2I) tasks.*

### 2. 🚀 Launching the Training

Once your configuration is ready, you can launch the training script. All experiment artifacts, including logs and
checkpoints, will be saved in `experiments/${experiment_name}`.

We provide convenient shell scripts to handle the complexities of launching distributed training jobs. You can use them
directly or adapt them for your environment.

* **For Full-Parameter Fine-Tuning:** `scripts/run/ft.sh`
* **For LoRA Fine-Tuning:** `scripts/run/ft_lora.sh`

> **⚠️ Note on LoRA Checkpoints:**
> Currently, when training with LoRA, the script saves the entire model's parameters (including the frozen base model
> weights) in the checkpoint. This is due to a limitation in easily extracting only the LoRA-related parameters when
> using FSDP.

## Performance

### Inference

|  Model   |         Mode          | Cards | Precision | Number of<br/>input images | Resolution | Scheduler | Steps | s/img |
|:--------:|:---------------------:|:-----:|:---------:|:--------------------------:|:----------:|:---------:|:-----:|:-----:|
| OmniGen2 |     Text-to-Image     |   1   |   BF16    |             -              | 1024x1024  |   Euler   |  50   |  120  |
| OmniGen2 |     Image Editing     |   1   |   BF16    |             1              |  832x1248  |   Euler   |  50   |  282  |
| OmniGen2 | In-context Generation |   1   |   BF16    |             1              |  768x1152  |   Euler   |  50   |  248  |
| OmniGen2 | In-context Generation |   1   |   BF16    |             2              | 1024x1024  |   Euler   |  50   |  870  |

### Training

|  Model   | Fine-tuning | Cards | Batch size | Resolution | Precision | s/step |                   Recipe                    |
|:--------:|:-----------:|:-----:|:----------:|:----------:|:---------:|:------:|:-------------------------------------------:|
| OmniGen2 |    Full     |   8   |     1      |  720x720   |   BF16    |  5.03  |      [ft.yml](configs/finetune/ft.yml)      |
| OmniGen2 |    LoRA     |   8   |     1      |  720x720   |   BF16    |  3.78  | [ft_lora.yml](configs/finetune/ft_lora.yml) |

## Acknowledgement

If you find OmniGen2 useful, please cite the original work:

```bibtex
@article{wu2025omnigen2,
  title={OmniGen2: Exploration to Advanced Multimodal Generation},
  author={Chenyuan Wu and Pengfei Zheng and Ruiran Yan and Shitao Xiao and Xin Luo and Yueze Wang and Wanli Li and Xiyan Jiang and Yexin Liu and Junjie Zhou and Ze Liu and Ziyi Xia and Chaofan Li and Haoge Deng and Jiahao Wang and Kun Luo and Bo Zhang and Defu Lian and Xinlong Wang and Zhongyuan Wang and Tiejun Huang and Zheng Liu},
  journal={arXiv preprint arXiv:2506.18871},
  year={2025}
}
```
