<h1 align="center">CannyEdit: Easy multitask image editing</h1>

<p align="center">
    <a href="https://vaynexie.github.io/CannyEdit/">
        <img alt="Build" src="https://img.shields.io/badge/Project%20Page-CannyEdit-yellow">
    </a>
    <a href="https://arxiv.org/abs/2508.06937">
         <img alt="Build" src="https://img.shields.io/badge/arXiv%20paper-2508.06397-b31b1b.svg">
    </a>
</p>

# Overview

This is the official MindSpore implementation of [CannyEdit](https://vaynexie.github.io/CannyEdit/).

CannyEdit is a novel training-free framework to support multitask image editing. It enables high-quality region-specific image edits, especially useful in cases where SOTA free-form image editing methods fail to ground edits accurately. Besides, it can support edits on multiple user-specific regions at one generation pass when multiple masks are given.

<p align="center">
   <img src=./assets/page_imgs/grid_image.png width=500 />
</p>
<p align="center">
  <em> Figure 1. Examples of CannyEdit </em>
</p>

## 📦 Requirements

mindspore  |  ascend driver   |cann  |
|:--:|:--:|:--:|
| >=2.6.0    | >=24.1.RC1 |   >=8.1.RC1 |



Install requirements
```shell
pip install mindone==0.4.0
pip install -r requirements.txt
```
Try `python -c "import mindone"`. If no error occurs, the installation is successful.

## 🚀 Quick Start
The pipeline of using CannyEdit consists of 3 steps:
1. Generate masks (Optional. Skipped if you have)
2. Generate prompts (Optional. Skipped if you have)
3. Generate edited image

### Step 1: Generate masks (Optional)
At first, the step needs model weights of [SAM2](https://github.com/facebookresearch/sam2/). Please download it using tools in `examples/sam2`.
```bash
cd examples/sam2/checkpoints && \
./download_ckpts.sh &&
```
And the checkpoints will be downloaded into examples/sam2/checkpoints.

Then, modify the path of checkpoint in the script file below. And run the shell script to launch the app of mask generator.
```bash
cd examples/canny_edit && \
bash run_app_mask.sh
```
Then open the address of http://localhost:5000. If you use browser remotely, you can set on your remote machine as below:
```bash
ssh -L 8081:localhost:5000 username@ip
```
According to the mapping, just open the address of http://localhost:8081 on your remote machine.

In the webpage of mask generator, choose specific method for corresponding editing task.

- Adding task: Circle a target area where you want to add an object or person. Then click "Generate Ellipse Mask"
- Replace and removal tasks: Draw a line on a certain area of an existing object or person. Then click "Generate SAM Mask"

### Step 2: Generate prompts (Optional)
In main.py, it will check if there is not source prompt for input image or target prompt for edited image. It will call Visual Language Model (VLM) to generate related prompts. Here we use [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).

### Step 3: Generate edited image
There are several examples listed in run_infer.sh. Just uncomment one of them to generate corresponding case.
```bash
bash run_infer.sh
```
Here are examples of output for each test case

- **case 1: Replace background with mountains**
```bash
python main.py \
  --image_path './assets/imgs/girl33.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/' \
  --prompt_local "A mountain." \
  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
  --prompt_target "A young girl with red hair smiles brightly, wearing a red and white checkered shirt, sitting on a bench with mountains in the background." \
  --mask_path "./assets/mask_temp/mask_209_inverse.png"
```


<p align="center">
  <em> From left to right, these are original image, mask image, and generated edited image. </em>
</p>

- **case 2: Replace the girl with a boy**
```bash
python main.py \
  --image_path './assets/imgs/girl33.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/' \
  --prompt_local "A boy smiling." \
  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
  --prompt_target "A young boy with red hair smiles brightly, wearing a red and white checkered shirt." \
  --mask_path "./assets/mask_temp/mask_208.png"
```
<p align="center">
  <em> From left to right, these are original image, mask image, and generated edited image. </em>
</p>

- **case 3: Add a monkey**
```bash
python main.py \
--image_path './assets/imgs/girl33.jpeg' \
--image_whratio_unchange \
--save_folder './results/' \
--prompt_local "A monkey playing." \
--prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
--prompt_target "A young girl with red hair smiles brightly, wearing a red and white checkered shirt, a monkey playing nearby." \
--mask_path "./assets/mask_temp/mask_213.png"
```
<p align="center">
  <em> From left to right, these are original image, mask image, and generated edited image. </em>
</p>

- **case 4: Remove the girl**
```bash
python main.py \
  --image_path './assets/imgs/girl33.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/' \
  --prompt_local '[remove]' \
  --mask_path "./assets/mask_temp/mask_208.png" \
  --dilate_mask
```
<p align="center">
  <em> From left to right, these are original image, mask image, and generated edited image. </em>
</p>

- **case 5: Replace the girl with a boy + add a monkey**
```bash
python main.py \
  --image_path './assets/imgs/girl33.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/' \
  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
  --prompt_local "A boy smiling." \
  --prompt_local "A monkey playing." \
  --mask_path "./assets/mask_temp/mask_208.png" \
  --mask_path "./assets/mask_temp/mask_215.png" \
  --prompt_target "A young boy wearing a red and white checkered shirt, a monkey playing nearby."
```
<p align="center">
  <em> From left to right, these are original image, two mask images, and generated edited image. </em>
</p>


## Performance


Experiments are tested on Ascend Atlas 800T A2 machines with pyantive mode.

- mindspore 2.7.0

| model      | cards | resolution | task           | steps | s/Step       | s/Image       |
|------------|-------|------------|----------------|-------|--------------|---------------|
| CannyEdit  | 1     | 768x768    | Replace        | 50    | 6.12         | 306           |
| CannyEdit  | 1     | 768x768    | Add            | 50    | 1.96         | 98            |
| CannyEdit  | 1     | 768x768    | Removal        | 50    | 6.6          | 330           |
| CannyEdit  | 1     | 768x768    | Replace + Add  | 50    | 5.7          | 285           |

## Acknowledgement
The codebase is modified based on [x-flux](https://github.com/XLabs-AI/x-flux).
