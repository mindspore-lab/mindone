# Video-Text Data Curation Pipeline

<img width="850" alt="data_pipeline_demo" src="https://github.com/user-attachments/assets/a7828e69-59b9-49f2-b762-a50ac5bd86a1" />

## Overview
This pipeline is designed to gather video-text pairs to train video generation models
based on text inputs.

First, raw videos — whether sourced from the internet or public
datasets — are divided into shorter clips using scene detection
techniques. We offer an optional filtering mechanism to select
specific video categories of interest. Following this, we incorporate
`imagededup` package to remove duplicate or near duplicate videos from the dataset.

Next, these videos undergo an evaluation process where multiple
scores are predicted using existing models. These scores include
aesthetic scoring, OCR (Optical Character Recognition) for text
detection, LPIPS scoring to assess motion, and a safety
checker to determine if the video is appropriate.
Only videos that meet satisfactory evaluation criteria advance
to the captioning step.

After captioning, a matching score is calculated to assess the
alignment between video and text. Samples with low matching scores
are filtered out.

In summary, our pipeline generates video-text pairs that exhibit
high aesthetic quality, significant video motion, and strong
semantic consistency.

## Example Workflow

### Configuration

The pipeline is configured using a `config.yaml` or `config.json`
file located in the `config/` directory. This file allows you to
specify paths, enable or disable pipeline steps, and set parameters
for each processing stage.

#### Set Root Paths
In `config.yaml`, modify the following paths to match your
directory structure:

```bash
paths:
  ROOT_VIDEO: "/path/to/video/folder" # Directory containing the original video files.
  ROOT_CLIPS: "/path/to/video/clips/folder" # Directory where video clips will be stored.
  ROOT_META: "/path/to/meta/folder" # Directory for metadata CSV files.
```

Similarly, modify the following if using `config.json`:

```bash
"paths": {
    "ROOT_VIDEO": "/path/to/video/folder",
    "ROOT_CLIPS": "/path/to/video/clips/folder",
    "ROOT_META": "/path/to/meta/folder",
    "PYTHONPATH": "$(pwd)" # no need to modify this
  },
```

#### Deduplication Setup
If you need to perform deduplication, run the following command:
```bash
python pipeline/datasets/imagededup/setup.py build_ext --inplace
```

#### Scoring Model Setup
You may need to first download pretrained models if aesthetic
scoring, OCR, LPIPS scoring, or NSFW filtering is needed. Refer
to the **Scoring** section below.

#### Captioning Model Setup
Please refer to the **Captioners** section below.

#### Customize Pipeline Steps
Enable or disable specific pipeline steps via setting `run`
and adjust their parameters as needed. We recommend keeping
the default for most parts. If you are interested in option
filtering, you may set `run` under `option_filtering` to `true`
and provide the video type you would like to keep under `option`.

### Usage

#### Via Config and Runner

We support using json or yaml config files. After setting the `config.yaml` or `config.json` file, run the entire pipeline via
```bash
python -m script.pipeline_runner ./config/config.yaml
```

OR

```bash
python -m script.pipeline_runner ./config/config.json
```

You will get the processed csv file containing metadata information
after running the pipeline. We also store the intermediate csv files
in between during processing.

## Scoring

### Aesthetic Scoring

To evaluate the aesthetic quality of videos, we use the
scoring model from [CLIP+MLP Aesthetic Score Predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor).
This model is trained on 176K SAC (Simulacra Aesthetic
Captions) pairs, 15K LAION-Logos (Logos) pairs, and
250K AVA (The Aesthetic Visual Analysis) image-text pairs.

The aesthetic score is between 1 and 10. Empirically, we
find that an aesthetic score above 4.5 can be considered
as fair.

For videos, we extract the first, last, and the middle
frames for evaluation.

For usage, first download the scoring model using the following command to `./pretrained_models/aesthetic.pth`.

```bash
wget https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth -O pretrained_models/aesthetic.pth
```

Use the following script to convert `.pth` to `.ckpt` for use in MindSpore.

```bash
python -m tools.pth_to_ckpt --model aesthetic \
 --pth_path 'pretrained_models/aesthetic.pth' \
 --save_path 'pretrained_models/aesthetic.ckpt' \
 --show_pth --show_ckpt --convert --value
```

Then, run the following command if using CPU. **Make sure** the meta file has column `path` (path to the sample).
```bash
python -m scoring.aesthetic.inference /path/to/meta.csv --use_cpu
```
If running on Ascend, you may use
```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/aesthetic/inference.py \
 /path/to/meta.csv
```
Modify `worker_num` and `local_worker_num` based on your resource.

This should output `/path/to/meta_aes.csv` with column `aes`.

### Matching Scoring

Matching scores are calculated to evaluate the alignment between an image/video and its caption.
Here, we use the [CLIP](https://github.com/openai/CLIP) model, which is trained on image-text pairs.
For videos, we extract the first, last, and the middle frame and compare it with the caption.
We record the highest score among the three as the matching score.

For usage, run the following command if using CPU. **Make sure** the meta file has the column `path` (path to the sample).
For matching scores for captions, the meta file should also have the column `text` (caption of the sample).
For option filtering, the argument `--option` must be provided
```bash
# for option filtering
python -m pipeline.scoring.matching.inference /path/to/meta.csv --use_cpu --option animal
```
If running on Ascend, you may use
```bash
export PYTHONPATH=$(pwd)
# calculate the matching scores with captions, the column `text` must be present
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/matching/inference.py \
 /path/to/meta.csv
```
Modify `worker_num` and `local_worker_num` based on your resource.

This should output `/path/to/meta_match.csv` with column `match`. Higher matching scores indicate better image-text/video-text alignment.
Empirically, a match score higher than 20 can be considered
as fair.

### OCR
OCR (Optical Character Recognition) is used to detect and recognize
text in images and video frames. We utilize **MindOCR** package for
this task, which supports a variety of state-of-the-art OCR
algorithms. MindOCR supports both detection and recognition of text
in natural scenes. By default, we use DB++ for detection and
CRNN for recognition. You can check the [MindOCR](https://github.com/mindspore-lab/mindocr/tree/main/tools/infer/text)
page for the full list.

First, download the DB++ model [here](https://download.mindspore.cn/toolkits/mindocr/dbnet/dbnetpp_resnet50_ch_en_general-884ba5b9.ckpt).
By default, you can put them in the folder
`./pretrained_models/` and you may rename the model as `dbnetpp.ckpt`.

Run the following command for inference. **Make sure** the meta file has the column `path` (path to the sample).

```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/ocr/inference.py \
 /path/to/meta.csv \
 --num_boxes \
 --max_single_percentage \
 --total_text_percentage
```

Note that you must have the `height` and `width` information available in the csv
file to use the option `max_single_percentage` or
`total_text_percentage`. You may get these information by running

```bash
python -m pipeline.datasets.datautil /path/to/meta.csv --info
```

You can find the results in the `ocr` column of the csv file. It
will be stored in the following format:
```angular2html
[{"transcription": "canada", "points": [[430, 148], [540, 148], [540, 171], [430, 171]]}, ...]
```

We may also have the following columns depending on the options enabled:

`num_boxes`: Total number of detected text boxes.

`max_single_percentage`: Maximum area percentage occupied by a single text box.

`total_text_percentage`: Total area percentage occupied by all text boxes.

### LPIPS Scoring (Motion Analysis)
LPIPS (Learned Perceptual Image Patch Similarity) is a
metric used to measure perceptual similarity between
images. In the context of videos, we use LPIPS to
quantify the perceptual changes between consecutive
frames, which can be an indicator of motion smoothness
and quality.

The LPIPS score is computed by extracting frames from the
video at regular intervals (default every 1 second) and
calculating the perceptual similarity between consecutive
frames.

The scores are weighted by the time difference between
frames to account for variable frame intervals due to
extraction issues.

If only one or no frame is extracted from a video,
a score of -1 is assigned, indicating insufficient
data to compute LPIPS.

To calculate the LPIPS score, first download the [LPIPS model](https://download-mindspore.osinfra.cn/toolkits/mindone/autoencoders/lpips_vgg-426bf45c.ckpt).
By default, you can put it in the folder `./pretrained_models/` and you may rename the model as `lpips.ckpt`.

Then, run the following command if using CPU. **Make sure** the meta file has the column
`path` (path to the sample).

```bash
python -m scoring.lpips.inference /path/to/meta.csv
```

If running on Ascend, you may use

```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/lpips/inference.py \
 /path/to/meta.csv
```

This should output `/path/to/meta_lpips.csv` with column `lpips`.

### NSFW

NSFW (Not Safe for Work) is a metric used to identify
restricted content in images. In the context of videos,
we extract frames and feed them into a trained NSFW
classifier to determine if frames contains content
that may be unsafe or inappropriate.

We use the same NSFW classifier as the one used in
stable diffusion 2.x. The classifier is trained via
a supervised approach, taking image features from
CLIP as input. The output is a number from 0 to 1,
representing the probability of the generated image
being NSFW. If the probability exceeds a certain
threshold (default 0.2), the image is considered NSFW.

To get started, first download the [NSFW model](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/safety_checker/l14_nsfw-c7c99ae7.ckpt).
By default, you can put it in the folder `./pretrained_models/` and you may rename the model as `nsfw_model.ckpt`.

Then, run the following command if using CPU. **Make sure** the meta file has the column
`path` (path to the sample).

```bash
python -m scoring.nsfw.inference /path/to/meta.csv
```

If running on Ascend, you may use

```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/scoring/nsfw/inference.py \
 /path/to/meta.csv
```

This should output `/path/to/meta_nsfw.csv` with column `nsfw`.


## Captioners

Human labeling of videos is expensive and time-consuming.
We adopt powerful image captioning models to generate
captions for videos. We support LLaVA, PLLaVA, and Qwen2-VL Captioning.

### LLaVA Captioning
LLaVA is utilized in the HunyuanVideo-I2V model. For usage,
 you may download the model [here](https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers)
 and put it under `./pretrained_models/llava-llama-3-8b-v1_1-transformers`. You can
 also use a customized directory by using the option
 `--pretrained_model_name_or_path` when running the script.

Currently, we only support captioning on Ascend.

```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/captioning/caption_llava.py \
 /path/to/meta.csv
```
Modify `worker_num` and `local_worker_num` based on your resource.

We support the following arguments for LLaVA captioning:

- `pretrained_model_name_or_path`: the LLaVA model directory.
- `question`: The prompt used to generate captions. Default "Describe the video in detail".
- `max_new_tokens`: Maximum new tokens generated by LLaVA, default 200.
- `bs`: Batch size. Default 1.
- `skip_if_existing`: Skip processing if output already exists. Default False.

### PLLaVA Captioning
HPC-AI captioned their training videos
with the [PLLaVA](https://github.com/magic-research/PLLaVA) model.
PLLaVA performs highly competitively on multiple
video-based text generation benchmarks including
[MVbench](https://paperswithcode.com/sota/video-question-answering-on-mvbench?p=pllava-parameter-free-llava-extension-from-1).

| Model      | Link                                                     |
| ------------ |----------------------------------------------------------|
| pllava-7b  | [pllava-7b · Hugging Face](https://huggingface.co/ermu2001/pllava-7b)   |
| pllava-13b | [pllava-13b · Hugging Face](https://huggingface.co/ermu2001/pllava-13b) |
| pllava-34b | [pllava-34b · Hugging Face](https://huggingface.co/ermu2001/pllava-34b) |

And put it under `./pretrained_models/pllava`. You can
also use a customized directory by using the option
`--pretrained_model_name_or_path` when running the script.

Currently, we only support captioning on Ascend.

```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/captioning/caption_pllava.py \
 /path/to/meta.csv
```
Modify `worker_num` and `local_worker_num` based on your resource.

We support the following arguments for PLLaVA captioning:

- `pretrained_model_name_or_path`: the PLLaVA model directory.
- `num_frames`: the number of frames to extract from the videos. Default 4.
- `question`: The prompt used to generate captions. Default "Describe the video in detail".
- `max_new_tokens`: Maximum new tokens generated by PLLaVA, default 200.
- `bs`: Batch size. Default 1.
- `skip_if_existing`: Skip processing if output already exists. Default False.

### Qwen2-VL Captioning

We recommend using Qwen2-VL captioning by default. Empirically, we observe that Qwen2-VL
consistently provide high-quality captions for videos.
Qwen2-VL-72B model showcases top-tier performance across
most metrics, often surpassing even closed-source
models like GPT-4o and Claude-3.5-Sonnet.

First, you may download the model on HuggingFace and put it under `./pretrained_models/Qwen2-VL-7B-Instruct`.:

| Model        | Link                                                                         |
|--------------|------------------------------------------------------------------------------|
| Qwen2-VL-2B-Instruct  | [Qwen/Qwen2-VL-2B · Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct)   |
| Qwen2-VL-7B-Instruct  | [Qwen/Qwen2-VL-7B · Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)   |
| Qwen2-VL-72B-Instruct | [Qwen/Qwen2-VL-72B · Hugging Face](https://huggingface.co/Qwen/Qwen2-VL-72B-Instruct) |


Currently, we only support captioning on Ascend.

```bash
export PYTHONPATH=$(pwd)
msrun --worker_num=2 --local_worker_num=2 --join=True \
 --log_dir=msrun_log pipeline/captioning/caption_qwen2vl.py \
 /path/to/meta.csv
```
Modify `worker_num` and `local_worker_num` based on your resource.

We support the following arguments for Qwen2-VL captioning:

- `pretrained_model_name_or_path`: the Qwen2-VL model directory.
- `question`: The prompt used to generate captions. Default "Describe the video in detail".
- `max_new_tokens`: Maximum new tokens generated by Qwen2-VL, default 200.
- `height`: Resized video height. Default 448.
- `width`: Resized video width. Default 672.
- `fps`: Frame rate of the resized video. Default 4.
- `bs`: Batch size. Default 1.
- `skip_if_existing`: Skip processing if output already exists. Default False.

**NOTE:** When running large-scale parallel inference,
the default `HCCL_CONNECT_TIMEOUT` setting might be
insufficient, potentially causing runtime errors with
AllGather operations. To avoid this issue, consider
setting `export HCCL_CONNECT_TIMEOUT=7200` (corresponds to
7200 seconds) or adjusting it according to your
specific needs.

Additionally, an empty caption will be generated if the
input exceeds the memory usage limit. You may try to
reduce the height, width, or fps to avoid this issue.

## Acknowledgement
This pipeline for video/image data processing pipeline in MindSpore is mostly
based on the [work](https://github.com/hpcaitech/Open-Sora/blob/main/docs/data_processing.md) by HPC-AI OpenSora. We thank them for their generous
support of the open source community.
