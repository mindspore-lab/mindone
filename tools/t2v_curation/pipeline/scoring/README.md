# Scoring and Filtering

- [Scoring and Filtering](#scoring-and-filtering)
  - [Aesthetic Score](#aesthetic-score)
  - [Matching Score](#matching-score)
  - [OCR](#OCR)
  - [LPIPS Motion Analysis](#lpips-score-motion-analysis)
  - [NSFW](#NSFW)
  - [Filtering](#filtering)

## Aesthetic Score

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

## Matching Score

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

## OCR
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

## LPIPS Score (Motion Analysis)
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

## NSFW

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

## Filtering
Once scores are obtained, it is simple to filter samples based on these scores. Here is an example to remove
samples of aesthetic score < 4.0.
```
python -m pipeline.datasets.datautil /path/to/meta.csv --aesmin 4
```
This should output `/path/to/meta_aesmin4.0.csv` with column `aes` >= 4.0

[Here](../datasets/README.md) you may find more filtering options.
