
This folder contains code and scripts for diffusion model evaluation, e.g.,

- CLIP Score For frame Consistency
- CLIP Score for Textual Alignment


Note that all the above metrics are computed based on neural network models.

> A convincing evaluation for diffusion models requires both visually qualitative comparision and quantitative measure. A higher CLIP score does not necessarily show one model is better than another.

## Usage

### CLIP Score for Frame Consistency

To compute the CLIP score on all frames of output video and report the average cosine similarity between all video frame pairs, please run

- Mindspore backend
```
export PYTHONPATH="your-mindone-path"
python mindone/tools/eval/eval_clip_score_video.py --video_data_dir <path-to-video-dir> --video_caption_path <path-to-video-caption-path> --ckpt_path <path-to-model>  --metric clip_score_frame
```
- PyTorch backend
```
export PYTHONPATH="your-mindone-path"
python mindone/tools/eval/eval_clip_score_video.py --backend pt --mode_name <HF-model-name> --video_data_dir <path-to-video-dir> --video_caption_path <path-to-video-caption-path> --ckpt_path <path-to-model>  --metric clip_score_frame
```

### CLIP Score for Textual Alignment

To compute the average CLIP score between all frames of the output video and the corresponding editing prompts, please run

- Mindspore backend
```
export PYTHONPATH="your-mindone-path"
python mindone/tools/eval/eval_clip_score_video.py --video_data_dir <path-to-video-dir> --video_caption_path <path-to-video-caption-path> --ckpt_path <path-to-model>  --metric clip_score_text
```
- PyTorch backend
```
export PYTHONPATH="your-mindone-path"
python mindone/tools/eval/eval_clip_score_video.py --backend pt --mode_name <HF-model-name> --video_data_dir <path-to-video-dir> --video_caption_path <path-to-video-caption-path> --ckpt_path <path-to-model>  --metric clip_score_text
```


By default, we use MindSpore backend for CLIP score computing (to run CLIP model inference and extract image & text features). You may swich to use `torch` by setting `--backend=pt`. The computational difference between these two backends is usually lower than 0.1%, which is neglectable.

For more usage, please run `python mindone/tools/eval/eval_clip_score_video.py -h`.

You need to download the checkpoint file for a CLIP model of your choice. Download links for some models are provided below.

- [clip_vit_b_16](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_b_16.ckpt)
- [clip_vit_b_32](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/XFormer_for_mindspore/clip/clip_vit_b_32.ckpt)
- [clip_vit_l_14](https://ascend-repo-modelzoo.obs.cn-east-2.myhuaweicloud.com/MindFormers/clip/clip_vit_l_14.ckpt) (Default)

For other compatible models, e.g., OpenCLIP, you can download `pytorch_model.bin` from HuggingFace (HF) and then convert to `.ckpt` using `tools/_common/clip/convert_weight.py`. When using a model other than the default, you should supply the path to your model's config file. Some useful examples are provided in `tools/_common/clip/configs`.

`video_data_dir` should lead to a directory containing videos. `video_caption_path` can be the path to an `.csv` file, which format is as follows: videos and captions are matched such that each caption corresponding to one video.

Format of `.csv`:
```
video,caption
video_name1.mp4,"an airliner is taxiing on the tarmac at Dubai Airport"
video_name2.mp4,"a pigeon sitting on the street near the house"
...
```



## Reference

[1] https://github.com/showlab/loveu-tgve-2023/tree/main
