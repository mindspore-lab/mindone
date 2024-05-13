# Utility Scripts

This folder is a collection of utility scripts, listed and explained below.

> All scripts need to be run in the root path of project, unless otherwise noted.

## eval.py

This script contains code and scripts for diffusion model evaluation, e.g.,

- CLIP Score For frame Consistency
- CLIP Score for Textual Alignment
- Fréchet video distance (FVD)
- Kernel video distance (FVD)


Note that all the above metrics are computed based on neural network models.

> A convincing evaluation for diffusion models requires both visually qualitative comparision and quantitative measure. A higher CLIP score does not necessarily show one model is better than another.


#### CLIP Score for Frame Consistency

To compute the CLIP score on all frames of output video and report the average cosine similarity between all video frame pairs, please run

```shell
python ./scripts/video_eval/eval.py --gen_video_dir <path-to-video-dir> --gen_csv_path <path-to-video-caption-path> --model_name <HF-model-name>  --metric clip_score_frame
```

#### CLIP Score for Textual Alignment

To compute the average CLIP score between all frames of the output video and the corresponding editing prompts, please run

```shell
python ./scripts/video_eval/eval.py --gen_video_dir <path-to-video-dir> --gen_csv_path <path-to-video-caption-path> --model_name <HF-model-name>  --metric clip_score_text
```

Format of `.csv`:
```
video,caption
video_name1.mp4,"an airliner is taxiing on the tarmac at Dubai Airport"
video_name2.mp4,"a pigeon sitting on the street near the house"
...
```

#### FVD and KVD

FVD and KVD measure how close a distribution of generated videos is to the ground-truth distribution. To compute the FVD and KVD between the real videos and the generated videos, please run

```
python ./scripts/video_eval/eval.py --gt_video_dir {dir_to_real_videos}  --gen_video_dir {dir_to_generated_videos} --sample_n_frames 64 --sample_stride 2 --metric FVD
```

> The default ckpt is None, it will automatically download the InceptionI3d checkpoint file for FVD and KVD. If you fail to downalod due to network problem, please manually download it from [this link](https://download.mindspore.cn/toolkits/mindone/stable_diffusion/fvd/inception_i3-02b0bb54.ckpt).

For more usage, please run `python ./scripts/video_eval/eval.py -h`.

## Reference

[1] https://github.com/showlab/loveu-tgve-2023/tree/main
[2] https://github.com/YingqingHe/LVDM/tree/main
