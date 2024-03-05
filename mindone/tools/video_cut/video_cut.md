# Video Cut

## Introduction
Video cut is a tool for processing videos, which can be used to generate video keyframes and cut video scenes.


## Usage

You can generate video keyframes or cut video scenes by running `video_cut.py`

### Generate video keyframes

```
python mindone/tools/video_cut/video_cut.py --video_data_path <path-to-videos> --keyframe_save_dir <path-to-save-keyframes> --task keyframe
```
> Note: `keyframe_save_dir` defaults to None. If you do not set `keyframe_save_dir`, keyframes will be saved in the `video_data_path` folder

### cut video scene

```
python mindone/tools/video_cut/video_cut.py --video_data_path <path-to-videos> --task scene
```

For more usage, please run `python mindone/tools/video_cut/video_cut.py -h`.
