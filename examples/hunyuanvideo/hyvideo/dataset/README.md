# HunyuanVideo Datasets

## 📚 Dataset Format

Similar to [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan), we use text-video-pair datasets to train HunyuanVideo.

The dataset consists of two parts: the input video folder and the annotation json file. The input video folder contains different videos, and the annotation json file contains the corresponding captions and other meta information, such as resolution, fps, and number of frames.

An input video folder should be organized as follows:
```bash
videos/
    Dogs/
        mixkit-a-panting-border-collie-patiently-holds-a-ball-on-its-50684.mp4
        ...
    Pets/
        mixkit-a-panting-border-collie-patiently-holds-a-ball-on-its-50684.mp4
        ...
    ...
```
The annotation json file should be like:

```json
[
    {
        "path": "Dogs/mixkit-a-panting-border-collie-patiently-holds-a-ball-on-its-50684.mp4",
        "resolution": {
            "height": 1080,
            "width": 1920
        },
        "cap": "The video features a black and white dog sitting upright against a plain background, consistently holding a colorful, textured toy in its mouth throughout the sequence. The dog's fur is well-groomed, and it is adorned with a blue collar or leash attachment around its neck. The dog maintains a friendly and playful demeanor, with no significant changes in its expression, posture, or the surrounding environment observed between the frames. The overall scene captures a cheerful and playful moment centered on the dog.",
        "fps": 23.976023976023978,
        "num_frames": 246
    },
    ...
]
```

The `path` field in the annotation json file is relative to the input video folder.
