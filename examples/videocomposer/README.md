# VideoComposer based on MindSpore

MindSpore implementation & optimization of [VideoComposer: Compositional Video Synthesis with Motion Controllability](https://arxiv.org/pdf/2306.02018.pdf).

## Main Features

- [x] Conditional Video Generation including the following tasks:
    - [x] Motion transfer from a video to a single image (exp02)
    - [x] Single sketch to videos with or without style guidance (exp03 and exp04)
    - [x] Depth to video with or without style guidance (exp5 and exp6)
    - [x] Genearte videos basd on multiple conditions:depth maps, local image, masks, motion, and sketch
- [x] Model Training (vanilla finetuning) supporting both Ascend 910A and 910B
- [x] Acceleration and Memeory Reduction
    - [x] Mixed Precision
    - [x] Graph Mode for Training
    - [x] Recompute

### TODOs
- Speed Up & Memory Usage Reduction (e.g., Flash Attention)
- Support more training tasks
- Support more training features: EMA, Gradient accumulation, Gradient clipping
- More effieicent online inference
- 910B + Lite inference
- Evaluation

## Installation

1. Create virtual environment
    ```shell
    conda create -n ms2.0 python=3.9
    conda activate ms2.0
    ```

2. Install requirements
    ```shell
    pip install -r requirements.txt
    ```

    Install `ffmpeg` by
    ```shell
    conda install ffmpeg
    ```

    If case you fail to install `motion-vector-extractor` via pip, please manually install it referring to the [official](https://github.com/LukasBommes/mv-extractor) repo.

## Prepare Pretrained Weights

The root path of downloading must be `${PROJECT_ROOT}\model_weights`, where `${PROJECT_ROOT}` means the root path of project.

Download the checkpoints shown in model_weights/README.md from https://download.mindspore.cn/toolkits/mindone/videocomposer/model_weights/ and https://download.mindspore.cn/toolkits/mindone/stable_diffusion/depth_estimator/midas_v3_dpt_large-c8fd1049.ckpt

## Inference

To run all video generation tasks, please run

```shell
bash run_net.sh
```

To run a single task, you can pick the corresponding snippet of code in `run_net`.sh, such as

```shell
python run_net.py\
    --cfg configs/exp02_motion_transfer_vs_style.yaml\
    --seed 9999\
    --input_video "demo_video/motion_transfer.mp4"\
    --image_path "demo_video/moon_on_water.jpg"\
    --style_image "demo_video/moon_on_water.jpg"\
    --input_text_desc "A beautiful big silver moon on the water"
```

It takes additional time for graph compilation to execute the first step inference (around 5~8 minutes).

### Key arguments for inference

You can adjust the arguemnts in `vc/config/base.py` (lower-priority) or `configs/exp{task_name}.yaml` (higher-priority, will overwrite base.py if overlap). Below are the key arguments influencing inference speed and memory usage.

- use_fp16: whether enable mixed precision inference
- mode: 0 for use graph mode,  1 for pynative mode


## Training

### Standalone Training
To run training on a sepecifc task, please run

```
python train.py --cfg configs/train{task_name}.yaml
```

E.g. `python train.py configs/train_exp02_motion_style.yaml `


### Distributed Training

Please generate the hccl config file on your running server at first referring to [this tutorial](https://github.com/mindspore-lab/mindocr/blob/main/docs/cn/tutorials/distribute_train.md#12-%E9%85%8D%E7%BD%AErank_table_file%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83). Then update `run_train_distribute.sh` by setting
```
rank_table_file=path/to/hccl_8p_01234567_xxx.json
```

Then execute,
```
bash run_train_distribute.sh
```

###  Key arguemnts for training

You can adjust the arguemnts in `configs/train_base.py` (lower-priority) or `configs/train_exp{task_name}.yaml` (higher-priority, will overwrite train_base.py if overlap). Below are the key arguments.

- max_frames: number of frames to generate for each sample. Without memory reduction tricks, it can be set  up to 8 for 910A (30GB memory), and 16 for 910B (60GB memory) for task-2 finetuning.
- optim: optimizer name, `adamw` or `momentum`. Recommend `momentum` for 910A to avoid OOM and `adamw` for 910B for better loss convergence.
- use_recompute: by enabling it, you can reduce memory usage with a small increase of time cost. For example, on 910A, the max number of trainable frames per batch increases from 8 to 14 after recompute enabled.
- `root_dir`: dataset root dir which should contains a csv annotation file. default is `demo_video`, which contains an example annotation file `demo_video/video_caption.csv` for demo traning.
