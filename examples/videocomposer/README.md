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

## Prepare Training Data
The training videos and their captions (.txt) should be placed in the following folder struture.
```
 ├── {DATA_DIR}
 │   ├── video_name1.mp4
 │   ├── video_name1.txt
 │   ├── video_name2.mp4
 │   ├── video_name2.txt
 │   ├── ...
```
Run `examples/videocomposer/tools/data_converter.py` to generate `video_caption.csv` in `{DATA_DIR}`.
```
python data_converter.py {DATA_DIR}
```
Format of `video_caption.csv`:
```
video,caption
video_name1.mp4,"an airliner is taxiing on the tarmac at Dubai Airport"
video_name2.mp4,"a pigeon sitting on the street near the house"
...
```

## Inference

### Online Inference

To run all video generation tasks on 910A or 910B, please run

```shell
bash run_infer.sh
```

On 910A, to run a single task, you can pick the corresponding snippet of code in `run_infer.sh`, such as

```shell
# export MS_ENABLE_GE=1  # for 910B
python infer.py \
    --cfg configs/exp02_motion_transfer_vs_style.yaml \
    --seed 9999 \
    --input_video "demo_video/motion_transfer.mp4" \
    --image_path "demo_video/moon_on_water.jpg" \
    --style_image "demo_video/moon_on_water.jpg" \
    --input_text_desc "A beautiful big silver moon on the water"
```

On 910B, you need enable the GE Mode first by running `export MS_ENABLE_GE=1`.

It takes additional time for graph compilation to execute the first step inference (around 5~8 minutes).

### Key arguments for inference

You can adjust the arguemnts in `vc/config/base.py` (lower-priority) or `configs/exp{task_name}.yaml` (higher-priority, will overwrite base.py if overlap). Below are the key arguments influencing inference speed and memory usage.

- use_fp16: whether enable mixed precision inference

### Offline (Mindspore Lite) Inference

#### Install Mindspore Lite
You need to have a Mindspore Lite Environment first for offline inference.

To install Mindspore Lite, please Refer to [Lite install](https://mindspore.cn/lite/docs/zh-CN/r2.1/use/downloads.html)

1. Download the supporting tar.gz and whl packages according to the environment.
2. Unzip the tar.gz package and install the corresponding version of the WHL package.

   ```shell
   tar -zxvf mindspore-lite-2.1.0-*.tar.gz
   pip install mindspore_lite-2.1.0-*.whl
   ```

3. Configure Lite's environment variables

   `LITE_HOME` is the folder path extracted from tar.gz, and it is recommended to use an absolute path.

   ```shell
   export LITE_HOME=/path/to/mindspore-lite-{version}-{os}-{platform}
   export LD_LIBRARY_PATH=$LITE_HOME/runtime/lib:$LITE_HOME/tools/converter/lib:$LD_LIBRARY_PATH
   export PATH=$LITE_HOME/tools/converter/converter:$LITE_HOME/tools/benchmark:$PATH


#### Export Mindspore Lite Model

For different task, you can use the corresponding snippet of the code in `run_infer.sh`, and change `infer.py` to `export.py` to save the MindIR model. Please remember to run `export MS_ENABLE_GE=1` first on 910B before running the code snippet.

```shell
# export MS_ENABLE_GE=1  # for 910B
python export.py\
    --cfg configs/exp02_motion_transfer_vs_style.yaml \
    --input_video "demo_video/motion_transfer.mp4" \
    --image_path "demo_video/moon_on_water.jpg" \
    --style_image "demo_video/moon_on_water.jpg" \
    --input_text_desc "A beautiful big silver moon on the water"
```

The exported MindIR models will be saved at `models/mindir` directory. Once the exporting is finished, you need to convert the MindIR model to Mindspore Lite MindIR model. We have provided a script `convert_lite.py` to convert all MindIR models in `models/mindir` directory. And please note that on 910B, you need to unset `MS_ENABLE_GE` environmental variable befor running the conversion.

```shell
unset MS_ENABLE_GE  # Remember to unset MS_ENABLE_GE on 910B
python convert_lite.py
```

#### Inference using the exported Lite models.

Then you can run the offline inference using the `infer_lite.py` for the given task, e.g,

```shell
python infer_lite.py\
    --cfg configs/exp02_motion_transfer_vs_style.yaml \
    --seed 9999 \
    --input_video "demo_video/motion_transfer.mp4" \
    --image_path "demo_video/moon_on_water.jpg" \
    --style_image "demo_video/moon_on_water.jpg" \
    --input_text_desc "A beautiful big silver moon on the water" \
```

The compiling time is much shorter compared with the online inference mode.


## Training

### Standalone Training:
To run training on a specific task, please refer to `run_train.sh`:

```bash
export GLOG_v=2  # Log message at or above this level. 0:INFO, 1:WARNING, 2:ERROR, 3:FATAL
export HCCL_CONNECT_TIMEOUT=6000
export ASCEND_GLOBAL_LOG_LEVEL=1  # Global log message level for Ascend. Setting it to 0 can slow down the process
export ASCEND_SLOG_PRINT_TO_STDOUT=0 # 1: detail, 0: simple
export DEVICE_ID=0  # The device id to runing training on

task_name=train_exp02_motion_transfer  # the default training task
yaml_file=configs/${task_name}.yaml
output_path=outputs
rm -rf ${output_path:?}/${task_name:?}
mkdir -p ${output_path:?}/${task_name:?}
export MS_COMPILER_CACHE_PATH=${output_path:?}/${task_name:?}

nohup python -u train.py  \
     -c $yaml_file  \
     --output_dir $output_path/$task_name \
    > $output_path/$task_name/train.log 2>&1 &

```
Under `configs/`, we provide several tasks' yaml files:
```bash
configs/
├── train_exp02_motion_transfer_vs_style.yaml
├── train_exp02_motion_transfer.yaml
├── train_exp03_sketch2video_style.yaml
├── train_exp04_sketch2video_wo_style.yaml
├── train_exp05_text_depths_wo_style.yaml
└── train_exp06_text_depths_vs_style.yaml
```

Taking `configs/train_exp02_motion_transfer.yaml` as an example, there is one critical argument:
```yaml
video_compositions: ['text', 'mask', 'depthmap', 'sketch', 'single_sketch', 'motion', 'image', 'local_image']
```
`video_compositions` defines all available conditions:
- `text`: the text embedding.
- `mask`: the masked video frames.
- `depthmap`: the depth images extracted from visual frames.
- `sketch`: the sketch images extracted from visual frames.
- `single_sketch`: the first sketch image from `sketch`.
- `motion`: the motion vectors extracted from the training video.
- `image`: the image embedding used as an image style vector.
- `local_image`: the first frame extracted from the training video.

However, not all conditions are included in the training process in each of tasks above. As defined in `configs/train_exp02_motion_transfer.yaml`,

```yaml
conditions_for_train: ['text', 'local_image', 'motion']
```
`conditions_for_train` defines the three conditions used for training which are `['text', 'local_image', 'motion']`.

### Distributed Training

Please generate the hccl config file on your running server at first referring to [this tutorial](https://github.com/mindspore-lab/mindocr/blob/main/docs/cn/tutorials/distribute_train.md#12-%E9%85%8D%E7%BD%AErank_table_file%E8%BF%9B%E8%A1%8C%E8%AE%AD%E7%BB%83). Then update `run_train_distribute.sh` by setting
```
rank_table_file=path/to/hccl_8p_01234567_xxx.json
```

After that, please set `task_name` according to your target task. The default training task is `train_exp02_motion_transfer`.

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
