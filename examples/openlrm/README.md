# OpenLRM: Open-Source Large Reconstruction Models
This is the mindspore re-implementation of the work [LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) based on open source repo [OpenLRM v1.1.1](https://github.com/3DTopia/OpenLRM).


<!--

<div style="text-align: left">
    <img src="assets/mesh_snapshot/crop.owl.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.owl.ply01.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.building.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.building.ply01.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.rose.ply00.png" width="12%" height="auto"/>
    <img src="assets/mesh_snapshot/crop.rose.ply01.png" width="12%" height="auto"/>
</div> -->

## **Introduction**
<p align="center">
  <img src="https://github.com/user-attachments/assets/4f5d4325-1fd5-4421-a958-9a11ecee0192"  width="70%" height="auto">
</p>
LRM is the first Large Reconstruction Model that predicts the 3D model of an object from a single input image within just few seconds. In contrast to many previous methods that are trained on small-scale datasets such as ShapeNet in a category-specific fashion, LRM adopts a highly scalable transformer-based architecture with 500 million learnable parameters to directly predict a neural radiance field (NeRF) from the input image. LRM model was trained in an end-to-end manner on massive multi-view data containing around 1 million objects, including both synthetic renderings from Objaverse and real captures from MVImgNet. This combination of a high-capacity model and large-scale training data empowers our model to be highly generalizable and produce high-quality 3D reconstructions from various testing inputs, including real-world in-the-wild captures and images created by generative models.

## Get Started
### Requirements
|mindspore |	ascend driver | firmware | cann tookit/kernel|
|--- | --- | --- | --- |
|2.3.1 | 24.1RC2 | 7.3.0.1.231 | 8.0.RC2.beta1|

### Dependencies
- Install requirements for OpenLRM.
  ```
  pip install -r requirements.txt
  ```

## Inference

### Prepare Images
- Prepare RGBA images or RGB images with white background (with some background removal tools, e.g., [Rembg](https://github.com/danielgatis/rembg), [Clipdrop](https://clipdrop.co)).

### Pretrained Models

- Model weights are released on [Hugging Face](https://huggingface.co/zxhezexin).
- We suggest to download checkpoints and configs locally, though weights will be downloaded automatically when you run the inference script for the first time.
  ```
  weight_dir
  ├── config.json
  └── model.safetensors
  ```

- Please be aware of the [license](LICENSE_WEIGHT) before using the weights.

### Run Inference on Pretrained Models
- Run the inference script for image-to-3D.
- You may specify which form of output to generate by setting the flags `EXPORT_VIDEO=true` and `EXPORT_MESH=true`.
- Please set default `INFER_CONFIG` according to the model you want to use: `infer-s.yaml` for small models, `infer-b.yaml` for base models and `infer-l.yaml` for large models.
- An example usage is as follows:

  ```
  # Example usage
  EXPORT_VIDEO=true
  EXPORT_MESH=true
  INFER_CONFIG="./configs/infer-b.yaml"
  MODEL_NAME="zxhezexin/openlrm-mix-base-1.1"
  IMAGE_INPUT="./assets/sample_input/owl.png"

  python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
  ```


## Training

### Configuration
- We provide training sample config files under `configs/train-sample-X.yaml`, which defaults to use 1 NPU with `fp32` precision in PYNATIVE_MODE.
- `configs/train-sample-large.yaml` can use 1 NPU with `bf16` mixed precision for training.
- Currently we only support loading Objaverse data curated by script in `scripts/data/objaverse/blender_script.py`.

  | Model | Dataset | Layers | Feat. Dim | Trip. Dim. | In. Res. |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | small | Objaverse | 12 | 512 | 32 | 224 |
  | base | Objaverse | 12 | 768 | 48 | 336 |
  | large | Objaverse | 16 | 1024 | 80 | 448 |


### Data Preparation
- We provide the core [Blender script](scripts/data/objaverse/blender_script.py) used to render Objaverse images.
- Please refer to [Objaverse Rendering](https://github.com/allenai/objaverse-rendering) for other scripts including distributed rendering.
- Objaverse dataset format:

  ```
  objaverse_dataset
  ├── meta.json
  ├── <UID#1>
  │   ├── pose
  │   │   ├── 000.npy
  │   │   ├── 001.npy
  │   │   ├── 002.npy
  │   │   ├── ...
  │   │   └── 031.npy
  │   ├── rgba
  │   │   ├── 000.png
  │   │   ├── 001.png
  │   │   ├── 002.png
  │   │   ├── ...
  │   │   └── 031.png
  │   └── intrinsics.npy
  ├── <UID#2>
  │   ├── pose
  │   ├── rgba
  │   └── intrinsics.npy
  ├── ...
  └── <UID#N>
      ├── pose
      ├── rgba
      └── intrinsics.npy
  ```

### Run Training
- Sample training config files are provided under `configs/train-sample-X.yaml`.
- Please replace data related paths in the config file with your own paths and customize the training settings.
- Some training sample scripts are provided under `scripts/train_X.sh`. An example training usage is as follows:

  ```
  # Example usage
  export MS_INDEPENDENT_DATASET=True
  export MS_ALLOC_CONF="enable_vmm:True;"
  export TRAIN_CONFIG="./config/train-sample-small.yaml"
  export DEVICE_ID=0
  EPOCH=100000
  OUTPUT_PATH=./outputs/train_small_e$EPOCH
  python -m openlrm.launch train.lrm --config $INFER_CONFIG --mode 1 --num_parallel_workers 1 --epochs $EPOCH
  ```

### Run Inference on Trained MindSpore Models
- A inference sample script is provided in `scripts/infer_ms.sh` to infer 3D assets and render videos by trained MindSpore checkpoints.
- An example inference usage is as follows:

  ```
  # Example usage
  EXPORT_VIDEO=true
  EXPORT_MESH=true
  INFER_CONFIG="./configs/infer-s.yaml"
  MODEL_NAME="outputs/small/2024-12-30T18-24-53"
  IMAGE_INPUT="sample_input"

  epoch=100000
  MODEL_CKPT=openlrm-e${epoch}.ckpt
  DEVICE_ID=0 python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME model_ckpt=$MODEL_CKPT epoch=$epoch image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
  ```
## Evaluation
### Evaluation Data Preparation
- Same format as [Data Preparation](#data-preparation).

### Run Evaluation on Trained Checkpoint
- Evaluation script `eval.py` input first image and camera information as input, and based on other camera information output other views of 3D model.
- Set evaluation data path in `MODEL_PATH/cfg.yaml`:
  ```
  dataset:
      subsets:
          -   name: objaverse
              meta_path:
                  val: <REPLACE_WITH_VAL_UIDS_IN_JSON>  # e.g. val_dataset/meta.json
  ```
- Compute and report inference speed, PSNR and SSIM at `OUTPUT_PATH/EPOCH/log_0.txt`.
- A sample evalution script is in `scripts/eval.sh`. An example is as follows:
  ```
  # Example usage
  MODEL_PATH=outputs/base/2025-01-06T14-37-07
  CKPT_NAME=openlrm-e50000.ckpt
  OUTPUT_PATH=output/base
  DEVICE_ID=0 python eval.py --model_path $MODEL_PATH --ckpt_name $CKPT_NAME --output_path $OUTPUT_PATH
  ```

## Performace
Experiments are tested on 1 ascend 910* with mindSpore 2.3.1 pynative mode.


### Inference Performance
- Input a single image, here reports speed of image-to-3D and image rendering.

| model name|  cards| batch size | resolution | precision | jit level| flash attn| (image to 3D) s/step | (render) img/s|  recipe| weight|
|---|---|---|---|---|---|---|---|---|---|---|
|openlrm-mix-base-1.1 |  1 | 1 | 336x336 |fp32| O0| OFF |   0.61 | 0.91 |[yaml](./configs/infer-b.yaml)| [weight](https://huggingface.co/zxhezexin/openlrm-mix-base-1.1)|

<!-- |Input| Output |
|:---|:---|
| | | -->

### Training Performance
- Train with objaverse data only:

|model name	| cards	| batch size	| resolution	| precision| recompute	| loss scaler	| jit level	| s/step	| batch/s|
|---|---|---|---|---|---|---|---|---|---|
|small | 1 | 1 | (4x)224x224 | fp32 | OFF | NONE | O0 | 2.00 | 0.50 |
|base | 1 | 1 | (4x)336x336 | fp32 | OFF | NONE | O0 | 2.63 | 0.38 |
|large | 1 | 1 | (4x)448x448 | bf16 | ON | static | O0 | 4.22 | 0.24 |  


- Evaluation on trained models:

|model name	| epoch| cards	| batch size	| resolution	| precision| jit level	| (infer+render) s/step	| PSNR | SSIM |
|---|---|---|---|---|---|---|---|---|---|
|small | 70k |1 | 1 | 224x224 | fp32  | O0 | 0.99 | 23.75 | 0.88  
|base | 40k |1 | 1 | 336x336 | fp32  | O0 | 1.90 | 20.79 | 0.86  

## License

- OpenLRM as a whole is licensed under the [Apache License, Version 2.0](LICENSE), while certain components are covered by [NVIDIA's proprietary license](LICENSE_NVIDIA). Users are responsible for complying with the respective licensing terms of each component.
- Model weights are licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE_WEIGHT). They are provided for research purposes only, and CANNOT be used commercially.
